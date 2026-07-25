"""Pipeline orchestration — the one function most users call."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

from slidescribe.config import Config
from slidescribe.llm import LLMClient
from slidescribe.pdf import build_pdf
from slidescribe.slides import detect_slides
from slidescribe.transcribe import full_text, transcribe

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """What a run produced."""

    pdf_path: str
    slides: List = field(default_factory=list)
    segments: List = field(default_factory=list)
    summary: Optional[dict] = None
    seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.pdf_path} — {len(self.slides)} slides, "
            f"{len(self.segments)} segments, {self.seconds:.0f}s"
        )


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        force=True,
    )


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> ProcessResult:
    """Turn a screen-shared recording into an illustrated PDF transcript.

    The pipeline runs end to end with no API key. Supplying one in
    ``config.llm`` adds slide arbitration, slide titles, and a summary.

    Args:
        video_path: Path to the input video.
        output_path: Destination PDF. Defaults to the input name with .pdf.
        config: Pipeline settings. Sensible defaults when omitted.

    Returns:
        A ``ProcessResult`` with the PDF path and the intermediate data.

    Raises:
        FileNotFoundError: If the input video does not exist.
    """
    started = time.time()
    config = config or Config()
    _setup_logging(config.verbose)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No such video: {video_path}")

    base = os.path.splitext(os.path.basename(video_path))[0]
    output_path = output_path or f"{base}.pdf"
    title = config.title or base.replace("_", " ").replace("-", " ").title()

    llm = LLMClient(config.llm)

    frames_dir = (
        os.path.join(os.path.dirname(os.path.abspath(output_path)), f"{base}_slides")
        if config.keep_frames
        else tempfile.mkdtemp(prefix="slidescribe_")
    )

    try:
        log.info("\n[1/4] Detecting slide changes")
        slides = detect_slides(video_path, frames_dir, config, llm=llm)

        log.info("\n[2/4] Transcribing audio")
        segments = transcribe(video_path, config)

        log.info("\n[3/4] Enrichment")
        if slides and llm.vision_ready and config.caption_slides:
            try:
                from tqdm.auto import tqdm

                it = tqdm(slides, desc="Titling slides")
            except ImportError:
                it = slides
            for s in it:
                s.title = llm.caption(s.image_path)
        elif config.caption_slides and llm.enabled and not llm.vision_ready:
            log.info("Model has no vision support — slides will be numbered.")

        text = full_text(segments)

        summary = None
        if config.summarize and llm.enabled and text.strip():
            log.info("Writing summary and action items")
            summary = llm.summarize(text)

        cleaned = None
        if config.clean_transcript and llm.enabled and text.strip():
            log.info("Cleaning transcript")
            cleaned = llm.clean(text)

        log.info("\n[4/4] Building PDF")
        meta = {
            "Slides": len(slides),
            "Segments": len(segments),
            "Mode": config.llm.model if llm.enabled else "no-LLM",
        }
        build_pdf(
            output_path,
            title=title,
            segments=segments,
            slides=slides,
            summary=summary,
            cleaned_text=cleaned,
            meta=meta,
        )

        elapsed = time.time() - started
        result = ProcessResult(
            pdf_path=output_path,
            slides=slides,
            segments=segments,
            summary=summary,
            seconds=elapsed,
        )
        log.info("\nDone in %.0fs — %s", elapsed, output_path)
        return result

    finally:
        if not config.keep_frames and os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
