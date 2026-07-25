"""Speech-to-text via faster-whisper.

No LLM here. faster-whisper is a dedicated ASR model (CTranslate2 port of
OpenAI Whisper) and is the best available tool for this job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class Segment:
    """One utterance with its position in the recording."""

    start: float
    end: float
    text: str

    def timestamp(self) -> str:
        m, s = divmod(int(self.start), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def detect_device(preferred: Optional[str] = None) -> tuple:
    """Return (device, compute_type) suited to the machine we are on."""
    if preferred == "cpu":
        return "cpu", "int8"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    if preferred == "cuda":
        log.warning("CUDA requested but unavailable — falling back to CPU.")
    return "cpu", "int8"


def transcribe(video_path: str, config) -> List[Segment]:
    """Transcribe audio from a video file into timestamped segments.

    Args:
        video_path: Path to the source video. faster-whisper reads the audio
            track directly via its bundled decoder, so no manual extraction
            step is needed.
        config: A ``Config`` instance.

    Returns:
        Segments in chronological order.
    """
    from faster_whisper import WhisperModel

    device, compute_type = detect_device(config.device)
    compute_type = config.compute_type or compute_type

    model_size = config.whisper_model
    if device == "cpu" and model_size in ("medium", "large-v3", "large-v2"):
        log.warning(
            "Model '%s' on CPU will be very slow. Consider 'small' or 'base'.",
            model_size,
        )

    log.info("Loading Whisper '%s' on %s (%s)", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        video_path,
        language=config.language,
        vad_filter=config.vad_filter,
        beam_size=5,
    )

    log.info(
        "Detected language '%s' (confidence %.2f), duration %.1fs",
        info.language,
        info.language_probability,
        info.duration,
    )

    out: List[Segment] = []
    try:
        from tqdm.auto import tqdm

        bar = tqdm(total=round(info.duration), unit="s", desc="Transcribing")
    except ImportError:
        bar = None

    last_end = 0.0
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            out.append(Segment(start=seg.start, end=seg.end, text=text))
        if bar:
            bar.update(max(0, round(seg.end - last_end)))
            last_end = seg.end
    if bar:
        bar.close()

    log.info("Transcribed %d segments", len(out))
    return out


def full_text(segments: List[Segment]) -> str:
    """Flatten segments into one string."""
    return " ".join(s.text for s in segments)
