"""Command-line entry point: ``slidescribe meeting.mp4``."""

from __future__ import annotations

import argparse
import os
import sys

from slidescribe.config import Config, LLMConfig
from slidescribe.pipeline import process_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slidescribe",
        description="Turn a screen-shared recording into an illustrated PDF transcript.",
        epilog=(
            "Runs with no API key. Add --model to enable slide titles, "
            "smarter slide detection, and a summary."
        ),
    )
    p.add_argument("video", help="Path to the input video.")
    p.add_argument("-o", "--output", help="Output PDF path.")
    p.add_argument("--title", help="Document title.")

    g = p.add_argument_group("transcription")
    g.add_argument(
        "--whisper-model", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper size. Default: small.",
    )
    g.add_argument("--language", help='Force a language code, e.g. "en".')
    g.add_argument("--device", choices=["cuda", "cpu"], help="Force a device.")

    g = p.add_argument_group("slide detection")
    g.add_argument("--sample-fps", type=float, default=1.0,
                   help="Frames sampled per second. Default: 1.0.")
    g.add_argument("--hash-threshold", type=int, default=6,
                   help="pHash distance for a change. Lower is more sensitive.")
    g.add_argument("--ssim-threshold", type=float, default=0.95,
                   help="SSIM below which frames differ. Higher is more sensitive.")
    g.add_argument("--min-slide-seconds", type=float, default=4.0,
                   help="Minimum gap between captures.")
    g.add_argument("--crop", help='Shared-screen region as "l,t,r,b" fractions, e.g. "0,0,0.75,1".')
    g.add_argument("--keep-frames", action="store_true",
                   help="Keep extracted slide images beside the PDF.")

    g = p.add_argument_group("LLM (all optional)")
    g.add_argument("--model", help='LiteLLM model string, e.g. "gpt-4o", "claude-sonnet-4-5", "gemini/gemini-2.0-flash".')
    g.add_argument("--api-key", help="Provider key. Falls back to the provider's env var.")
    g.add_argument("--api-base", help="Custom endpoint, e.g. http://localhost:11434 for Ollama.")
    g.add_argument("--max-vision-calls", type=int, default=60,
                   help="Ceiling on image requests per video.")
    g.add_argument("--clean-transcript", action="store_true",
                   help="Punctuate and paragraph the transcript. Uses more tokens.")
    g.add_argument("--no-summary", action="store_true", help="Skip summary and action items.")
    g.add_argument("--no-captions", action="store_true", help="Skip slide titles.")
    g.add_argument("--no-arbitrate", action="store_true",
                   help="Skip vision confirmation of ambiguous slide changes.")

    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    crop = None
    if args.crop:
        try:
            parts = tuple(float(x) for x in args.crop.split(","))
            if len(parts) != 4:
                raise ValueError
            crop = parts
        except ValueError:
            print('--crop must be four numbers, e.g. "0,0,0.75,1"', file=sys.stderr)
            return 2

    llm = LLMConfig(
        model=args.model or os.environ.get("SLIDESCRIBE_MODEL") or None,
        api_key=args.api_key or os.environ.get("SLIDESCRIBE_API_KEY") or None,
        api_base=args.api_base or os.environ.get("SLIDESCRIBE_API_BASE") or None,
        max_vision_calls=args.max_vision_calls,
    )

    config = Config(
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        sample_fps=args.sample_fps,
        hash_threshold=args.hash_threshold,
        ssim_threshold=args.ssim_threshold,
        min_slide_seconds=args.min_slide_seconds,
        crop=crop,
        llm=llm,
        arbitrate_slides=not args.no_arbitrate,
        caption_slides=not args.no_captions,
        clean_transcript=args.clean_transcript,
        summarize=not args.no_summary,
        title=args.title,
        keep_frames=args.keep_frames,
        verbose=not args.quiet,
    )

    try:
        result = process_video(args.video, args.output, config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.quiet:
        print(result.pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
