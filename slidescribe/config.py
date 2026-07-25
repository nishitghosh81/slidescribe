"""Configuration objects for the SlideScribe pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Provider-agnostic LLM settings.

    SlideScribe never imports a provider SDK directly. It hands these values to
    LiteLLM, which routes to OpenAI, Anthropic, Gemini, Groq, Together, Ollama,
    OpenRouter, vLLM, or anything else it supports.

    Leave ``model`` as None to run the pipeline with zero LLM calls. Everything
    mechanical (transcription, slide detection, PDF assembly) still works.

    Args:
        model: LiteLLM model string. Examples:
            "gpt-4o", "claude-sonnet-4-5", "gemini/gemini-2.0-flash",
            "groq/llama-3.3-70b-versatile", "ollama/llava".
        api_key: Provider key. Read from the matching env var if omitted.
        api_base: Override base URL. Needed for Ollama, vLLM, or any
            self-hosted OpenAI-compatible endpoint.
        vision: Whether the model accepts images. Auto-probed when None.
        max_vision_calls: Hard ceiling on image requests per video, so a long
            recording cannot quietly run up a bill.
        temperature: Sampling temperature for all calls.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    vision: Optional[bool] = None
    max_vision_calls: int = 60
    temperature: float = 0.2

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build config from SLIDESCRIBE_* environment variables."""
        return cls(
            model=os.environ.get("SLIDESCRIBE_MODEL") or None,
            api_key=os.environ.get("SLIDESCRIBE_API_KEY") or None,
            api_base=os.environ.get("SLIDESCRIBE_API_BASE") or None,
        )


@dataclass
class Config:
    """End-to-end pipeline settings.

    Defaults are tuned for a typical screen-shared conference recording at
    720p or 1080p. The sampling and threshold values are the two knobs worth
    touching first if detection is too eager or too sleepy.
    """

    # --- Transcription -------------------------------------------------
    whisper_model: str = "small"
    """faster-whisper size: tiny, base, small, medium, large-v3.
    "small" is the sweet spot on a free Colab T4. Use "tiny" on CPU."""

    language: Optional[str] = None
    """Force a language code (e.g. "en") to skip auto-detection."""

    compute_type: Optional[str] = None
    """CTranslate2 compute type. Auto-selected per device when None."""

    device: Optional[str] = None
    """"cuda" or "cpu". Auto-detected when None."""

    vad_filter: bool = True
    """Drop silence before transcribing. Faster and fewer hallucinated lines."""

    # --- Slide detection -----------------------------------------------
    sample_fps: float = 1.0
    """Frames sampled per second of video. 1.0 catches any slide held >1s."""

    hash_threshold: int = 6
    """Perceptual-hash Hamming distance above which frames are 'different'.
    Lower = more sensitive. 4-10 is the useful band. Measured on real screen
    shares: cursor movement and webcam noise sit around 3-4, while genuine
    slide changes land at 6+."""

    ssim_threshold: float = 0.95
    """Structural similarity below which frames are 'different'. Higher = more
    sensitive. Cursor and webcam churn typically stays above 0.97; real slide
    changes drop to 0.94 or lower. 0.95 sits in that gap."""

    ssim_corroboration: float = 0.98
    """A perceptual-hash hit only counts as a change when SSIM is also below
    this. Guards against pHash instability on flat, high-contrast slides, where
    a moving cursor can otherwise mimic a real slide change."""

    min_slide_seconds: float = 4.0
    """Ignore slide changes closer together than this. Kills animation churn."""

    crop: Optional[tuple] = None
    """(left, top, right, bottom) as 0-1 fractions. Crop to the shared-screen
    region to ignore webcam tiles. None analyses the full frame."""

    max_slides: int = 120
    """Safety ceiling on captured slides."""

    # --- LLM layer -------------------------------------------------------
    llm: LLMConfig = field(default_factory=LLMConfig)

    arbitrate_slides: bool = True
    """Ask a vision model to confirm ambiguous slide changes. Requires vision."""

    caption_slides: bool = True
    """Ask a vision model for a short title per slide. Requires vision."""

    clean_transcript: bool = False
    """Ask a text model to punctuate and paragraph the transcript. Costs tokens
    proportional to transcript length, so it is off by default."""

    summarize: bool = True
    """Generate an executive summary and action items. Text model only."""

    # --- Output ----------------------------------------------------------
    title: Optional[str] = None
    """Document title. Falls back to the input filename."""

    keep_frames: bool = False
    """Leave extracted slide images on disk next to the PDF."""

    verbose: bool = True
