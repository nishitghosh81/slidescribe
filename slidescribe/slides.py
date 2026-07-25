"""Slide-change detection.

Two stages, cheap to expensive:

1. Sample frames and compare consecutive ones with a perceptual hash plus
   structural similarity. Pure computer vision, no model, no API key.
2. Optionally hand the ambiguous cases to a vision model, which can tell a
   genuine slide change from a moving cursor or an animated build.

Stage 1 alone is good on clean recordings. Stage 2 is what handles webcam
tiles, bullet animations, and embedded video.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Slide:
    """A captured slide and where it lives in the recording."""

    index: int
    timestamp: float
    image_path: str
    title: Optional[str] = None
    end_timestamp: Optional[float] = None
    confidence: str = "high"
    _bgr: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def clock(self) -> str:
        m, s = divmod(int(self.timestamp), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _apply_crop(frame: np.ndarray, crop) -> np.ndarray:
    """Crop to a fractional (left, top, right, bottom) box."""
    if not crop:
        return frame
    h, w = frame.shape[:2]
    l, t, r, b = crop
    return frame[int(t * h) : int(b * h), int(l * w) : int(r * w)]


def _phash(gray: np.ndarray, size: int = 32, keep: int = 8) -> np.ndarray:
    """Perceptual hash via DCT. Returns a flat boolean array."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:keep, :keep]
    med = np.median(low)
    return (low > med).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on downscaled grayscale frames.

    Implemented directly so scikit-image stays an optional dependency.
    """
    a = cv2.resize(a, (256, 144), interpolation=cv2.INTER_AREA).astype(np.float64)
    b = cv2.resize(b, (256, 144), interpolation=cv2.INTER_AREA).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    return float(np.mean(num / den))


def detect_slides(video_path: str, out_dir: str, config, llm=None) -> List[Slide]:
    """Find slide changes and write one image per slide.

    Args:
        video_path: Source video.
        out_dir: Directory to write slide images into.
        config: A ``Config`` instance.
        llm: Optional ``LLMClient`` for arbitrating ambiguous changes.

    Returns:
        Captured slides in chronological order.
    """
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0.0
    step = max(1, int(round(fps / config.sample_fps)))

    log.info(
        "Video: %.1fs at %.1f fps — sampling every %d frames (%.1f fps)",
        duration, fps, step, config.sample_fps,
    )

    try:
        from tqdm.auto import tqdm

        bar = tqdm(total=total or None, unit="f", desc="Scanning frames")
    except ImportError:
        bar = None

    slides: List[Slide] = []
    prev_hash = None
    prev_gray = None
    last_capture_t = -1e9
    ambiguous: List[int] = []
    idx = 0

    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = idx / fps
                region = _apply_crop(frame, config.crop)
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                h = _phash(gray)

                is_first = prev_hash is None
                changed = False
                score = 1.0
                dist = 0

                if not is_first:
                    dist = _hamming(prev_hash, h)
                    score = _ssim(prev_gray, gray)
                    # SSIM is the reliable signal and must always agree. A
                    # perceptual hash alone is too jumpy on flat, high-contrast
                    # slides: a cursor crossing a static slide can produce the
                    # same Hamming distance as a genuine slide change, so
                    # letting pHash fire on its own floods the output with
                    # duplicates. pHash instead acts as a sensitivity boost,
                    # catching changes that are structurally similar but
                    # visually distinct (a chart swapped for another chart).
                    ssim_says_changed = score < config.ssim_threshold
                    both_say_changed = (
                        dist >= config.hash_threshold
                        and score < config.ssim_corroboration
                    )
                    changed = ssim_says_changed or both_say_changed

                if is_first or (changed and t - last_capture_t >= config.min_slide_seconds):
                    if len(slides) >= config.max_slides:
                        log.warning("Hit max_slides=%d — stopping capture.", config.max_slides)
                        break

                    # Borderline signals get flagged for the vision model.
                    borderline = (not is_first) and (
                        dist < config.hash_threshold * 1.5
                        and score > config.ssim_threshold - 0.03
                    )

                    n = len(slides)
                    path = os.path.join(out_dir, f"slide_{n:03d}.jpg")
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

                    if slides:
                        slides[-1].end_timestamp = t

                    slides.append(
                        Slide(
                            index=n,
                            timestamp=t,
                            image_path=path,
                            confidence="low" if borderline else "high",
                            _bgr=frame.copy() if borderline else None,
                        )
                    )
                    if borderline:
                        ambiguous.append(n)
                    last_capture_t = t

                prev_hash, prev_gray = h, gray
        idx += 1
        if bar:
            bar.update(1)

    cap.release()
    if bar:
        bar.close()

    if slides:
        slides[-1].end_timestamp = duration or slides[-1].timestamp + 60

    log.info("Captured %d candidate slides (%d ambiguous)", len(slides), len(ambiguous))

    if ambiguous and llm and llm.vision_ready and config.arbitrate_slides:
        slides = _arbitrate(slides, ambiguous, llm, config)

    for i, s in enumerate(slides):
        s.index = i
        s._bgr = None

    return slides


def _arbitrate(slides: List[Slide], ambiguous: List[int], llm, config) -> List[Slide]:
    """Ask a vision model whether each borderline capture is really new.

    Only the flagged frames are sent, so cost stays proportional to the number
    of genuinely uncertain moments rather than to video length.
    """
    budget = min(len(ambiguous), config.llm.max_vision_calls)
    if budget < len(ambiguous):
        log.warning(
            "Vision budget %d < %d ambiguous frames — arbitrating the first %d.",
            budget, len(ambiguous), budget,
        )

    drop = set()
    for n in ambiguous[:budget]:
        if n == 0:
            continue
        cur = slides[n]
        prev = slides[n - 1]
        if cur._bgr is None:
            continue
        try:
            same = llm.same_slide(prev.image_path, cur.image_path)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            log.warning("Vision arbitration failed on slide %d: %s", n, exc)
            continue
        if same:
            drop.add(n)

    if not drop:
        return slides

    log.info("Vision model merged %d duplicate captures", len(drop))
    kept = []
    for i, s in enumerate(slides):
        if i in drop:
            # Fold this capture's span into the slide it duplicates.
            if kept:
                kept[-1].end_timestamp = s.end_timestamp
            try:
                os.remove(s.image_path)
            except OSError:
                pass
        else:
            kept.append(s)
    return kept
