"""Tests for SlideScribe.

Run with: pytest -q

The slide-detection tests build a synthetic screen share containing the two
things that break naive pixel comparison: a moving cursor and a noisy webcam
tile. Detection must find the real slide changes and ignore both.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from slidescribe.config import Config, LLMConfig
from slidescribe.llm import LLMClient, _extract_json
from slidescribe.pdf import build_pdf
from slidescribe.slides import _hamming, _phash, _ssim, detect_slides
from slidescribe.transcribe import Segment

W, H, FPS = 1280, 720, 10
SECONDS_PER_SLIDE = 12
N_SLIDES = 4


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 48s synthetic screen share: 4 slides, moving cursor, noisy webcam."""
    path = str(tmp_path_factory.mktemp("v") / "deck.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    decks = [
        ("Q3 Revenue Overview", (30, 40, 60)),
        ("Market Expansion", (60, 30, 40)),
        ("Cost Structure", (35, 55, 35)),
        ("Next Steps", (50, 45, 25)),
    ]
    rng = np.random.default_rng(0)
    for i, (text, bg) in enumerate(decks):
        base = np.full((H, W, 3), bg, np.uint8)
        cv2.putText(base, text, (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (240, 240, 240), 4)
        for b in range(3):
            cv2.putText(base, f"- point {b + 1} of section {i + 1}", (110, 340 + b * 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 205, 210), 2)
        for f in range(SECONDS_PER_SLIDE * FPS):
            frame = base.copy()
            cx = 300 + int(200 * np.sin(f / 8))          # moving cursor
            cv2.circle(frame, (cx, 600), 9, (255, 255, 255), -1)
            frame[20:160, W - 220:W - 20] = rng.integers(  # noisy webcam tile
                0, 255, (140, 200, 3), dtype=np.uint8
            )
            writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def clean_video(tmp_path_factory):
    """Same deck, no webcam tile — just a cursor moving over flat slides.

    Regression guard: a perceptual hash is unstable on flat, high-contrast
    frames, and a cursor alone can produce the same Hamming distance as a real
    slide change. Without SSIM corroboration this video yields ~8 false
    captures instead of 4.
    """
    path = str(tmp_path_factory.mktemp("v2") / "clean.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i, (text, bg) in enumerate([
        ("Q3 Revenue Overview", (30, 40, 60)), ("Market Expansion", (60, 30, 40)),
        ("Cost Structure", (35, 55, 35)), ("Next Steps", (50, 45, 25))]):
        base = np.full((H, W, 3), bg, np.uint8)
        cv2.putText(base, text, (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (240,) * 3, 4)
        for b in range(3):
            cv2.putText(base, f"- point {b + 1} of section {i + 1}", (110, 340 + b * 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 205, 210), 2)
        for f in range(SECONDS_PER_SLIDE * FPS):
            frame = base.copy()
            cv2.circle(frame, (300 + int(200 * np.sin(f / 8)), 600), 9, (255,) * 3, -1)
            writer.write(frame)
    writer.release()
    return path


def test_cursor_alone_does_not_trigger_captures(clean_video, tmp_path):
    slides = detect_slides(clean_video, str(tmp_path / "f"), Config(verbose=False))
    assert len(slides) == N_SLIDES, (
        f"expected {N_SLIDES}, got {len(slides)} — pHash is firing without "
        "SSIM corroboration"
    )


@pytest.fixture
def segments():
    return [
        Segment(t, t + 4, f"Spoken content at second {t} about the current slide.")
        for t in range(0, N_SLIDES * SECONDS_PER_SLIDE, 4)
    ]


# --- detection -------------------------------------------------------

def test_finds_every_slide_change(video, tmp_path):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    assert len(slides) == N_SLIDES


def test_ignores_cursor_and_webcam_noise(video, tmp_path):
    """No extra captures beyond the real changes."""
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    assert len(slides) <= N_SLIDES, "cursor or webcam churn caused false captures"


def test_timestamps_land_on_transitions(video, tmp_path):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    for i, s in enumerate(slides):
        assert abs(s.timestamp - i * SECONDS_PER_SLIDE) < 2.0


def test_spans_are_contiguous(video, tmp_path):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    for a, b in zip(slides, slides[1:]):
        assert a.end_timestamp == pytest.approx(b.timestamp, abs=0.5)
    assert slides[-1].end_timestamp > slides[-1].timestamp


def test_images_written_and_readable(video, tmp_path):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    for s in slides:
        assert os.path.exists(s.image_path)
        assert cv2.imread(s.image_path) is not None


def test_min_slide_seconds_suppresses_rapid_captures(video, tmp_path):
    cfg = Config(verbose=False, min_slide_seconds=90.0)
    slides = detect_slides(video, str(tmp_path / "f"), cfg)
    assert len(slides) == 1


def test_max_slides_is_respected(video, tmp_path):
    cfg = Config(verbose=False, max_slides=2)
    assert len(detect_slides(video, str(tmp_path / "f"), cfg)) <= 2


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        detect_slides("/nope/missing.mp4", str(tmp_path / "f"), Config(verbose=False))


# --- primitives ------------------------------------------------------

def test_identical_frames_score_as_identical():
    a = np.full((720, 1280), 128, np.uint8)
    assert _hamming(_phash(a), _phash(a.copy())) == 0
    assert _ssim(a, a.copy()) == pytest.approx(1.0, abs=1e-6)


def test_different_frames_separate():
    a = np.zeros((720, 1280), np.uint8)
    b = np.zeros((720, 1280), np.uint8)
    cv2.putText(b, "TOTALLY DIFFERENT", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 8)
    assert _hamming(_phash(a), _phash(b)) > 0
    assert _ssim(a, b) < 0.99


# --- LLM layer -------------------------------------------------------

def test_no_key_is_inert_not_broken():
    client = LLMClient(LLMConfig())
    assert client.enabled is False
    assert client.vision_ready is False
    assert client.check()["ok"] is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! {"a": 1} hope that helps', {"a": 1}),
        ("no json at all", None),
    ],
)
def test_json_parsing_survives_provider_quirks(raw, expected):
    assert _extract_json(raw) == expected


# --- PDF -------------------------------------------------------------

def test_pdf_with_slides(video, tmp_path, segments):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    out = str(tmp_path / "out.pdf")
    build_pdf(out, "Test Call", segments, slides)
    assert os.path.getsize(out) > 1000


def test_pdf_without_slides_still_builds(tmp_path, segments):
    out = str(tmp_path / "text.pdf")
    build_pdf(out, "Audio Only", segments, [])
    assert os.path.getsize(out) > 500


def test_pdf_with_summary_section(video, tmp_path, segments):
    slides = detect_slides(video, str(tmp_path / "f"), Config(verbose=False))
    out = str(tmp_path / "sum.pdf")
    build_pdf(out, "With Summary", segments, slides,
              summary={"summary": "An overview.", "key_points": ["A"], "action_items": ["B"]})
    assert os.path.getsize(out) > 1000


def test_pdf_escapes_xml_in_transcript(tmp_path):
    out = str(tmp_path / "esc.pdf")
    build_pdf(out, "Escaping <&>", [Segment(0, 1, "a < b & c > d")], [])
    assert os.path.getsize(out) > 500


def test_empty_transcript_does_not_crash(tmp_path):
    out = str(tmp_path / "empty.pdf")
    build_pdf(out, "Silent", [], [])
    assert os.path.exists(out)
