"""Provider-agnostic LLM access.

Everything the pipeline needs from a model goes through ``LLMClient``. The
client speaks to LiteLLM, which routes to whatever provider the model string
names — OpenAI, Anthropic, Gemini, Groq, Together, Mistral, DeepSeek, xAI,
OpenRouter, Ollama, vLLM, or any OpenAI-compatible endpoint.

The pipeline never imports a provider SDK, and every method degrades to a safe
default when no model is configured or a call fails. A run with no key produces
a complete PDF; a run with a key produces a better one.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import List, Optional

log = logging.getLogger(__name__)


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _extract_json(text: str):
    """Pull a JSON object or array out of a model reply.

    Providers differ wildly in structured-output support, so rather than
    depending on JSON mode or tool use, we prompt for JSON and parse
    defensively.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class LLMClient:
    """Thin wrapper over LiteLLM with graceful degradation.

    Args:
        config: An ``LLMConfig``. If ``config.model`` is None the client is
            inert — every method returns its no-LLM fallback.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = config.enabled
        self.vision_ready = False
        self._calls = 0
        self._completion = None

        if not self.enabled:
            log.info("No model configured — running in no-LLM mode.")
            return

        try:
            from litellm import completion

            self._completion = completion
        except ImportError:
            log.warning("litellm not installed — running in no-LLM mode.")
            self.enabled = False
            return

        if config.vision is None:
            self.vision_ready = self._probe_vision()
        else:
            self.vision_ready = bool(config.vision)

        log.info(
            "LLM ready: %s (vision: %s)",
            config.model,
            "yes" if self.vision_ready else "no",
        )

    # -- plumbing ------------------------------------------------------

    def _call(self, messages, max_tokens: int = 1024) -> Optional[str]:
        if not self.enabled:
            return None
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        resp = self._completion(**kwargs)
        return resp.choices[0].message.content

    def _probe_vision(self) -> bool:
        """Send one tiny image to learn whether this model accepts images.

        Doing this once up front beats discovering the answer twenty minutes
        into a long video.
        """
        pixel = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        try:
            self._call(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Reply with the word: ok"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{pixel}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=5,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.info("Model does not accept images (%s) — using CV-only detection.", type(exc).__name__)
            return False

    def _vision_call(self, prompt: str, image_paths: List[str], max_tokens: int = 256):
        if self._calls >= self.config.max_vision_calls:
            raise RuntimeError("vision call budget exhausted")
        content = [{"type": "text", "text": prompt}]
        for p in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64(p)}"},
                }
            )
        self._calls += 1
        return self._call([{"role": "user", "content": content}], max_tokens=max_tokens)

    def check(self) -> dict:
        """Validate the configuration with one cheap call.

        Returns a dict with ``ok``, ``vision``, and ``detail`` so a UI can show
        a clear result before the user commits to a long run.
        """
        if not self.enabled:
            return {"ok": True, "vision": False, "detail": "No-LLM mode."}
        try:
            reply = self._call(
                [{"role": "user", "content": "Reply with exactly: ready"}],
                max_tokens=10,
            )
            return {
                "ok": True,
                "vision": self.vision_ready,
                "detail": f"Model responded: {(reply or '').strip()[:40]}",
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "vision": False, "detail": str(exc)[:300]}

    # -- capabilities used by the pipeline -----------------------------

    def same_slide(self, path_a: str, path_b: str) -> bool:
        """True when both frames show the same slide.

        Catches the cases pixel comparison cannot: a moved cursor, a webcam
        tile, a bullet animating in, or a video playing inside one slide.
        """
        prompt = (
            "These are two frames from a screen-shared presentation. "
            "Answer SAME if they show the same slide (even if a cursor moved, "
            "a webcam overlay changed, an animation revealed more of the same "
            "content, or embedded video is playing). Answer DIFFERENT if the "
            "presenter has moved to genuinely new content. "
            "Reply with one word: SAME or DIFFERENT."
        )
        reply = self._vision_call(prompt, [path_a, path_b], max_tokens=10)
        return bool(reply) and "SAME" in reply.upper()

    def caption(self, image_path: str) -> Optional[str]:
        """A short title for a slide, used as a PDF section heading."""
        prompt = (
            "Give a short title for this presentation slide, 8 words maximum. "
            "Use the slide's own heading if it has one. Reply with the title "
            "only, no quotes and no preamble."
        )
        try:
            reply = self._vision_call(prompt, [image_path], max_tokens=40)
        except Exception as exc:  # noqa: BLE001
            log.warning("Captioning failed: %s", exc)
            return None
        if not reply:
            return None
        return reply.strip().strip('"').split("\n")[0][:90] or None

    def clean(self, text: str) -> Optional[str]:
        """Punctuate and paragraph a raw transcript chunk."""
        prompt = (
            "Clean up this meeting transcript. Fix punctuation and "
            "capitalisation, remove filler words, and break it into "
            "paragraphs. Do not summarise, reword, or drop any content. "
            "Reply with the cleaned transcript only.\n\n" + text
        )
        try:
            return self._call(
                [{"role": "user", "content": prompt}],
                max_tokens=min(4000, len(text) // 2 + 500),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcript cleanup failed: %s", exc)
            return None

    def summarize(self, text: str) -> Optional[dict]:
        """Executive summary, key points, and action items.

        Returns a dict with ``summary`` (str), ``key_points`` (list), and
        ``action_items`` (list), or None if the model is unavailable.
        """
        prompt = (
            "Read this meeting transcript and reply with a JSON object only, "
            "no markdown fences and no commentary, with exactly these keys:\n"
            '  "summary": a 3-5 sentence overview,\n'
            '  "key_points": an array of up to 8 short strings,\n'
            '  "action_items": an array of up to 10 short strings, each naming '
            "the owner if the transcript states one. Use an empty array if "
            "there are none.\n\nTranscript:\n" + text[:60000]
        )
        try:
            reply = self._call([{"role": "user", "content": prompt}], max_tokens=1500)
        except Exception as exc:  # noqa: BLE001
            log.warning("Summarisation failed: %s", exc)
            return None
        data = _extract_json(reply or "")
        if not isinstance(data, dict):
            log.warning("Could not parse summary JSON — skipping summary section.")
            return None
        return {
            "summary": str(data.get("summary", "")).strip(),
            "key_points": [str(x) for x in data.get("key_points", [])][:8],
            "action_items": [str(x) for x in data.get("action_items", [])][:10],
        }
