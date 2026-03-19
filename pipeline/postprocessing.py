"""
JSON parsing, JSON extraction from markdown code fences, and retry logic.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeoutError
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vertexai.generative_models import GenerativeModel

from config.settings import config


# ── JSON parsing ─────────────────────────────────────────────────────────────


def safe_parse_json(text: str | None) -> dict | None:
    """
    Parse a JSON string returned by the model.

    Handles:
    - Markdown code fences (```json ... ```)
    - Responses that are JSON arrays instead of objects
    """
    if not text:
        return None
    text = text.strip()
    # Strip ``` fences
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0]
        return parsed
    except (json.JSONDecodeError, IndexError):
        return None


# ── Category normalization ────────────────────────────────────────────────────


def normalize_category(
    category: str | None,
    valid_categories: list[str],
    fallback: str = "Fall",
) -> str:
    """
    Return `category` if it is in `valid_categories`, else `fallback`.

    Case-insensitive match attempted first.
    """
    if not category:
        return fallback
    if category in valid_categories:
        return category
    # Case-insensitive fallback
    lower_map = {c.lower(): c for c in valid_categories}
    return lower_map.get(category.strip().lower(), fallback)


# ── Retry wrapper ─────────────────────────────────────────────────────────────


RETRYABLE_ERRORS = (
    "invalid argument", "400", "unavailable", "internal", "503", "429",
    "resource_exhausted", "deadline", "timeout",
)

# Per-call timeout in seconds. Vertex AI video calls can take 30–90s;
# 120s gives headroom without blocking the whole sweep indefinitely.
_REQUEST_TIMEOUT_S = 120


def generate_with_retry(
    model: GenerativeModel,
    parts: list,
    generation_config: dict,
    tries: int | None = None,
) -> Any:
    """
    Call `model.generate_content` with exponential-backoff retry and a hard timeout.

    Retries on transient errors (400, 503, rate-limit, internal, quota, timeout).
    Raises immediately on non-retryable errors.
    """
    max_tries = tries or config.retry_attempts
    last_err: Exception | None = None

    for attempt in range(max_tries):
        # Run generate_content in a worker thread so we can enforce a hard
        # timeout. shutdown(wait=False) lets hung threads drain in the background
        # without blocking the retry loop or the sweep.
        _ex = ThreadPoolExecutor(max_workers=1)
        _fut = _ex.submit(model.generate_content, parts, generation_config=generation_config)
        try:
            result = _fut.result(timeout=_REQUEST_TIMEOUT_S)
            _ex.shutdown(wait=False)
            return result
        except _FuturesTimeoutError:
            _ex.shutdown(wait=False)
            last_err = TimeoutError(
                f"generate_content timed out after {_REQUEST_TIMEOUT_S}s"
            )
            time.sleep(2 ** attempt)
            continue
        except Exception as exc:
            _ex.shutdown(wait=False)
            last_err = exc
            msg = str(exc).lower()
            if any(token in msg for token in RETRYABLE_ERRORS):
                time.sleep(2 ** attempt)
                continue
            raise  # non-retryable — surface immediately

    raise last_err  # type: ignore[misc]
