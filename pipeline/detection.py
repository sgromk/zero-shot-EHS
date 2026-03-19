"""
Stage 1 — Binary accident detection.

Supports:
- Single call (fast)
- Majority-vote over N calls (higher recall)
- any(votes) policy (current best practice)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vertexai.generative_models import GenerativeModel, Part

from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompts ───────────────────────────────────────────────────────────────────

BINARY_PROMPT_DEFAULT = """\
You are a workplace safety inspector analyzing a video clip.
Your priority is to avoid false negatives:
if there is ANY physical impact or consequence to a person, \
it MUST be labeled as an accident.

Mark accident TRUE if ANY of the following occur (even minor):
- falls or partially falls (knees/butt/hand hits ground or clear loss of support)
- slips/trips with clear stumble resulting in ground/object contact OR visible distress
- struck by any object with visible bodily reaction
- electrical shock with any visible reaction (jerk, recoil, collapse)
- exposure to fire/steam/gas that visibly affects the person (contact, flinch, collapse, distress)
- caught/trapped in machinery (even briefly)
- vehicle contact or impact

Mark accident FALSE only if:
- no physical consequence happens to the person (walking normally, stable posture)
- purely showing safe procedure
- only a near miss (no contact, no fall, no bodily reaction)
- hazard (welding/fire/electricity) appears but no person is affected

Return ONLY JSON:
{
  "incident_detected": true/false,
  "confidence": 0.0-1.0
}
"""

BINARY_PROMPT_HIGH_RECALL = """\
You are a surveillance safety monitor.

Your job is to catch EVERY real accident — missing one is unacceptable.
When in doubt, flag it as an accident and let the next stage decide.

Mark accident TRUE if you observe any of the following, even briefly or subtly:
- A person falls, stumbles, slips, or trips with any loss of footing
- A person is struck by any object, tool, or vehicle
- A person makes contact with machinery, electrical equipment, fire, steam, or gas
- A person shows ANY sign of pain, distress, recoil, or collapse
- A person is caught, pinned, or trapped by anything

Mark accident FALSE ONLY if:
- The video clearly shows only safe, controlled behavior with no incident at all
- A hazard is present but zero persons are anywhere near it

If you are UNSURE, mark TRUE.

Return ONLY JSON:
{
  "incident_detected": true/false,
  "confidence": 0.0-1.0
}
"""

BINARY_PROMPT_HIGH_RECALL = """\
You are a surveillance safety monitor reviewing workplace footage.

Your only job is to ensure NO real accident goes undetected.
Missing an accident is a critical failure. Flagging a borderline case is fine.

Mark accident TRUE if you observe ANY of the following, even briefly or ambiguously:
- A person falls, stumbles, slips, trips, or partially loses their footing
- A person is struck by any object, tool, equipment, or vehicle
- A person makes contact with machinery, electrical equipment, fire, steam, or hazardous substances
- A person shows any sign of pain, distress, surprise, recoil, or collapse
- A person is caught, pinned, entangled, or trapped by anything
- Any rapidly uncontrolled motion or loss of balance by a worker

Mark accident FALSE only when:
- The clip exclusively shows clearly safe, controlled work with zero incident indicators
- A hazard exists but no worker is anywhere near it and no incident occurs

When in doubt: mark TRUE. Downstream review will filter false positives.

Return ONLY JSON:
{
  "incident_detected": true/false,
  "confidence": 0.0-1.0
}
"""

BINARY_PROMPT_STRICT = """\
You are a strict workplace safety inspector.

Carefully observe the entire video.

An accident should ONLY be marked TRUE if:
- A human is physically harmed, falls, is struck, electrocuted, trapped, or injured.
- There is a clear negative impact on a person.

DO NOT classify as accident if:
- Welding sparks are visible but no human is harmed.
- Fire is present but no person is affected.
- Equipment operates normally.
- Safety demonstrations are shown.

Return ONLY JSON:
{
  "incident_detected": true/false,
  "confidence": 0.0-1.0
}
"""

# Alias for backward compatibility
BINARY_PROMPT_ANIMATED = BINARY_PROMPT_DEFAULT

# Available prompt variants for ablation studies
BINARY_PROMPTS: dict[str, str] = {
    "default": BINARY_PROMPT_DEFAULT,
    "animated": BINARY_PROMPT_DEFAULT,   # backward-compat alias
    "strict": BINARY_PROMPT_STRICT,
    "high_recall": BINARY_PROMPT_HIGH_RECALL,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    incident_detected: bool
    confidence: float
    votes: list[bool]
    raw_responses: list[str]
    latency_s: float


# ── Core logic ────────────────────────────────────────────────────────────────

def detect_single(
    video_part: Part,
    model: GenerativeModel | None = None,
    prompt: str = BINARY_PROMPT_ANIMATED,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[bool, float, str]:
    """
    Single binary detection call.

    Returns (incident_detected, confidence, raw_response_text).
    """
    _model = model or vertex_model
    _temp = temperature if temperature is not None else config.temperature

    gen_config: dict = {"temperature": _temp, "response_mime_type": "application/json"}
    if top_k is not None:
        gen_config["top_k"] = top_k
    if top_p is not None:
        gen_config["top_p"] = top_p

    response = generate_with_retry(
        model=_model,
        parts=[video_part, prompt],
        generation_config=gen_config,
    )
    raw = response.text
    parsed = safe_parse_json(raw) or {}
    detected = bool(parsed.get("incident_detected", False))
    confidence = float(parsed.get("confidence", 0.5))
    return detected, confidence, raw


def detect(
    video_part: Part,
    model: GenerativeModel | None = None,
    prompt: str = BINARY_PROMPT_ANIMATED,
    n_votes: int | None = None,
    vote_policy: str = "any",
    confidence_threshold: float | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> DetectionResult:
    """
    Run Stage 1 binary detection with optional majority voting.

    Parameters
    ----------
    n_votes : int
        Number of independent calls to make (default from config).
    vote_policy : str
        "any"      — incident if ANY vote is True  (high recall)
        "majority" — incident if > n_votes/2 are True
        "all"      — incident if ALL votes are True (high precision)
    confidence_threshold : float
        Minimum confidence required for a positive detection.
    """
    n = n_votes if n_votes is not None else config.stage1_votes
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else config.confidence_threshold
    )

    t0 = time.perf_counter()

    if n == 1:
        # Fast path — no threading overhead
        detected, conf, raw = detect_single(
            video_part, model=model, prompt=prompt,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        raw_results = [(detected, conf, raw)]
    else:
        # All votes are independent — fire them in parallel.
        # GenerativeModel.generate_content() is thread-safe (stateless HTTP).
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(
                    detect_single, video_part,
                    model, prompt, temperature, top_k, top_p,
                )
                for _ in range(n)
            ]
            raw_results = [f.result() for f in futures]

    elapsed = time.perf_counter() - t0

    votes: list[bool] = []
    confidences: list[float] = []
    raws: list[str] = []
    for detected, conf, raw in raw_results:
        votes.append(detected and conf >= threshold)
        confidences.append(conf)
        raws.append(raw)

    if vote_policy == "any":
        incident = any(votes)
    elif vote_policy == "majority":
        incident = sum(votes) > n / 2
    elif vote_policy == "all":
        incident = all(votes)
    else:
        raise ValueError(f"Unknown vote_policy: {vote_policy!r}")

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return DetectionResult(
        incident_detected=incident,
        confidence=mean_confidence,
        votes=votes,
        raw_responses=raws,
        latency_s=elapsed,
    )
