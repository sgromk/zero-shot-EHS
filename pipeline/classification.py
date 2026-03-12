"""
Stage 2 — Multi-class accident classification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vertexai.generative_models import GenerativeModel, Part

from config.categories import MERGED_CATEGORIES, DEFAULT_FALLBACK_CATEGORY
from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT_DEFAULT = """\
You are a workplace safety analyst reviewing a video clip that contains an accident.

Classify the PRIMARY accident type into EXACTLY ONE of the following:

Arc Flash
Caught In Machine
Electrocution
Fall
Fire
Gas Inhalation
Lifting
Slip
Struck by Object
Trip
Vehicle Incident

Classification Rules:
- Slip: Loss of footing due to slippery surface.
- Trip: Foot strikes obstacle while walking.
- Fall: Fall without clear slip/trip cause.
- Electrocution: Electrical shock to a human.
- Arc Flash: Visible electrical arc causing injury.
- Struck by Object: Human hit by falling/moving object.
- Vehicle Incident: Human impacted by vehicle.
- Fire: Human injured due to fire.
- Gas Inhalation: Human harmed by gas exposure.
- Caught In Machine: Human trapped or caught.
- Lifting: Injury during lifting operation.

If multiple events occur, choose the most severe injury-causing event.

Return ONLY JSON:
{
  "category": "ONE exact category from list above",
  "incident_start_time": "hh:mm:ss",
  "incident_end_time": "hh:mm:ss",
  "confidence": 0.0-1.0,
  "description": "clear factual description of what happened",
  "root_cause_analysis": "professional short safety analysis"
}
"""

# Alias for backward compatibility
CLASSIFICATION_PROMPT_ANIMATED = CLASSIFICATION_PROMPT_DEFAULT

CLASSIFICATION_PROMPTS: dict[str, str] = {
    "default": CLASSIFICATION_PROMPT_DEFAULT,
    "animated": CLASSIFICATION_PROMPT_DEFAULT,  # backward-compat alias
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    category: str
    incident_start_time: str | None
    incident_end_time: str | None
    confidence: float
    description: str
    root_cause_analysis: str
    raw_response: str
    latency_s: float
    fallback_used: bool = False


# ── Core logic ────────────────────────────────────────────────────────────────

def classify(
    video_part: Part,
    model: GenerativeModel | None = None,
    prompt: str = CLASSIFICATION_PROMPT_ANIMATED,
    valid_categories: list[str] | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> ClassificationResult:
    """
    Run Stage 2 classification on a video.

    Returns a ClassificationResult with the predicted category and metadata.
    """
    _model = model or vertex_model
    _temp = temperature if temperature is not None else config.temperature
    _categories = valid_categories or MERGED_CATEGORIES

    gen_config: dict = {"temperature": _temp, "response_mime_type": "application/json"}
    if top_k is not None:
        gen_config["top_k"] = top_k
    if top_p is not None:
        gen_config["top_p"] = top_p

    t0 = time.perf_counter()
    response = generate_with_retry(
        model=_model,
        parts=[video_part, prompt],
        generation_config=gen_config,
    )
    elapsed = time.perf_counter() - t0

    raw = response.text
    parsed = safe_parse_json(raw) or {}

    raw_category = parsed.get("category")
    normalized, fallback_used = _normalize_with_flag(raw_category, _categories)

    return ClassificationResult(
        category=normalized,
        incident_start_time=parsed.get("incident_start_time"),
        incident_end_time=parsed.get("incident_end_time"),
        confidence=float(parsed.get("confidence", 0.5)),
        description=parsed.get("description", ""),
        root_cause_analysis=parsed.get("root_cause_analysis", ""),
        raw_response=raw,
        latency_s=elapsed,
        fallback_used=fallback_used,
    )


def _normalize_with_flag(
    category: str | None,
    valid_categories: list[str],
) -> tuple[str, bool]:
    """Normalize category and return (normalized, fallback_used)."""
    if not category:
        return DEFAULT_FALLBACK_CATEGORY, True
    if category in valid_categories:
        return category, False
    lower_map = {c.lower(): c for c in valid_categories}
    matched = lower_map.get(category.strip().lower())
    if matched:
        return matched, False
    return DEFAULT_FALLBACK_CATEGORY, True
