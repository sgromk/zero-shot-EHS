"""
Stage 2 — Multi-class accident classification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

CLASSIFICATION_PROMPT_STRUCTURED = """\
You are a senior workplace safety expert and EHS (Environmental, Health & Safety) analyst \
reviewing a video clip that has been flagged as containing an accident.

── STEP 1: OBSERVE ────────────────────────────────────────────────────────────
Before classifying, note:
- The type of workplace or environment
- What physical event caused injury or harm
- The direct source of harm (object, surface, vehicle, machine, substance, energy)
- Any secondary events that also caused harm

── STEP 2: CLASSIFY ALL INCIDENT TYPES ───────────────────────────────────────
One clip may contain multiple distinct accident types. \
Report ALL types you observe with confidence ≥ 0.5, ordered by confidence.

Use these rules to choose categories:

VEHICLE INCIDENT — A powered vehicle in motion is the direct cause.
  The vehicle (forklift, pallet jack, delivery truck, industrial cart, crane) \
strikes, runs over, pins, or knocks the person down.
  ✓ Forklift strikes or pins a worker
  ✓ Truck hits a pedestrian in a work zone
  ✗ Box falling OFF a forklift → that is Struck by Object
  ✗ Person tripping NEAR a vehicle → that is Trip or Fall

STRUCK BY OBJECT — A falling, flying, swinging, or thrown object hits the person.
  The object is not itself a powered vehicle.
  ✓ Box falls from shelf, tool dropped from above, flying debris

SLIP  — Foot loses grip on wet/slippery/oily surface; person falls backward or sideways.
TRIP  — Foot catches on an obstacle, ridge, or cord; person pitches forward.
FALL  — Person falls from height (ladder, platform, stairs, elevated surface).
  Use Fall when height is the key factor, not a surface slip or foot obstacle.

CAUGHT IN MACHINE — Body part is trapped, pulled into, or crushed by moving machinery.
ELECTROCUTION     — Electrical current passes through a person's body.
ARC FLASH         — An explosive electrical arc directly injures a person.
FIRE              — Person is burned or injured by direct flame, heat, or explosion.
GAS INHALATION    — Person is harmed by inhaling gas, fumes, smoke, or vapour.
LIFTING           — Injury during manual lifting, carrying, or pushing of loads.

Valid category names (copy exactly):
Arc Flash | Caught In Machine | Electrocution | Fall | Fire | Gas Inhalation | \
Lifting | Slip | Struck by Object | Trip | Vehicle Incident

── STEP 3: EHS REPORT ────────────────────────────────────────────────────────
Generate a brief professional EHS report suitable for incident documentation.

Return ONLY valid JSON in this exact format:
{
  "reasoning": "2-3 sentences: what you observed and what events occurred",
  "incidents": [
    {"category": "PRIMARY category name", "confidence": 0.0-1.0},
    {"category": "SECONDARY category name if applicable", "confidence": 0.0-1.0}
  ],
  "incident_start_time": "hh:mm:ss",
  "incident_end_time": "hh:mm:ss",
  "description": "clear factual description of what happened",
  "ehs_report": {
    "severity": "low | medium | high | critical",
    "immediate_actions": "actions taken or required immediately after the incident",
    "root_cause": "underlying cause of the incident",
    "contributing_factors": "additional factors that contributed",
    "corrective_measures": "preventive actions to avoid recurrence"
  }
}

Notes:
- incidents must contain at least one entry
- Only include categories with confidence ≥ 0.5
- Order incidents by confidence descending (highest first)
- severity: low=first-aid only, medium=medical treatment, high=hospitalisation, critical=life-threatening/fatal
"""

CLASSIFICATION_PROMPTS: dict[str, str] = {
    "default":    CLASSIFICATION_PROMPT_DEFAULT,
    "animated":   CLASSIFICATION_PROMPT_DEFAULT,   # backward-compat alias
    "structured": CLASSIFICATION_PROMPT_STRUCTURED,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    category: str                  # primary (highest-confidence) category
    categories: list               # all detected: [{"category": ..., "confidence": ...}]
    incident_start_time: str | None
    incident_end_time: str | None
    confidence: float              # primary category confidence
    description: str
    root_cause_analysis: str
    ehs_report: dict               # full EHS report (empty dict for legacy prompts)
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

    # Support both response formats:
    #   Legacy (default/animated prompts): {"category": "Fall", "confidence": 0.8, ...}
    #   Structured prompt:                 {"incidents": [...], "ehs_report": {...}, ...}
    incidents_raw = parsed.get("incidents")
    if incidents_raw and isinstance(incidents_raw, list):
        # Multi-label path: normalise each entry, drop below 0.5, sort by confidence desc
        valid_incidents = []
        for inc in incidents_raw:
            cat_raw = inc.get("category", "")
            conf = float(inc.get("confidence", 0.5))
            if conf < 0.5:
                continue
            norm, _ = _normalize_with_flag(cat_raw, _categories)
            valid_incidents.append({"category": norm, "confidence": conf})
        valid_incidents.sort(key=lambda x: x["confidence"], reverse=True)

        if valid_incidents:
            primary_category = valid_incidents[0]["category"]
            primary_confidence = valid_incidents[0]["confidence"]
            fallback_used = primary_category == DEFAULT_FALLBACK_CATEGORY
        else:
            primary_category, fallback_used = DEFAULT_FALLBACK_CATEGORY, True
            primary_confidence = 0.5
            valid_incidents = [{"category": primary_category, "confidence": primary_confidence}]

        categories = valid_incidents
        root_cause = parsed.get("ehs_report", {}).get("root_cause", "")
        ehs_report = parsed.get("ehs_report", {})
    else:
        # Legacy single-label path
        raw_category = parsed.get("category")
        primary_category, fallback_used = _normalize_with_flag(raw_category, _categories)
        primary_confidence = float(parsed.get("confidence", 0.5))
        categories = [{"category": primary_category, "confidence": primary_confidence}]
        root_cause = parsed.get("root_cause_analysis", "")
        ehs_report = {}

    return ClassificationResult(
        category=primary_category,
        categories=categories,
        incident_start_time=parsed.get("incident_start_time"),
        incident_end_time=parsed.get("incident_end_time"),
        confidence=primary_confidence,
        description=parsed.get("description", ""),
        root_cause_analysis=root_cause,
        ehs_report=ehs_report,
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
