"""
Structured accident output: decomposed components + severity ordering.

Direction 2 from the research plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vertexai.generative_models import GenerativeModel, Part

from config.categories import MERGED_CATEGORIES, sort_by_severity
from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompt ────────────────────────────────────────────────────────────────────

STRUCTURED_PROMPT = """\
IMPORTANT: This is an ANIMATED workplace safety video.

This video contains an accident. Analyze it carefully and:

1. Identify ALL accident components (e.g., a Trip that causes a Fall = two components).
2. Order components from cause → effect.
3. Identify the PRIMARY (most severe) category.

Valid categories:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire,
Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident

Return ONLY JSON:
{
  "primary_category": "most severe category",
  "accident_components": ["cause_category", "effect_category"],
  "severity_ordered": ["most_severe", ..., "least_severe"],
  "near_miss": false,
  "incident_start_time": "hh:mm:ss",
  "incident_end_time": "hh:mm:ss",
  "confidence": 0.0-1.0,
  "description": "factual description",
  "root_cause_analysis": "short safety report"
}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StructuredAccidentOutput:
    primary_category: str
    accident_components: list[str]
    severity_ordered: list[str]
    near_miss: bool
    incident_start_time: str | None
    incident_end_time: str | None
    confidence: float
    description: str
    root_cause_analysis: str
    raw_response: str
    latency_s: float


# ── Core logic ────────────────────────────────────────────────────────────────

def classify_structured(
    video_part: Part,
    model: GenerativeModel | None = None,
    temperature: float | None = None,
) -> StructuredAccidentOutput:
    """
    Return a structured accident classification with decomposed components
    and severity ordering.
    """
    _model = model or vertex_model
    _temp = temperature if temperature is not None else config.temperature

    t0 = time.perf_counter()
    response = generate_with_retry(
        model=_model,
        parts=[video_part, STRUCTURED_PROMPT],
        generation_config={
            "temperature": _temp,
            "response_mime_type": "application/json",
        },
    )
    elapsed = time.perf_counter() - t0

    raw = response.text
    parsed = safe_parse_json(raw) or {}

    # Validate components
    raw_components: list[str] = parsed.get("accident_components", [])
    valid_components = [c for c in raw_components if c in MERGED_CATEGORIES]
    if not valid_components:
        valid_components = [parsed.get("primary_category", "Fall")]

    # Build severity-ordered list
    severity_ordered = sort_by_severity(valid_components)

    primary = parsed.get("primary_category", "Fall")
    if primary not in MERGED_CATEGORIES:
        primary = severity_ordered[0] if severity_ordered else "Fall"

    return StructuredAccidentOutput(
        primary_category=primary,
        accident_components=valid_components,
        severity_ordered=severity_ordered,
        near_miss=bool(parsed.get("near_miss", False)),
        incident_start_time=parsed.get("incident_start_time"),
        incident_end_time=parsed.get("incident_end_time"),
        confidence=float(parsed.get("confidence", 0.5)),
        description=parsed.get("description", ""),
        root_cause_analysis=parsed.get("root_cause_analysis", ""),
        raw_response=raw,
        latency_s=elapsed,
    )
