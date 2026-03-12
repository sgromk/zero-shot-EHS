"""
Near-miss classification module (Direction 6).

Extends binary detection to a three-class output:
  safe | near_miss | accident
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vertexai.generative_models import GenerativeModel, Part

from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompt ────────────────────────────────────────────────────────────────────

NEAR_MISS_PROMPT = """\
IMPORTANT: This is an ANIMATED workplace safety video.

Classify this video into EXACTLY ONE of the following three categories:

1. "accident"   — A human character makes physical contact with a hazard,
                  falls, collapses, is struck, or is clearly injured.

2. "near_miss"  — A human character is in danger and narrowly avoids harm:
                  no physical contact occurs, no fall/collapse completes,
                  but the character was clearly at risk.

3. "safe"       — No hazard interaction occurs. Normal activity shown.

Return ONLY JSON:
{
  "incident_type": "accident | near_miss | safe",
  "near_miss_category": "potential_fall | potential_electrocution | potential_slip | \
potential_trip | potential_struck_by_object | potential_caught_in_machine | \
potential_arc_flash | potential_fire | potential_gas_inhalation | \
potential_vehicle_incident | potential_lifting | none",
  "confidence": 0.0-1.0,
  "description": "brief factual description"
}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class NearMissResult:
    incident_type: str          # "accident" | "near_miss" | "safe"
    near_miss_category: str     # "potential_fall" | ... | "none"
    confidence: float
    description: str
    raw_response: str
    latency_s: float

    @property
    def is_accident(self) -> bool:
        return self.incident_type == "accident"

    @property
    def is_near_miss(self) -> bool:
        return self.incident_type == "near_miss"

    @property
    def is_safe(self) -> bool:
        return self.incident_type == "safe"


VALID_INCIDENT_TYPES = {"accident", "near_miss", "safe"}


# ── Core logic ────────────────────────────────────────────────────────────────

def classify_near_miss(
    video_part: Part,
    model: GenerativeModel | None = None,
    temperature: float | None = None,
) -> NearMissResult:
    """
    Classify a video as accident, near-miss, or safe.
    """
    _model = model or vertex_model
    _temp = temperature if temperature is not None else config.temperature

    t0 = time.perf_counter()
    response = generate_with_retry(
        model=_model,
        parts=[video_part, NEAR_MISS_PROMPT],
        generation_config={
            "temperature": _temp,
            "response_mime_type": "application/json",
        },
    )
    elapsed = time.perf_counter() - t0

    raw = response.text
    parsed = safe_parse_json(raw) or {}

    incident_type = parsed.get("incident_type", "safe")
    if incident_type not in VALID_INCIDENT_TYPES:
        incident_type = "safe"

    return NearMissResult(
        incident_type=incident_type,
        near_miss_category=parsed.get("near_miss_category", "none"),
        confidence=float(parsed.get("confidence", 0.5)),
        description=parsed.get("description", ""),
        raw_response=raw,
        latency_s=elapsed,
    )
