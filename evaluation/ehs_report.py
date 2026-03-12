"""
Direction 10 (Optional) — EHS incident report auto-population.

Maps structured accident output fields to standard EHS/OSHA incident report fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── OSHA recordable injury type mapping ──────────────────────────────────────
OSHA_CATEGORY_MAP: dict[str, str] = {
    "Arc Flash":          "Contact with Electric Current",
    "Caught In Machine":  "Caught In/Under/Between",
    "Electrocution":      "Contact with Electric Current",
    "Fall":               "Fall to Lower Level",
    "Fire":               "Exposure to Harmful Substance or Environment",
    "Gas Inhalation":     "Exposure to Harmful Substance or Environment",
    "Lifting":            "Overexertion and Bodily Reaction",
    "Slip":               "Fall on Same Level",
    "Struck by Object":   "Struck by Object",
    "Trip":               "Fall on Same Level",
    "Vehicle Incident":   "Transportation Incidents",
}

# ── Severity tier mapping ─────────────────────────────────────────────────────
SEVERITY_MAP: dict[str, str] = {
    "Arc Flash":          "Critical",
    "Electrocution":      "Critical",
    "Fire":               "Critical",
    "Gas Inhalation":     "Critical",
    "Caught In Machine":  "High",
    "Vehicle Incident":   "High",
    "Struck by Object":   "Medium",
    "Fall":               "Medium",
    "Lifting":            "Low",
    "Slip":               "Low",
    "Trip":               "Low",
}


# ── EHS Report dataclass ──────────────────────────────────────────────────────

@dataclass
class EHSIncidentReport:
    # ── Identification ──────────────────────────────────────────────────────
    video_id: str
    incident_date: str = ""
    report_generated_at: str = ""

    # ── Classification ──────────────────────────────────────────────────────
    incident_type: str = ""               # Our category (e.g. "Fall")
    incident_components: list[str] = field(default_factory=list)  # Decomposed
    osha_classification: str = ""         # OSHA event type
    severity: str = ""                    # Critical / High / Medium / Low
    near_miss: bool = False

    # ── Timing ──────────────────────────────────────────────────────────────
    incident_start_time: str | None = None
    incident_end_time: str | None = None

    # ── Narrative ───────────────────────────────────────────────────────────
    description: str = ""
    root_cause: str = ""
    corrective_actions: list[str] = field(default_factory=list)

    # ── Regulatory ──────────────────────────────────────────────────────────
    osha_recordable: bool = True

    # ── Model metadata ──────────────────────────────────────────────────────
    model_confidence: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "incident_date": self.incident_date,
            "report_generated_at": self.report_generated_at,
            "incident_type": self.incident_type,
            "incident_components": self.incident_components,
            "osha_classification": self.osha_classification,
            "severity": self.severity,
            "near_miss": self.near_miss,
            "incident_start_time": self.incident_start_time,
            "incident_end_time": self.incident_end_time,
            "description": self.description,
            "root_cause": self.root_cause,
            "corrective_actions": self.corrective_actions,
            "osha_recordable": self.osha_recordable,
            "model_confidence": self.model_confidence,
            "model_name": self.model_name,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

def build_ehs_report(
    video_id: str,
    category: str,
    components: list[str] | None = None,
    near_miss: bool = False,
    incident_start_time: str | None = None,
    incident_end_time: str | None = None,
    description: str = "",
    root_cause_analysis: str = "",
    confidence: float = 0.0,
    model_name: str = "",
) -> EHSIncidentReport:
    """
    Build a structured EHS incident report from classification output.
    """
    from datetime import datetime

    osha_cls = OSHA_CATEGORY_MAP.get(category, "Other")
    severity = SEVERITY_MAP.get(category, "Unknown")
    osha_recordable = severity in ("Critical", "High", "Medium")

    # Generate basic corrective actions based on category
    corrective_actions = _suggest_corrective_actions(category)

    return EHSIncidentReport(
        video_id=video_id,
        report_generated_at=datetime.now().isoformat(),
        incident_type=category,
        incident_components=components or [category],
        osha_classification=osha_cls,
        severity=severity,
        near_miss=near_miss,
        incident_start_time=incident_start_time,
        incident_end_time=incident_end_time,
        description=description,
        root_cause=root_cause_analysis,
        corrective_actions=corrective_actions,
        osha_recordable=osha_recordable,
        model_confidence=confidence,
        model_name=model_name,
    )


def _suggest_corrective_actions(category: str) -> list[str]:
    """Return basic corrective action suggestions for a category."""
    suggestions: dict[str, list[str]] = {
        "Slip":           ["Apply non-slip flooring", "Post wet floor signs", "Improve drainage"],
        "Trip":           ["Mark and remove floor hazards", "Improve housekeeping", "Enhance lighting"],
        "Fall":           ["Install guardrails", "Require fall protection PPE", "Mark elevation changes"],
        "Electrocution":  ["Lockout/Tagout procedure audit", "Insulate exposed conductors", "Electrical safety training"],
        "Arc Flash":      ["Arc flash hazard analysis", "Require PPE (arc-rated)", "Update electrical labels"],
        "Fire":           ["Inspect fire suppression systems", "Review hot work permits", "Conduct fire drill"],
        "Gas Inhalation": ["Install gas monitors", "Improve ventilation", "Require SCBA in confined spaces"],
        "Caught In Machine": ["Guard all moving parts", "Enforce LOTO procedures", "Machinery safety audit"],
        "Struck by Object": ["Demarcate dropped-object exclusion zones", "Require hard hats", "Secure elevated materials"],
        "Lifting":        ["Implement mechanical lifting aids", "Ergonomics training", "Weight limit signage"],
        "Vehicle Incident": ["Separate pedestrian and vehicle lanes", "Speed limit enforcement", "Spotter requirements"],
    }
    return suggestions.get(category, ["Conduct incident investigation", "Review safety procedures"])
