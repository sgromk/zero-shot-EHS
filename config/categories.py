"""
Accident category definitions, severity ordering, and near-miss mappings.
"""

# ── Core classification labels ─────────────────────────────────────────────────
MERGED_CATEGORIES: list[str] = [
    "Arc Flash",
    "Caught In Machine",
    "Electrocution",
    "Fall",
    "Fire",
    "Gas Inhalation",
    "Lifting",
    "Slip",
    "Struck by Object",
    "Trip",
    "Vehicle Incident",
]

# ── Severity ranking (most → least severe) ───────────────────────────────────
SEVERITY_ORDER: list[str] = [
    "Arc Flash",
    "Electrocution",
    "Fire",
    "Gas Inhalation",
    "Caught In Machine",
    "Vehicle Incident",
    "Struck by Object",
    "Fall",
    "Lifting",
    "Slip",
    "Trip",
]

# ── Near-miss potential category mapping ─────────────────────────────────────
NEAR_MISS_CATEGORIES: list[str] = [
    "Potential Arc Flash",
    "Potential Caught In Machine",
    "Potential Electrocution",
    "Potential Fall",
    "Potential Fire",
    "Potential Gas Inhalation",
    "Potential Lifting Injury",
    "Potential Slip",
    "Potential Struck by Object",
    "Potential Trip",
    "Potential Vehicle Incident",
]

# ── Fallback when category parse fails ────────────────────────────────────────
DEFAULT_FALLBACK_CATEGORY = "Fall"

# ── Binary labels ─────────────────────────────────────────────────────────────
LABEL_ACCIDENT = "Accident"
LABEL_NO_ACCIDENT = "No Accident"
LABEL_NEAR_MISS = "Near Miss"

# ── Ground-truth category normalization ───────────────────────────────────────
# The spreadsheet contains compound/variant labels that don't appear in
# MERGED_CATEGORIES. Map them to the nearest canonical label.
GT_CATEGORY_NORMALIZATION: dict[str, str] = {
    # Case inconsistency
    "Caught in Machine": "Caught In Machine",
    # Compound labels — map to the causal mechanism (the slip/trip caused the fall)
    "Trip and Fall": "Trip",
    "Slip and Fall": "Slip",
    # When two events co-occur, keep the more severe
    "Electrocution and Fall": "Electrocution",
    # No-accident rows
    "No Accident": "No Accident",
}


def normalize_gt_category(category: str) -> str:
    """
    Normalise a raw ground-truth incident_type string to a canonical label.

    Strips whitespace, applies GT_CATEGORY_NORMALIZATION, and returns the
    canonical MERGED_CATEGORIES label (or the original value if already valid).
    """
    if not isinstance(category, str):
        return "No Accident"
    category = category.strip()
    return GT_CATEGORY_NORMALIZATION.get(category, category)


def severity_rank(category: str) -> int:
    """Return the severity rank of a category (lower = more severe). Unknown → 999."""
    try:
        return SEVERITY_ORDER.index(category)
    except ValueError:
        return 999


def sort_by_severity(categories: list[str]) -> list[str]:
    """Sort a list of category strings from most to least severe."""
    return sorted(categories, key=severity_rank)
