"""
Direction 4 — Sampling experiments (top_k / top_p sweeps).

Produces probabilistic candidate accident probabilities by running
multiple classification calls with varied sampling parameters,
then aggregating with heuristic post-processing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vertexai.generative_models import GenerativeModel, Part

from config.categories import MERGED_CATEGORIES
from config.settings import config
from pipeline.client import vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json

# ── Prompt ────────────────────────────────────────────────────────────────────

SAMPLING_CLASSIFICATION_PROMPT = """\
IMPORTANT: This is an ANIMATED workplace safety video.

Classify the PRIMARY accident type into EXACTLY ONE of the following:

Arc Flash, Caught In Machine, Electrocution, Fall, Fire,
Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident

Return ONLY JSON:
{
  "category": "ONE exact category",
  "confidence": 0.0-1.0
}
"""


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SamplingResult:
    """Probabilistic distribution over accident categories from N samples."""
    category_probs: dict[str, float]   # {"Fall": 0.5, "Trip": 0.3, ...}
    top_category: str
    multi_label: list[str]             # categories above threshold
    n_samples: int
    raw_categories: list[str]


# ── Core logic ────────────────────────────────────────────────────────────────

def sample_classifications(
    video_part: Part,
    model: GenerativeModel | None = None,
    n_samples: int = 10,
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    min_prob_threshold: float = 0.15,
) -> SamplingResult:
    """
    Run classification N times with stochastic sampling to produce
    a probability distribution over accident categories.

    Parameters
    ----------
    n_samples : int
        Number of classification calls to make.
    temperature : float
        Sampling temperature (higher = more diverse).
    top_k : int | None
        Nucleus top-k parameter.
    top_p : float | None
        Nucleus top-p parameter.
    min_prob_threshold : float
        Minimum probability to include in multi-label output.

    Returns
    -------
    SamplingResult with category_probs dict.
    """
    _model = model or vertex_model

    generation_config: dict = {"temperature": temperature}
    if top_k is not None:
        generation_config["top_k"] = top_k
    if top_p is not None:
        generation_config["top_p"] = top_p
    generation_config["response_mime_type"] = "application/json"

    raw_categories: list[str] = []

    for _ in range(n_samples):
        response = generate_with_retry(
            model=_model,
            parts=[video_part, SAMPLING_CLASSIFICATION_PROMPT],
            generation_config=generation_config,
        )
        parsed = safe_parse_json(response.text) or {}
        cat = parsed.get("category")
        if cat in MERGED_CATEGORIES:
            raw_categories.append(cat)

    if not raw_categories:
        return SamplingResult(
            category_probs={},
            top_category="Fall",
            multi_label=["Fall"],
            n_samples=n_samples,
            raw_categories=[],
        )

    counts = Counter(raw_categories)
    total = len(raw_categories)
    category_probs = {cat: count / total for cat, count in counts.most_common()}

    top_category = max(category_probs, key=category_probs.get)  # type: ignore[arg-type]

    # Heuristic: single-label if top > 0.5, else multi-label above threshold
    if category_probs[top_category] > 0.5:
        multi_label = [top_category]
    else:
        multi_label = [
            cat for cat, prob in category_probs.items() if prob >= min_prob_threshold
        ]

    return SamplingResult(
        category_probs=category_probs,
        top_category=top_category,
        multi_label=multi_label,
        n_samples=n_samples,
        raw_categories=raw_categories,
    )


def sampling_sweep(
    video_part: Part,
    top_k_values: list[int | None] | None = None,
    top_p_values: list[float | None] | None = None,
    temperatures: list[float] | None = None,
    n_samples: int = 10,
) -> list[dict]:
    """
    Run a sweep over top_k, top_p, and temperature combinations.

    Returns a list of result dicts suitable for logging.
    """
    _top_k_values = top_k_values or [None, 5, 10, 20, 40]
    _top_p_values = top_p_values or [None, 0.7, 0.9, 0.95]
    _temperatures = temperatures or [0.3, 0.5, 0.7]

    sweep_results = []

    for temp in _temperatures:
        for top_k in _top_k_values:
            for top_p in _top_p_values:
                result = sample_classifications(
                    video_part=video_part,
                    n_samples=n_samples,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                )
                sweep_results.append(
                    {
                        "temperature": temp,
                        "top_k": top_k,
                        "top_p": top_p,
                        "n_samples": n_samples,
                        "top_category": result.top_category,
                        "multi_label": result.multi_label,
                        "category_probs": result.category_probs,
                    }
                )

    return sweep_results
