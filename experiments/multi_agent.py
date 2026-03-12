"""
Direction 5 — Multi-agent validation pipeline (judge/jury).

Three-agent pattern:
  Agent 1 (Classifier) → initial prediction
  Agent 2 (Verifier)   → agree / disagree + correction
  Agent 3 (Judge)      → final arbitration (only on disagreement; uses stronger model)
"""

from __future__ import annotations

from dataclasses import dataclass

from vertexai.generative_models import GenerativeModel, Part

from config.categories import MERGED_CATEGORIES, DEFAULT_FALLBACK_CATEGORY
from config.settings import config
from pipeline.client import get_model, vertex_model
from pipeline.postprocessing import generate_with_retry, safe_parse_json


# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """\
IMPORTANT: This is an ANIMATED workplace safety video.

Classify the PRIMARY accident type into EXACTLY ONE of:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire,
Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident

Return ONLY JSON:
{ "category": "ONE exact category", "confidence": 0.0-1.0 }
"""

VERIFIER_PROMPT_TEMPLATE = """\
IMPORTANT: This is an ANIMATED workplace safety video.

A previous model classified this video as: "{initial_category}"

Do you agree with this classification?

Valid categories:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire,
Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident

Return ONLY JSON:
{{
  "agree": true/false,
  "category": "your category (same if agree, corrected if not)",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
"""

JUDGE_PROMPT_TEMPLATE = """\
IMPORTANT: This is an ANIMATED workplace safety video.

Two models disagree on the accident classification:
  Model 1 classified it as: "{category_1}"
  Model 2 classified it as: "{category_2}"

Review the video and make the final determination.

Valid categories:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire,
Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident

Return ONLY JSON:
{{
  "final_category": "ONE exact category",
  "confidence": 0.0-1.0,
  "reasoning": "explanation of decision"
}}
"""


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MultiAgentResult:
    final_category: str
    classifier_category: str
    verifier_category: str
    judge_used: bool
    judge_category: str | None
    classifier_confidence: float
    verifier_confidence: float
    judge_confidence: float | None
    verifier_reasoning: str
    judge_reasoning: str | None
    agreement: bool


# ── Core logic ────────────────────────────────────────────────────────────────

def multi_agent_classify(
    video_part: Part,
    classifier_model: GenerativeModel | None = None,
    verifier_model: GenerativeModel | None = None,
    judge_model_name: str = "gemini-2.5-pro",
    temperature: float = 0.0,
) -> MultiAgentResult:
    """
    Run the three-agent classification pipeline.

    Agent 3 (judge) is only called when Agents 1 and 2 disagree.
    """
    _classifier = classifier_model or vertex_model
    _verifier = verifier_model or vertex_model
    _temp = temperature

    # ── Agent 1: Classifier ───────────────────────────────────────────────────
    resp1 = generate_with_retry(
        model=_classifier,
        parts=[video_part, CLASSIFIER_PROMPT],
        generation_config={
            "temperature": _temp,
            "response_mime_type": "application/json",
        },
    )
    parsed1 = safe_parse_json(resp1.text) or {}
    cat1 = parsed1.get("category", DEFAULT_FALLBACK_CATEGORY)
    if cat1 not in MERGED_CATEGORIES:
        cat1 = DEFAULT_FALLBACK_CATEGORY
    conf1 = float(parsed1.get("confidence", 0.5))

    # ── Agent 2: Verifier ─────────────────────────────────────────────────────
    verifier_prompt = VERIFIER_PROMPT_TEMPLATE.format(initial_category=cat1)
    resp2 = generate_with_retry(
        model=_verifier,
        parts=[video_part, verifier_prompt],
        generation_config={
            "temperature": _temp,
            "response_mime_type": "application/json",
        },
    )
    parsed2 = safe_parse_json(resp2.text) or {}
    agree = bool(parsed2.get("agree", True))
    cat2 = parsed2.get("category", cat1)
    if cat2 not in MERGED_CATEGORIES:
        cat2 = cat1
    conf2 = float(parsed2.get("confidence", 0.5))
    reasoning2 = parsed2.get("reasoning", "")

    # ── Agent 3: Judge (only on disagreement) ─────────────────────────────────
    judge_used = False
    cat3: str | None = None
    conf3: float | None = None
    reasoning3: str | None = None

    if not agree and cat1 != cat2:
        judge_used = True
        judge_model = get_model(judge_model_name)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            category_1=cat1, category_2=cat2
        )
        resp3 = generate_with_retry(
            model=judge_model,
            parts=[video_part, judge_prompt],
            generation_config={
                "temperature": _temp,
                "response_mime_type": "application/json",
            },
        )
        parsed3 = safe_parse_json(resp3.text) or {}
        cat3 = parsed3.get("final_category", cat2)
        if cat3 not in MERGED_CATEGORIES:
            cat3 = cat2
        conf3 = float(parsed3.get("confidence", 0.5))
        reasoning3 = parsed3.get("reasoning", "")

    final_category = cat3 if judge_used and cat3 else (cat2 if agree else cat1)

    return MultiAgentResult(
        final_category=final_category,
        classifier_category=cat1,
        verifier_category=cat2,
        judge_used=judge_used,
        judge_category=cat3,
        classifier_confidence=conf1,
        verifier_confidence=conf2,
        judge_confidence=conf3,
        verifier_reasoning=reasoning2,
        judge_reasoning=reasoning3,
        agreement=agree,
    )
