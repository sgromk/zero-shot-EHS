"""
EHS Incident Detection API

Two-stage Gemini pipeline exposed as an HTTP endpoint for integration testing
and production deployment demonstration.

Architecture
------------
POST /classify
  Accepts a raw video file (multipart/form-data).
  Runs Stage 1 binary gate (3-vote ensemble, high-recall prompt).
  If no accident detected: returns immediately — Stage 2 is not called.
  If accident detected: runs Stage 2 CoT classification + OSHA EHS report.

GET /health
  Returns model configuration and service status.

GET /metrics
  Returns aggregate request counters and latency statistics since startup.

Usage
-----
    # From repo root:
    uvicorn api.endpoint:app --host 0.0.0.0 --port 8000

    # Test with curl:
    curl -X POST localhost:8000/classify \\
         -F "video=@data/videos/VID001_Electric_Forklift.../original.mp4"

Configuration
-------------
MODEL_NAME, N_VOTES, and sampling parameters are set at module level below.
Override via environment variables for deployment:
    EHS_STAGE1_MODEL, EHS_STAGE2_MODEL, EHS_N_VOTES, EHS_TEMPERATURE
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ── Repo root on sys.path ──────────────────────────────────────────────────────
_repo_root = str(Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_repo_root, ".env"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline.client import get_model
from pipeline.classification import classify, CLASSIFICATION_PROMPT_STRUCTURED
from pipeline.detection import detect, BINARY_PROMPT_HIGH_RECALL
from run_logging.experiment_logger import (
    COST_PER_1K_INPUT_TOKENS,
    COST_PER_1K_OUTPUT_TOKENS,
    _STAGE1_INPUT_TOKENS,
    _STAGE1_OUTPUT_TOKENS,
    _STAGE2_INPUT_TOKENS,
    _STAGE2_OUTPUT_TOKENS,
)

# ── Runtime configuration ─────────────────────────────────────────────────────
STAGE1_MODEL = os.getenv("EHS_STAGE1_MODEL", "gemini-2.5-flash")
STAGE2_MODEL = os.getenv("EHS_STAGE2_MODEL", "gemini-2.5-flash")
N_VOTES      = int(os.getenv("EHS_N_VOTES", "3"))
TEMPERATURE  = float(os.getenv("EHS_TEMPERATURE", "0.7"))
TOP_K        = int(os.getenv("EHS_TOP_K", "40"))
TOP_P        = float(os.getenv("EHS_TOP_P", "0.95"))
VOTE_POLICY  = "any"

ACCEPTED_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
ACCEPTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}

# ── In-memory metrics (thread-safe) ───────────────────────────────────────────
_lock         = threading.Lock()
_total_reqs   = 0
_total_s1_ms: list[float] = []
_total_s2_ms: list[float] = []
_detections   = 0
_errors       = 0


def _record(s1_ms: float, s2_ms: float, detected: bool) -> None:
    global _total_reqs, _detections
    with _lock:
        _total_reqs += 1
        _total_s1_ms.append(s1_ms)
        if detected:
            _detections += 1
            _total_s2_ms.append(s2_ms)


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── Cost helper ───────────────────────────────────────────────────────────────

def _estimate_cost(stage1_detected: bool) -> float:
    s1_in  = COST_PER_1K_INPUT_TOKENS.get(STAGE1_MODEL,  0.000075)
    s1_out = COST_PER_1K_OUTPUT_TOKENS.get(STAGE1_MODEL, 0.000300)
    cost = N_VOTES * (
        _STAGE1_INPUT_TOKENS  / 1000 * s1_in +
        _STAGE1_OUTPUT_TOKENS / 1000 * s1_out
    )
    if stage1_detected:
        s2_in  = COST_PER_1K_INPUT_TOKENS.get(STAGE2_MODEL,  0.000075)
        s2_out = COST_PER_1K_OUTPUT_TOKENS.get(STAGE2_MODEL, 0.000300)
        cost += (
            _STAGE2_INPUT_TOKENS  / 1000 * s2_in +
            _STAGE2_OUTPUT_TOKENS / 1000 * s2_out
        )
    return round(cost, 6)


# ── Pydantic response models ───────────────────────────────────────────────────

class IncidentCategory(BaseModel):
    category: str
    confidence: float


class EHSReport(BaseModel):
    severity: Optional[str] = None
    pre_incident_activity: Optional[str] = None
    what_happened: Optional[str] = None
    injury_description: Optional[str] = None
    direct_agent: Optional[str] = None
    immediate_actions: Optional[str] = None
    root_cause: Optional[str] = None
    contributing_factors: Optional[str] = None
    corrective_measures: Optional[str] = None


class ClassifyResponse(BaseModel):
    incident_detected: bool
    primary_category: Optional[str] = None
    categories: list[IncidentCategory] = []
    reasoning: Optional[str] = None
    incident_start_time: Optional[str] = None
    incident_end_time: Optional[str] = None
    confidence: float
    description: Optional[str] = None
    ehs_report: Optional[EHSReport] = None
    stage1_votes: list[bool]
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms: float
    estimated_cost_usd: float
    stage1_model: str
    stage2_model: str


class HealthResponse(BaseModel):
    status: str
    stage1_model: str
    stage2_model: str
    config: dict


class MetricsResponse(BaseModel):
    total_requests: int
    total_detections: int
    detection_rate: float
    stage1_latency_p50_ms: float
    stage1_latency_p95_ms: float
    stage2_latency_p50_ms: float
    stage2_latency_p95_ms: float
    error_count: int


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EHS Incident Detection API",
    description=(
        "Real-time workplace accident detection using a two-stage Gemini pipeline. "
        "Stage 1 is a high-recall binary gate (3-vote ensemble). "
        "Stage 2 produces multi-label classification and a structured OSHA 300-series "
        "EHS incident report, running only on clips where Stage 1 detected an accident."
    ),
    version="1.0.0",
    contact={"name": "EHS Capstone Project"},
    license_info={"name": "Internal research use"},
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and configuration",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Liveness probe. Returns the active model configuration so callers can
    verify which model tier and prompt strategy is deployed.
    """
    return HealthResponse(
        status="ok",
        stage1_model=STAGE1_MODEL,
        stage2_model=STAGE2_MODEL,
        config={
            "n_votes":       N_VOTES,
            "vote_policy":   VOTE_POLICY,
            "temperature":   TEMPERATURE,
            "top_k":         TOP_K,
            "top_p":         TOP_P,
            "stage1_prompt": "high_recall",
            "stage2_prompt": "structured_cot_multi_label",
        },
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Aggregate request statistics since startup",
    tags=["System"],
)
def metrics() -> MetricsResponse:
    """
    Returns in-memory aggregate counters and latency percentiles for all
    requests handled since the server started. Resets on restart.
    """
    with _lock:
        n = _total_reqs
        det = _detections
        err = _errors
        s1  = list(_total_s1_ms)
        s2  = list(_total_s2_ms)

    return MetricsResponse(
        total_requests=n,
        total_detections=det,
        detection_rate=round(det / n, 4) if n else 0.0,
        stage1_latency_p50_ms=round(_pct(s1, 50), 1),
        stage1_latency_p95_ms=round(_pct(s1, 95), 1),
        stage2_latency_p50_ms=round(_pct(s2, 50), 1),
        stage2_latency_p95_ms=round(_pct(s2, 95), 1),
        error_count=err,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a video clip for workplace accidents",
    tags=["Detection"],
    responses={
        200: {"description": "Classification completed (accident detected or not)"},
        422: {"description": "Invalid file format or empty payload"},
        500: {"description": "Upstream model error"},
    },
)
async def classify_video(
    video: UploadFile = File(..., description="Video file to analyse (mp4/mov/avi/webm)"),
) -> ClassifyResponse:
    """
    Run the two-stage EHS detection pipeline on a video clip.

    **Stage 1 — Binary gate**
    Fires `n_votes=3` independent Gemini calls in parallel using the
    `high_recall` prompt. If *any* vote flags an accident, the clip proceeds
    to Stage 2. Non-accident clips stop here, incurring only the Stage 1 cost.

    **Stage 2 — Classification and EHS report** *(accident clips only)*
    A single Gemini call with the structured chain-of-thought prompt returns:
    - Ranked multi-label incident classification (all types with confidence ≥ 0.5)
    - CoT scene observation (`reasoning` field)
    - Full OSHA 300-series EHS incident report

    **Cost model** (Gemini 2.5 Flash, 3-vote ensemble)
    - Non-accident clip: ~$0.000207
    - Accident clip:     ~$0.000469
    """
    # ── Input validation ──────────────────────────────────────────────────────
    filename = video.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension {ext!r}. Accepted: {sorted(ACCEPTED_EXTENSIONS)}",
        )

    video_bytes = await video.read()
    if len(video_bytes) < 2048:
        raise HTTPException(status_code=422, detail="Video payload is too small or empty.")

    # ── Build Vertex AI Part ──────────────────────────────────────────────────
    try:
        from vertexai.generative_models import Part
        video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
        s1_model = get_model(STAGE1_MODEL)
        s2_model = get_model(STAGE2_MODEL)
    except Exception as exc:
        with _lock:
            global _errors
            _errors += 1
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {exc}") from exc

    # ── Stage 1 — Binary gate ─────────────────────────────────────────────────
    try:
        detection = detect(
            video_part=video_part,
            model=s1_model,
            prompt=BINARY_PROMPT_HIGH_RECALL,
            n_votes=N_VOTES,
            vote_policy=VOTE_POLICY,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        )
    except Exception as exc:
        with _lock:
            _errors += 1
        raise HTTPException(status_code=500, detail=f"Stage 1 detection failed: {exc}") from exc

    s1_ms = round(detection.latency_s * 1000, 1)

    # ── Early return: no accident ─────────────────────────────────────────────
    if not detection.incident_detected:
        _record(s1_ms, 0.0, detected=False)
        return ClassifyResponse(
            incident_detected=False,
            primary_category="No Accident",
            categories=[],
            confidence=round(detection.confidence, 4),
            stage1_votes=detection.votes,
            stage1_latency_ms=s1_ms,
            stage2_latency_ms=0.0,
            total_latency_ms=s1_ms,
            estimated_cost_usd=_estimate_cost(False),
            stage1_model=STAGE1_MODEL,
            stage2_model=STAGE2_MODEL,
        )

    # ── Stage 2 — Classification and EHS report ───────────────────────────────
    try:
        classification = classify(
            video_part=video_part,
            model=s2_model,
            prompt=CLASSIFICATION_PROMPT_STRUCTURED,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        )
    except Exception as exc:
        with _lock:
            _errors += 1
        raise HTTPException(status_code=500, detail=f"Stage 2 classification failed: {exc}") from exc

    s2_ms = round(classification.latency_s * 1000, 1)
    _record(s1_ms, s2_ms, detected=True)

    ehs: Optional[EHSReport] = None
    if classification.ehs_report:
        try:
            ehs = EHSReport(**classification.ehs_report)
        except Exception:
            ehs = None

    return ClassifyResponse(
        incident_detected=True,
        primary_category=classification.category,
        categories=[IncidentCategory(**c) for c in classification.categories],
        reasoning=classification.reasoning or None,
        incident_start_time=classification.incident_start_time,
        incident_end_time=classification.incident_end_time,
        confidence=round(classification.confidence, 4),
        description=classification.description or None,
        ehs_report=ehs,
        stage1_votes=detection.votes,
        stage1_latency_ms=s1_ms,
        stage2_latency_ms=s2_ms,
        total_latency_ms=round(s1_ms + s2_ms, 1),
        estimated_cost_usd=_estimate_cost(True),
        stage1_model=STAGE1_MODEL,
        stage2_model=STAGE2_MODEL,
    )
