"""
EHS Incident Detection API

Two-stage Gemini pipeline exposed as an HTTP endpoint for integration testing
and production deployment demonstration.

Architecture
------------
POST /classify
  Synchronous. Accepts a raw video file (multipart/form-data).
  Runs Stage 1 binary gate (3-vote ensemble, high-recall prompt).
  If no accident detected: returns immediately — Stage 2 is not called.
  If accident detected: runs Stage 2 CoT classification + OSHA EHS report.

POST /submit
  Async job submission. Returns a job_id immediately (<100ms).
  Use GET /status/{job_id} to poll for the result.

GET /status/{job_id}
  Poll for job result. Status: "queued" | "processing" | "complete" | "failed".
  Results expire after 10 minutes.

GET /health
  Returns model configuration and service status.

GET /metrics
  Returns aggregate request counters, latency statistics, and queue depth
  since startup.

Usage
-----
    # From repo root:
    uvicorn api.endpoint:app --host 0.0.0.0 --port 8000

    # Test with curl:
    curl -X POST localhost:8000/classify \\
         -F "video=@data/videos/VID001_Electric_Forklift.../original.mp4"

    # Async job pattern:
    curl -X POST localhost:8000/submit \\
         -F "video=@data/videos/VID001_.../original.mp4"
    # → {"job_id": "abc123", "status": "queued"}
    curl localhost:8000/status/abc123

Configuration
-------------
MODEL_NAME, N_VOTES, and sampling parameters are set at module level below.
Override via environment variables for deployment:
    EHS_STAGE1_MODEL, EHS_STAGE2_MODEL, EHS_N_VOTES, EHS_TEMPERATURE
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# ── Repo root on sys.path ──────────────────────────────────────────────────────
_repo_root = str(Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_repo_root, ".env"))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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

MAX_VIDEO_BYTES   = 500 * 1024 * 1024   # 500 MB hard limit
ACCEPTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
JOB_TTL_S         = 600                 # job results expire after 10 minutes

# ── Structured logging (GCP Cloud Logging picks up stdout JSON) ───────────────

def _log(severity: str, message: str, **kv: Any) -> None:
    print(json.dumps({"severity": severity, "message": message, **kv}), flush=True)


# ── In-memory metrics (thread-safe) ───────────────────────────────────────────
_lock         = threading.Lock()
_total_reqs   = 0
_in_flight    = 0
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


# ── Async job store ───────────────────────────────────────────────────────────
# Supports the POST /submit → GET /status/{job_id} pattern for clients that
# cannot hold a connection open for the 8–20s pipeline duration.

_job_queue:  asyncio.Queue   = None   # type: ignore  (set at startup)
_job_store:  dict[str, dict] = {}     # job_id → {status, result, created_at}
_job_lock:   asyncio.Lock    = None   # type: ignore  (set at startup)


async def _job_worker() -> None:
    """Background task: drains _job_queue and runs the pipeline for each job."""
    while True:
        job_id, video_bytes, filename = await _job_queue.get()
        async with _job_lock:
            if job_id in _job_store:
                _job_store[job_id]["status"] = "processing"

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_pipeline, video_bytes, filename, job_id
            )
            async with _job_lock:
                if job_id in _job_store:
                    _job_store[job_id]["status"] = "complete"
                    _job_store[job_id]["result"] = result
        except Exception as exc:
            _log("ERROR", "job_failed", job_id=job_id, error=str(exc))
            async with _job_lock:
                if job_id in _job_store:
                    _job_store[job_id]["status"] = "failed"
                    _job_store[job_id]["error"] = str(exc)
        finally:
            _job_queue.task_done()

        # TTL cleanup — remove expired jobs
        now = time.time()
        async with _job_lock:
            expired = [jid for jid, j in _job_store.items()
                       if now - j["created_at"] > JOB_TTL_S]
            for jid in expired:
                del _job_store[jid]


def _run_pipeline(video_bytes: bytes, filename: str, request_id: str) -> dict:
    """Synchronous pipeline execution — called from a thread executor."""
    from vertexai.generative_models import Part

    video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
    s1_model   = get_model(STAGE1_MODEL)
    s2_model   = get_model(STAGE2_MODEL)

    # Stage 1
    det = detect(
        video_part=video_part,
        model=s1_model,
        prompt=BINARY_PROMPT_HIGH_RECALL,
        n_votes=N_VOTES,
        vote_policy=VOTE_POLICY,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    s1_ms = round(det.latency_s * 1000, 1)

    if not det.incident_detected:
        _record(s1_ms, 0.0, detected=False)
        _log("INFO", "classify_complete", request_id=request_id,
             incident_detected=False, stage1_ms=s1_ms, stage2_ms=0,
             cost_usd=_estimate_cost(False))
        return dict(
            incident_detected=False,
            primary_category="No Accident",
            categories=[],
            confidence=round(det.confidence, 4),
            stage1_votes=det.votes,
            stage1_latency_ms=s1_ms,
            stage2_latency_ms=0.0,
            total_latency_ms=s1_ms,
            estimated_cost_usd=_estimate_cost(False),
            stage1_model=STAGE1_MODEL,
            stage2_model=STAGE2_MODEL,
        )

    # Stage 2
    cls = classify(
        video_part=video_part,
        model=s2_model,
        prompt=CLASSIFICATION_PROMPT_STRUCTURED,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    s2_ms = round(cls.latency_s * 1000, 1)
    _record(s1_ms, s2_ms, detected=True)
    _log("INFO", "classify_complete", request_id=request_id,
         incident_detected=True, primary_category=cls.category,
         stage1_ms=s1_ms, stage2_ms=s2_ms,
         cost_usd=_estimate_cost(True))

    ehs = dict(cls.ehs_report) if cls.ehs_report else None
    return dict(
        incident_detected=True,
        primary_category=cls.category,
        categories=cls.categories,
        reasoning=cls.reasoning or None,
        incident_start_time=cls.incident_start_time,
        incident_end_time=cls.incident_end_time,
        confidence=round(cls.confidence, 4),
        description=cls.description or None,
        ehs_report=ehs,
        stage1_votes=det.votes,
        stage1_latency_ms=s1_ms,
        stage2_latency_ms=s2_ms,
        total_latency_ms=round(s1_ms + s2_ms, 1),
        estimated_cost_usd=_estimate_cost(True),
        stage1_model=STAGE1_MODEL,
        stage2_model=STAGE2_MODEL,
    )


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
    request_id: str


class SubmitResponse(BaseModel):
    job_id: str
    status: str   # "queued"


class StatusResponse(BaseModel):
    job_id: str
    status: str   # "queued" | "processing" | "complete" | "failed"
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    stage1_model: str
    stage2_model: str
    config: dict


class MetricsResponse(BaseModel):
    total_requests: int
    requests_in_flight: int
    total_detections: int
    detection_rate: float
    stage1_latency_p50_ms: float
    stage1_latency_p95_ms: float
    stage2_latency_p50_ms: float
    stage2_latency_p95_ms: float
    error_count: int
    async_queue_depth: int


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EHS Incident Detection API",
    description=(
        "Real-time workplace accident detection using a two-stage Gemini pipeline. "
        "Stage 1 is a high-recall binary gate (3-vote ensemble). "
        "Stage 2 produces multi-label classification and a structured OSHA 300-series "
        "EHS incident report, running only on clips where Stage 1 detected an accident."
    ),
    version="1.1.0",
    contact={"name": "EHS Capstone Project"},
    license_info={"name": "Internal research use"},
)


@app.on_event("startup")
async def _startup() -> None:
    global _job_queue, _job_lock
    _job_queue = asyncio.Queue()
    _job_lock  = asyncio.Lock()
    asyncio.create_task(_job_worker())
    _log("INFO", "server_startup", stage1_model=STAGE1_MODEL,
         stage2_model=STAGE2_MODEL, n_votes=N_VOTES)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and configuration",
    tags=["System"],
)
def health() -> HealthResponse:
    """Liveness probe. Returns the active model configuration."""
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
    Includes current in-flight count and async queue depth.
    """
    with _lock:
        n        = _total_reqs
        inflight = _in_flight
        det      = _detections
        err      = _errors
        s1       = list(_total_s1_ms)
        s2       = list(_total_s2_ms)

    q_depth = _job_queue.qsize() if _job_queue else 0

    return MetricsResponse(
        total_requests=n,
        requests_in_flight=inflight,
        total_detections=det,
        detection_rate=round(det / n, 4) if n else 0.0,
        stage1_latency_p50_ms=round(_pct(s1, 50), 1),
        stage1_latency_p95_ms=round(_pct(s1, 95), 1),
        stage2_latency_p50_ms=round(_pct(s2, 50), 1),
        stage2_latency_p95_ms=round(_pct(s2, 95), 1),
        error_count=err,
        async_queue_depth=q_depth,
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a video clip for workplace accidents (synchronous)",
    tags=["Detection"],
    responses={
        200: {"description": "Classification completed (accident detected or not)"},
        422: {"description": "Invalid file format, empty payload, or file too large"},
        503: {"description": "Upstream model unavailable (quota exhausted or retries exceeded)"},
    },
)
async def classify_video(
    request: Request,
    video: UploadFile = File(..., description="Video file to analyse (mp4/mov/avi/webm)"),
) -> ClassifyResponse:
    """
    Run the two-stage EHS detection pipeline on a video clip.

    Blocks until the full pipeline completes (~8–20s). For clients with short
    timeouts, use `POST /submit` + `GET /status/{job_id}` instead.

    A `X-Request-ID` header is echoed in the response and all server log lines
    for distributed tracing.

    **Cost model** (Gemini 2.5 Flash, 3-vote ensemble)
    - Non-accident clip: ~$0.000207
    - Accident clip:     ~$0.000469
    """
    global _in_flight, _errors

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

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
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=422,
                            detail=f"Video exceeds {MAX_VIDEO_BYTES // 1_048_576} MB limit.")

    with _lock:
        _in_flight += 1
    _log("INFO", "classify_start", request_id=request_id,
         filename=filename, size_bytes=len(video_bytes))

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_pipeline, video_bytes, filename, request_id
        )
    except Exception as exc:
        with _lock:
            _errors += 1
        _log("ERROR", "classify_error", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"Pipeline error: {exc}") from exc
    finally:
        with _lock:
            _in_flight -= 1

    result["request_id"] = request_id

    # Coerce categories list into Pydantic models for the response
    result["categories"] = [
        IncidentCategory(**c) if isinstance(c, dict) else c
        for c in result.get("categories", [])
    ]
    if result.get("ehs_report") and isinstance(result["ehs_report"], dict):
        try:
            result["ehs_report"] = EHSReport(**result["ehs_report"])
        except Exception:
            result["ehs_report"] = None

    return ClassifyResponse(**result)


@app.post(
    "/submit",
    response_model=SubmitResponse,
    summary="Submit a video for async classification (non-blocking)",
    tags=["Detection"],
    responses={
        200: {"description": "Job accepted and queued"},
        422: {"description": "Invalid file format, empty payload, or file too large"},
    },
)
async def submit_video(
    request: Request,
    video: UploadFile = File(..., description="Video file to analyse (mp4/mov/avi/webm)"),
) -> SubmitResponse:
    """
    Accept a video clip and return a `job_id` immediately (<100ms).
    Poll `GET /status/{job_id}` for the result.

    Use this endpoint when your HTTP client has a timeout shorter than the
    pipeline duration (~8–20s for C=1, longer under concurrent load).
    """
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
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=422,
                            detail=f"Video exceeds {MAX_VIDEO_BYTES // 1_048_576} MB limit.")

    job_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    async with _job_lock:
        _job_store[job_id] = {"status": "queued", "created_at": time.time(), "result": None}

    await _job_queue.put((job_id, video_bytes, filename))
    _log("INFO", "job_queued", job_id=job_id, queue_depth=_job_queue.qsize())

    return SubmitResponse(job_id=job_id, status="queued")


@app.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="Poll for async job result",
    tags=["Detection"],
    responses={
        200: {"description": "Job status (queued / processing / complete / failed)"},
        404: {"description": "Job not found or expired (TTL=10 min)"},
    },
)
async def job_status(job_id: str) -> StatusResponse:
    """
    Returns the current status of a job submitted via `POST /submit`.
    Results are retained for 10 minutes after completion, then discarded.
    """
    async with _job_lock:
        job = _job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404,
                            detail=f"Job {job_id!r} not found or expired.")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )
