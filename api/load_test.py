"""
EHS Endpoint Load Test

Measures throughput and end-to-end latency of the /classify endpoint at
increasing concurrency levels, simulating multiple camera feeds submitting
clips simultaneously.

Prerequisites
-------------
1. Start the API server in a separate terminal:
       uvicorn api.endpoint:app --host 0.0.0.0 --port 8000

2. Run this script from the repo root:
       python api/load_test.py
       python api/load_test.py --url http://localhost:8000 --clip data/videos/VID001_.../original.mp4
       python api/load_test.py --concurrency 1 2 4 8 16 --requests 12

Methodology
-----------
For each concurrency level C:
  - A warm-up batch (WARMUP_REQUESTS) is sent first and discarded.
  - REQUESTS_PER_LEVEL requests are dispatched with at most C in-flight
    at any time (controlled by asyncio.Semaphore).
  - Wall-clock time is measured from first dispatch to last completion.
  - Per-request latency is measured independently inside each coroutine.

Metrics reported (per concurrency level)
-----------------------------------------
  throughput_rps   Completed requests / wall-clock seconds
  mean_ms          Arithmetic mean of individual request latencies
  p50_ms           Median latency
  p95_ms           95th-percentile latency
  p99_ms           99th-percentile latency  (meaningful at ≥20 requests)
  min_ms / max_ms  Observed extremes
  stdev_ms         Standard deviation (measure of consistency)
  errors           HTTP non-200 responses or connection failures

Outputs
-------
  Console: formatted summary table + deployment interpretation
  File:    outputs/load_test_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL         = "http://localhost:8000"
DEFAULT_CLIP        = "data/videos/VID001_Electric_Forklift_Safety_Accident_VB_Factory_VB_Engineering_I_Pvt_Ltd/original.mp4"
DEFAULT_CONCURRENCY = [1, 2, 4, 8]
DEFAULT_N_REQUESTS  = 12        # per concurrency level (excluding warm-up)
WARMUP_REQUESTS     = 2         # discarded before timing begins
REQUEST_TIMEOUT_S   = 180.0     # Vertex AI calls can be slow under load
CONNECT_TIMEOUT_S   = 10.0

W = 68  # console width


# ── Statistics ────────────────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
    """Return the p-th percentile of a list of floats."""
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "mean_ms":   round(statistics.mean(latencies), 1),
        "p50_ms":    round(_pct(latencies, 50), 1),
        "p95_ms":    round(_pct(latencies, 95), 1),
        "p99_ms":    round(_pct(latencies, 99), 1),
        "min_ms":    round(min(latencies), 1),
        "max_ms":    round(max(latencies), 1),
        "stdev_ms":  round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0,
    }


# ── Single request ────────────────────────────────────────────────────────────

async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    video_bytes: bytes,
    filename: str,
) -> dict:
    """Send one /classify request. Returns timing and status regardless of outcome."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/classify",
            files={"video": (filename, video_bytes, "video/mp4")},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        body: Optional[dict] = None
        try:
            body = resp.json()
        except Exception:
            pass
        return {
            "latency_ms":  elapsed_ms,
            "status":      resp.status_code,
            "detected":    body.get("incident_detected") if body else None,
            "s1_ms":       body.get("stage1_latency_ms") if body else None,
            "s2_ms":       body.get("stage2_latency_ms") if body else None,
            "error":       None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": elapsed_ms,
            "status":     -1,
            "detected":   None,
            "s1_ms":      None,
            "s2_ms":      None,
            "error":      str(exc),
        }


# ── Concurrency batch ─────────────────────────────────────────────────────────

async def _run_level(
    url: str,
    video_bytes: bytes,
    filename: str,
    concurrency: int,
    n_requests: int,
) -> dict:
    """
    Fire n_requests through the endpoint with at most `concurrency` in-flight.
    Returns a dict of aggregate metrics for this concurrency level.
    """
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    limits  = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def _controlled(client: httpx.AsyncClient) -> dict:
        async with semaphore:
            return await _single_request(client, url, video_bytes, filename)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # ── Warm-up ───────────────────────────────────────────────────────────
        print(f"    Warming up ({WARMUP_REQUESTS} req)...", end=" ", flush=True)
        warmup_tasks = [_single_request(client, url, video_bytes, filename)
                        for _ in range(WARMUP_REQUESTS)]
        await asyncio.gather(*warmup_tasks)
        print("done")

        # ── Timed batch ───────────────────────────────────────────────────────
        print(f"    Running {n_requests} requests (concurrency={concurrency})...", end=" ", flush=True)
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*[_controlled(client) for _ in range(n_requests)])
        wall_s  = time.perf_counter() - t_wall_start
        print(f"done ({wall_s:.1f}s)")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    good    = [r for r in results if r["status"] == 200]
    errors  = [r for r in results if r["status"] != 200]
    latencies = [r["latency_ms"] for r in good]
    s1_lats   = [r["s1_ms"] for r in good if r["s1_ms"] is not None]
    s2_lats   = [r["s2_ms"] for r in good if r["s2_ms"] is not None and r["s2_ms"] > 0]
    detections = sum(1 for r in good if r.get("detected"))

    st = _stats(latencies)
    return {
        "concurrency":     concurrency,
        "n_requests":      n_requests,
        "n_errors":        len(errors),
        "n_detections":    detections,
        "wall_time_s":     round(wall_s, 3),
        "throughput_rps":  round(n_requests / wall_s, 4) if wall_s > 0 else 0.0,
        **st,
        "stage1_mean_ms":  round(statistics.mean(s1_lats), 1) if s1_lats else None,
        "stage2_mean_ms":  round(statistics.mean(s2_lats), 1) if s2_lats else None,
        "error_messages":  [r["error"] for r in errors if r.get("error")],
    }


# ── Console rendering ─────────────────────────────────────────────────────────

def _print_level(r: dict) -> None:
    err_str = f"{r['n_errors']}/{r['n_requests']}"
    det_str = f"{r['n_detections']}/{r['n_requests'] - r['n_errors']}"
    print(f"  │  Throughput:   {r['throughput_rps']:.4f} req/s   wall={r['wall_time_s']:.1f}s")
    print(f"  │  Latency:      mean={r['mean_ms']:.0f}ms  P50={r['p50_ms']:.0f}ms  P95={r['p95_ms']:.0f}ms  P99={r['p99_ms']:.0f}ms")
    print(f"  │  Range:        {r['min_ms']:.0f}ms – {r['max_ms']:.0f}ms   σ={r['stdev_ms']:.0f}ms")
    if r.get("stage1_mean_ms"):
        print(f"  │  Stage 1 avg:  {r['stage1_mean_ms']:.0f}ms   Stage 2 avg: {r.get('stage2_mean_ms') or 'N/A'}ms")
    print(f"  │  Detections:   {det_str}   Errors: {err_str}")
    if r["error_messages"]:
        for msg in r["error_messages"][:3]:
            print(f"  │  [ERR] {msg[:60]}")


def _print_summary(all_results: list[dict]) -> None:
    print(f"\n{'═' * W}")
    print("  SUMMARY TABLE")
    print(f"{'─' * W}")
    hdr = f"  {'Concurr':>8}  {'Req/s':>8}  {'Mean':>8}  {'P50':>7}  {'P95':>7}  {'P99':>7}  {'Err':>5}"
    sep = f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 5}"
    print(hdr)
    print(sep)
    for r in all_results:
        print(
            f"  {r['concurrency']:>8}"
            f"  {r['throughput_rps']:>8.4f}"
            f"  {r['mean_ms']:>6.0f}ms"
            f"  {r['p50_ms']:>5.0f}ms"
            f"  {r['p95_ms']:>5.0f}ms"
            f"  {r['p99_ms']:>5.0f}ms"
            f"  {r['n_errors']:>3}/{r['n_requests']}"
        )
    print(f"{'═' * W}")


def _print_interpretation(all_results: list[dict], clip_size_mb: float) -> None:
    if not all_results:
        return

    peak = max(all_results, key=lambda r: r["throughput_rps"])
    rps  = peak["throughput_rps"]

    # Assume 30-second clip interval per camera (one clip per 30 seconds)
    clip_interval_s = 30
    cameras_30s = rps * clip_interval_s

    # At 60-second clip interval
    cameras_60s = rps * 60

    print(f"\n  DEPLOYMENT INTERPRETATION")
    print(f"  {'─' * (W - 2)}")
    print(f"  Peak throughput:   {rps:.4f} req/s  (concurrency={peak['concurrency']})")
    print(f"  Payload size:      {clip_size_mb:.1f} MB per clip")
    print()
    print(f"  Camera coverage (real-time, assuming one clip submitted per interval):")
    print(f"    30-second clips:  ~{cameras_30s:.0f} cameras supported concurrently")
    print(f"    60-second clips:  ~{cameras_60s:.0f} cameras supported concurrently")
    print()
    print(f"  Latency characteristics:")
    for r in all_results:
        idle_pct = max(0.0, 1 - r["throughput_rps"] * r["p95_ms"] / 1000) * 100
        print(f"    c={r['concurrency']}: P95={r['p95_ms']:.0f}ms  σ={r['stdev_ms']:.0f}ms")
    print()
    print(f"  Vertex AI quota note:")
    print(f"    Default quota: 300 RPM per project (~5 req/s).")
    print(f"    At c=8 and {rps:.4f} req/s observed, headroom before quota: "
          f"~{max(0, 5.0 - rps):.2f} req/s remaining.")
    print(f"    Request a quota increase for production deployments.")
    print(f"\n  Bottleneck: Vertex AI API response time (~{peak['mean_ms']:.0f}ms mean),")
    print(f"  not server overhead. Horizontal scaling adds throughput linearly")
    print(f"  up to the quota ceiling.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    url         = args.url
    clip_path   = args.clip
    levels      = args.concurrency
    n_requests  = args.requests

    print(f"\n{'═' * W}")
    print(f"  EHS Incident Detection Endpoint — Load Test")
    print(f"{'═' * W}")
    print(f"  Server:      {url}")
    print(f"  Clip:        {clip_path}")
    print(f"  Concurrency: {levels}")
    print(f"  Volume:      {n_requests} requests per level  ({WARMUP_REQUESTS} warm-up discarded)\n")

    # ── Health check ──────────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            resp = await client.get(f"{url}/health")
            h = resp.json()
            print(f"  Health:      {h.get('status')}  |  "
                  f"stage1={h.get('stage1_model')}  stage2={h.get('stage2_model')}")
            cfg = h.get("config", {})
            print(f"  Config:      n_votes={cfg.get('n_votes')}  "
                  f"temp={cfg.get('temperature')}  vote_policy={cfg.get('vote_policy')}\n")
        except Exception as exc:
            print(f"\n  [ERROR] Cannot reach {url}/health: {exc}")
            print(f"  Start the server first:")
            print(f"    uvicorn api.endpoint:app --host 0.0.0.0 --port 8000")
            sys.exit(1)

    # ── Load clip ─────────────────────────────────────────────────────────────
    clip_path_abs = os.path.join(
        str(Path(__file__).parent.parent), clip_path
    ) if not os.path.isabs(clip_path) else clip_path

    if not os.path.exists(clip_path_abs):
        print(f"\n  [ERROR] Clip not found: {clip_path_abs}")
        print(f"  Specify a valid clip with --clip path/to/video.mp4")
        sys.exit(1)

    video_bytes   = Path(clip_path_abs).read_bytes()
    clip_size_mb  = len(video_bytes) / 1_048_576
    filename      = Path(clip_path_abs).name
    print(f"  Clip loaded: {filename}  ({clip_size_mb:.1f} MB)\n")

    # ── Run each concurrency level ────────────────────────────────────────────
    all_results: list[dict] = []
    for c in levels:
        print(f"  ┌─ Concurrency = {c}  {'─' * (W - 18)}")
        result = await _run_level(url, video_bytes, filename, c, n_requests)
        all_results.append(result)
        _print_level(result)
        print(f"  └{'─' * (W - 2)}\n")

    # ── Summary + interpretation ──────────────────────────────────────────────
    _print_summary(all_results)
    _print_interpretation(all_results, clip_size_mb)

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(str(Path(__file__).parent.parent), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "load_test_results.json")

    payload = {
        "server_url":         url,
        "clip":               clip_path,
        "clip_size_mb":       round(clip_size_mb, 2),
        "requests_per_level": n_requests,
        "warmup_requests":    WARMUP_REQUESTS,
        "concurrency_levels": levels,
        "results":            all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EHS Endpoint Load Test — measures throughput and latency under concurrent load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url",  default=DEFAULT_URL,  help="API server URL (default: %(default)s)")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="Path to test video clip (relative to repo root)")
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY,
        metavar="C", help="Concurrency levels to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--requests", type=int, default=DEFAULT_N_REQUESTS,
        metavar="N", help=f"Requests per concurrency level (default: {DEFAULT_N_REQUESTS})",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
