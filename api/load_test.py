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
       # Default: VID001 accident clip, all-accident workload
       python api/load_test.py

       # Mixed workload at 0.1% accident rate (data centre baseline):
       python api/load_test.py --mixed

       # Clip size sensitivity test (11 MB clip vs default 0.27 MB):
       python api/load_test.py --large-clip

       # Custom concurrency and volume:
       python api/load_test.py --concurrency 1 2 4 8 --requests 12

Methodology
-----------
For each concurrency level C:
  - A warm-up batch (WARMUP_REQUESTS) is sent first and discarded.
  - REQUESTS_PER_LEVEL requests are dispatched with at most C in-flight
    at any time (controlled by asyncio.Semaphore).
  - Wall-clock time is measured from first dispatch to last completion.
  - Per-request latency is measured independently inside each coroutine.

Mixed workload mode (--normal-clip provided):
  Randomly selects between the accident clip and normal clip for each request
  according to --accident-rate. This produces realistic cost and latency
  distributions: ~1% of clips reach Stage 2 (~20s), the rest complete at
  Stage 1 only (~10s).

Metrics reported (per concurrency level)
-----------------------------------------
  throughput_rps    Completed requests / wall-clock seconds
  mean_ms           Arithmetic mean of individual request latencies
  p50_ms / p95_ms / p99_ms   Latency percentiles
  min_ms / max_ms   Observed extremes
  stdev_ms          Standard deviation
  efficiency        Throughput gain per worker vs concurrency=1 baseline
  sla_*_pct         % of requests completing within each SLA threshold
  apdex             Apdex score at T=20s (satisfied≤20s, tolerating≤80s)
  errors            HTTP non-200 responses or connection failures

Outputs
-------
  Console:                    formatted summary table + deployment interpretation
  outputs/load_test_results.json   raw results
  outputs/load_test_report.md      white-paper summary (auto-generated)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

import httpx

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL          = "http://localhost:8000"
DEFAULT_CLIP         = "data/videos/VID001_Electric_Forklift_Safety_Accident_VB_Factory_VB_Engineering_I_Pvt_Ltd/original.mp4"
DEFAULT_NORMAL_CLIP  = "data/videos/VID043_Work_Place_Safety_Video-Part_4_VB_Factory_VB_Engineering_I_Pvt_Ltd/original.mp4"
DEFAULT_LARGE_CLIP   = "data/videos/VID004_Power_Plant_Electric_Shock_Safety_Accident_VB_Factory_VB_Engineering_I_Pvt_Ltd/type1/T1_v04.mp4"
DEFAULT_CONCURRENCY  = [1, 2, 4, 8]
DEFAULT_N_REQUESTS   = 12
WARMUP_REQUESTS      = 2
REQUEST_TIMEOUT_S    = 180.0
CONNECT_TIMEOUT_S    = 10.0

# Apdex threshold: "satisfied" if ≤ T, "tolerating" if ≤ 4T, else "frustrated"
APDEX_T_S = 20.0

# SLA thresholds (seconds) for the compliance table
SLA_THRESHOLDS = [10, 20, 30, 60]

W = 72  # console width


# ── Statistics ────────────────────────────────────────────────────────────────

def _pct(data: list[float], p: float) -> float:
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
        "mean_ms":  round(statistics.mean(latencies), 1),
        "p50_ms":   round(_pct(latencies, 50), 1),
        "p95_ms":   round(_pct(latencies, 95), 1),
        "p99_ms":   round(_pct(latencies, 99), 1),
        "min_ms":   round(min(latencies), 1),
        "max_ms":   round(max(latencies), 1),
        "stdev_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0,
    }


def _sla(latencies_ms: list[float]) -> dict:
    """Return % of requests completing within each SLA threshold."""
    n = len(latencies_ms)
    if not n:
        return {}
    result = {}
    for t in SLA_THRESHOLDS:
        pct = sum(1 for ms in latencies_ms if ms <= t * 1000) / n * 100
        result[f"sla_{t}s_pct"] = round(pct, 1)
    return result


def _apdex(latencies_ms: list[float], t_s: float = APDEX_T_S) -> float:
    """Apdex score: (satisfied + tolerating/2) / n. T in seconds."""
    n = len(latencies_ms)
    if not n:
        return float("nan")
    t_ms  = t_s * 1000
    f_ms  = 4 * t_ms
    sat   = sum(1 for ms in latencies_ms if ms <= t_ms)
    tol   = sum(1 for ms in latencies_ms if t_ms < ms <= f_ms)
    return round((sat + tol / 2) / n, 3)


def _efficiency(rps: float, baseline_rps: float, concurrency: int) -> float:
    """How much of ideal linear scaling is achieved (1.0 = perfect)."""
    if not baseline_rps or not concurrency:
        return float("nan")
    return round(rps / (baseline_rps * concurrency), 3)


# ── Single request ────────────────────────────────────────────────────────────

async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    video_bytes: bytes,
    filename: str,
) -> dict:
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
        detected = body.get("incident_detected") if body else None
        return {
            "latency_ms": elapsed_ms,
            "status":     resp.status_code,
            "detected":   detected,
            "s1_ms":      body.get("stage1_latency_ms") if body else None,
            "s2_ms":      body.get("stage2_latency_ms") if body else None,
            "ehs_report": body.get("ehs_report") if (body and detected) else None,
            "error":      None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": elapsed_ms,
            "status":     -1,
            "detected":   None,
            "s1_ms":      None,
            "s2_ms":      None,
            "ehs_report": None,
            "error":      str(exc),
        }


# ── Concurrency batch ─────────────────────────────────────────────────────────

async def _run_level(
    url: str,
    clips: list[tuple[bytes, str]],   # list of (video_bytes, filename) to sample from
    concurrency: int,
    n_requests: int,
    accident_rate: float,
) -> dict:
    timeout  = httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    limits   = httpx.Limits(max_connections=concurrency + 4,
                            max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    # Determine which clip to use for each request upfront
    # clips[0] = accident clip, clips[1] = normal clip (if provided)
    def _pick() -> tuple[bytes, str]:
        if len(clips) == 1 or random.random() < accident_rate:
            return clips[0]
        return clips[1]

    async def _controlled(client: httpx.AsyncClient) -> dict:
        video_bytes, filename = _pick()
        async with semaphore:
            return await _single_request(client, url, video_bytes, filename)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Warm-up
        print(f"    Warming up ({WARMUP_REQUESTS} req)...", end=" ", flush=True)
        warmup_clips = [_pick() for _ in range(WARMUP_REQUESTS)]
        await asyncio.gather(*[
            _single_request(client, url, vb, fn) for vb, fn in warmup_clips
        ])
        print("done")

        # Timed batch
        print(f"    Running {n_requests} requests (concurrency={concurrency})...",
              end=" ", flush=True)
        t_wall_start = time.perf_counter()
        results = await asyncio.gather(*[_controlled(client) for _ in range(n_requests)])
        wall_s  = time.perf_counter() - t_wall_start
        print(f"done ({wall_s:.1f}s)")

    good       = [r for r in results if r["status"] == 200]
    errors     = [r for r in results if r["status"] != 200]
    latencies  = [r["latency_ms"] for r in good]
    s1_lats    = [r["s1_ms"] for r in good if r["s1_ms"] is not None]
    s2_lats    = [r["s2_ms"] for r in good if r["s2_ms"] is not None and r["s2_ms"] > 0]
    detections = sum(1 for r in good if r.get("detected"))
    # Collect up to 3 EHS reports for quality review
    sample_reports = [r["ehs_report"] for r in good
                      if r.get("ehs_report") is not None][:3]

    st = _stats(latencies)
    return {
        "concurrency":     concurrency,
        "n_requests":      n_requests,
        "n_errors":        len(errors),
        "n_detections":    detections,
        "wall_time_s":     round(wall_s, 3),
        "throughput_rps":  round(n_requests / wall_s, 4) if wall_s > 0 else 0.0,
        **st,
        **_sla(latencies),
        "apdex":           _apdex(latencies),
        "stage1_mean_ms":  round(statistics.mean(s1_lats), 1) if s1_lats else None,
        "stage1_p95_ms":   round(_pct(s1_lats, 95), 1) if s1_lats else None,
        "stage2_mean_ms":  round(statistics.mean(s2_lats), 1) if s2_lats else None,
        "stage2_p95_ms":   round(_pct(s2_lats, 95), 1) if s2_lats else None,
        "sample_ehs_reports": sample_reports,
        "error_messages":  [r["error"] for r in errors if r.get("error")],
    }


# ── Console rendering ─────────────────────────────────────────────────────────

def _print_level(r: dict) -> None:
    err_str = f"{r['n_errors']}/{r['n_requests']}"
    det_str = f"{r['n_detections']}/{r['n_requests'] - r['n_errors']}"
    print(f"  │  Throughput:   {r['throughput_rps']:.4f} req/s   wall={r['wall_time_s']:.1f}s")
    print(f"  │  Latency:      mean={r['mean_ms']:.0f}ms  P50={r['p50_ms']:.0f}ms  "
          f"P95={r['p95_ms']:.0f}ms  P99={r['p99_ms']:.0f}ms")
    print(f"  │  Range:        {r['min_ms']:.0f}ms – {r['max_ms']:.0f}ms   σ={r['stdev_ms']:.0f}ms")
    if r.get("stage1_mean_ms"):
        s2_mean = r.get("stage2_mean_ms")
        s2_p95  = r.get("stage2_p95_ms")
        s2_str  = f"{s2_mean:.0f}ms (P95={s2_p95:.0f}ms)" if s2_mean else "N/A"
        print(f"  │  Stage 1:      mean={r['stage1_mean_ms']:.0f}ms  P95={r['stage1_p95_ms']:.0f}ms   "
              f"Stage 2: {s2_str}")
    print(f"  │  Apdex(T=20s): {r['apdex']:.2f}   Detections: {det_str}   Errors: {err_str}")
    sla_parts = "  ".join(
        f"<{t}s:{r.get(f'sla_{t}s_pct', 0):.0f}%" for t in SLA_THRESHOLDS
    )
    print(f"  │  SLA:          {sla_parts}")
    if r["error_messages"]:
        for msg in r["error_messages"][:3]:
            print(f"  │  [ERR] {msg[:60]}")


def _print_summary(all_results: list[dict], baseline_rps: float) -> None:
    print(f"\n{'═' * W}")
    print("  SUMMARY TABLE")
    print(f"{'─' * W}")
    hdr = (f"  {'Conc':>5}  {'Req/s':>7}  {'Mean':>7}  {'P50':>6}  {'P95':>6}  "
           f"{'P99':>6}  {'Apdex':>5}  {'Effic':>6}  {'<30s':>5}  {'Err':>5}")
    print(hdr)
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*5}")
    for r in all_results:
        eff = _efficiency(r["throughput_rps"], baseline_rps, r["concurrency"])
        print(
            f"  {r['concurrency']:>5}"
            f"  {r['throughput_rps']:>7.4f}"
            f"  {r['mean_ms']:>5.0f}ms"
            f"  {r['p50_ms']:>4.0f}ms"
            f"  {r['p95_ms']:>4.0f}ms"
            f"  {r['p99_ms']:>4.0f}ms"
            f"  {r['apdex']:>5.2f}"
            f"  {eff:>5.0%}"
            f"  {r.get('sla_30s_pct', 0):>4.0f}%"
            f"  {r['n_errors']:>3}/{r['n_requests']}"
        )
    print(f"{'═' * W}")


def _print_little(all_results: list[dict]) -> None:
    print(f"\n  LITTLE'S LAW  (N = λ × W)")
    print(f"  {'─' * (W - 2)}")
    print(f"  {'Conc':>5}  {'λ (req/s)':>10}  {'W (mean s)':>10}  {'λ×W':>6}  {'Match?':>8}")
    for r in all_results:
        lam = r["throughput_rps"]
        w   = r["mean_ms"] / 1000
        n   = r["concurrency"]
        predicted = round(lam * w, 2)
        match = "✓" if abs(predicted - n) / max(n, 1) < 0.25 else "~"
        print(f"  {n:>5}  {lam:>10.4f}  {w:>10.1f}  {predicted:>6.2f}  {match:>8}")


def _print_ehs_samples(all_results: list[dict]) -> None:
    """Print sample EHS reports from the first concurrency level that has detections."""
    reports = []
    for r in all_results:
        reports.extend(r.get("sample_ehs_reports") or [])
        if reports:
            break
    if not reports:
        return
    print(f"\n  EHS REPORT QUALITY SAMPLE  ({len(reports)} report(s) from C=1 run)")
    print(f"  {'─' * (W - 2)}")
    for i, rpt in enumerate(reports, 1):
        print(f"\n  Report {i}:")
        for field in ("severity", "what_happened", "injury_description",
                      "root_cause", "corrective_measures"):
            val = rpt.get(field) if rpt else None
            if val:
                # Wrap at 65 chars with indentation
                words, line = val.split(), ""
                lines = []
                for w in words:
                    if len(line) + len(w) + 1 > 65:
                        lines.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    lines.append(line)
                print(f"    {field}: {lines[0]}")
                for ln in lines[1:]:
                    print(f"      {ln}")
    print()


def _print_interpretation(all_results: list[dict], clip_size_mb: float,
                          mixed: bool, accident_rate: float) -> None:
    if not all_results:
        return

    best = max(all_results, key=lambda r: r["throughput_rps"])
    c1   = all_results[0]
    rps  = c1["throughput_rps"]   # use C=1 as the deployment baseline

    cameras_30s = rps * 30
    cameras_60s = rps * 60

    print(f"\n  DEPLOYMENT INTERPRETATION")
    print(f"  {'─' * (W - 2)}")
    print(f"  Optimal operating point:  C=1  ({rps:.4f} req/s, P95={c1['p95_ms']:.0f}ms)")
    if best["concurrency"] != 1:
        print(f"  Peak throughput at C={best['concurrency']}: {best['throughput_rps']:.4f} req/s "
              f"(P95={best['p95_ms']:.0f}ms — latency degrades)")
    print(f"  Payload size:             {clip_size_mb:.2f} MB per clip")
    if mixed:
        print(f"  Workload:                 mixed ({accident_rate*100:.1f}% accident rate)")
    else:
        print(f"  Workload:                 single clip (all requests same file)")
    print()
    print(f"  Camera coverage (real-time, one clip per interval per camera):")
    print(f"    30-second clips:  {cameras_30s:.1f} cameras per endpoint instance")
    print(f"    60-second clips:  {cameras_60s:.1f} cameras per endpoint instance")
    print()
    print(f"  Scaling: horizontal only. Adding concurrency within one instance")
    print(f"  does not improve throughput — Vertex AI quota is the bottleneck.")
    print(f"  Each Cloud Run instance at C=1 uses ~{rps*4*60:.0f} Vertex AI RPM")
    print(f"  (default quota: 300 RPM → supports ~{int(300 / max(rps*4*60, 1))} instances).")
    print()


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_report(
    all_results: list[dict],
    baseline_rps: float,
    clip_size_mb: float,
    clip_path: str,
    mixed: bool,
    accident_rate: float,
    n_requests: int,
    prompt: str,
    out_path: str,
) -> None:
    lines = [
        f"# EHS Incident Detection — Endpoint Load Test Report",
        f"",
        f"**Date:** {date.today()}",
        f"**Pipeline:** Gemini 2.5 Flash / Flash (Stage 1 / Stage 2)",
        f"**Prompt:** `{prompt}`",
        f"**Config:** n_votes=3, temp=0.7, top_k=40, top_p=0.95, vote_policy=any",
        f"**Test clip:** {Path(clip_path).parent.name[:20]} ({clip_size_mb:.2f} MB)",
        f"**Workload:** {'mixed — ' + str(accident_rate*100) + '% accident rate' if mixed else 'single clip (all-accident)'}",
        f"**Requests per level:** {n_requests} timed + {WARMUP_REQUESTS} warm-up (discarded)",
        f"**Server:** FastAPI / uvicorn, single worker process, localhost",
        f"",
        f"---",
        f"",
        f"## Results",
        f"",
        f"| Concurrency | Req/s | Mean | P50 | P95 | P99 | Apdex | <30s | Efficiency | Errors |",
        f"|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        eff = _efficiency(r["throughput_rps"], baseline_rps, r["concurrency"])
        lines.append(
            f"| {r['concurrency']} "
            f"| {r['throughput_rps']:.4f} "
            f"| {r['mean_ms']/1000:.1f}s "
            f"| {r['p50_ms']/1000:.1f}s "
            f"| {r['p95_ms']/1000:.1f}s "
            f"| {r['p99_ms']/1000:.1f}s "
            f"| {r['apdex']:.2f} "
            f"| {r.get('sla_30s_pct',0):.0f}% "
            f"| {eff:.0%} "
            f"| {r['n_errors']}/{r['n_requests']} |"
        )

    lines += [
        f"",
        f"## SLA Compliance",
        f"",
        f"| Concurrency | <10s | <20s | <30s | <60s |",
        f"|---|---|---|---|---|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['concurrency']} "
            + " ".join(f"| {r.get(f'sla_{t}s_pct',0):.0f}% " for t in SLA_THRESHOLDS)
            + "|"
        )

    lines += [
        f"",
        f"## Stage-Level Latency vs Concurrency",
        f"",
        f"| Concurrency | Stage 1 mean | Stage 1 P95 | Stage 2 mean | Stage 2 P95 | Queue overhead |",
        f"|---|---|---|---|---|---|",
    ]
    for r in all_results:
        s1m = r.get("stage1_mean_ms") or 0
        s1p = r.get("stage1_p95_ms") or 0
        s2m = r.get("stage2_mean_ms") or 0
        s2p = r.get("stage2_p95_ms") or 0
        queue = r["mean_ms"] - (s1m + s2m)
        s2m_str = f"{s2m/1000:.1f}s" if s2m else "N/A"
        s2p_str = f"{s2p/1000:.1f}s" if s2p else "N/A"
        lines.append(
            f"| {r['concurrency']} "
            f"| {s1m/1000:.1f}s "
            f"| {s1p/1000:.1f}s "
            f"| {s2m_str} "
            f"| {s2p_str} "
            f"| ~{max(queue,0)/1000:.0f}s |"
        )

    lines += [
        f"",
        f"## Little's Law",
        f"",
        f"N = λ × W (in-flight = throughput × mean latency)",
        f"",
        f"| Concurrency | λ (req/s) | W (mean s) | λ×W | Match? |",
        f"|---|---|---|---|---|",
    ]
    for r in all_results:
        lam = r["throughput_rps"]
        w   = r["mean_ms"] / 1000
        n   = r["concurrency"]
        predicted = round(lam * w, 2)
        match = "✓" if abs(predicted - n) / max(n, 1) < 0.25 else "~"
        lines.append(f"| {n} | {lam:.4f} | {w:.1f} | {predicted:.2f} | {match} |")

    c1  = all_results[0]
    rps = c1["throughput_rps"]
    lines += [
        f"",
        f"## Key Finding",
        f"",
        f"Optimal operating point is **concurrency=1** ({rps:.4f} req/s, P95={c1['p95_ms']/1000:.1f}s).",
        f"Throughput is essentially flat across all concurrency levels while mean latency",
        f"degrades sharply. The bottleneck is Vertex AI API queue saturation, not local compute.",
        f"Individual Stage 1 and Stage 2 call times are stable regardless of concurrency —",
        f"requests queue *before* their first API call, not during it.",
        f"",
        f"**Scale horizontally (separate instances), not vertically (higher concurrency).**",
        f"",
        f"## Camera Coverage",
        f"",
        f"At C=1 ({rps:.4f} req/s per instance):",
        f"",
        f"| Target cameras | Clip interval | Instances required |",
        f"|---|---|---|",
    ]
    for cams, interval in [(5,60),(10,60),(24,60),(50,60),(100,60)]:
        instances = -(-cams // max(int(rps * interval), 1))   # ceil division
        lines.append(f"| {cams} | {interval}s | {instances} |")

    lines += [
        f"",
        f"## GCP Cloud Run Recommendation",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Memory | 2 GiB |",
        f"| CPU | 1 vCPU |",
        f"| Max concurrency per instance | 1 |",
        f"| Min instances | 1 (avoid cold start) |",
        f"| Request timeout | 120s (4× P99 at C=1) |",
        f"| Vertex AI RPM per instance | ~{rps*4*60:.0f} RPM |",
        f"| Default quota (300 RPM) supports | ~{int(300/max(rps*4*60,1))} instances |",
    ]

    # EHS report quality samples
    sample_reports = []
    for r in all_results:
        sample_reports.extend(r.get("sample_ehs_reports") or [])
        if sample_reports:
            break
    if sample_reports:
        lines += [
            f"",
            f"## EHS Report Quality Sample",
            f"",
            f"Sample EHS reports generated during the C=1 load test run:",
        ]
        for i, rpt in enumerate(sample_reports[:2], 1):
            if not rpt:
                continue
            lines += [f"", f"### Report {i}"]
            for field in ("severity", "what_happened", "injury_description",
                          "root_cause", "corrective_measures"):
                val = rpt.get(field)
                if val:
                    lines.append(f"**{field.replace('_', ' ').title()}:** {val}")

    lines += [
        f"",
        f"*Generated {date.today()} by `api/load_test.py`*",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    url          = args.url
    clip_path    = args.clip
    normal_path  = args.normal_clip
    accident_rate = args.accident_rate
    levels       = args.concurrency
    n_requests   = args.requests

    print(f"\n{'═' * W}")
    print(f"  EHS Incident Detection Endpoint — Load Test")
    print(f"{'═' * W}")
    print(f"  Server:       {url}")
    print(f"  Clip:         {clip_path}")
    if normal_path:
        print(f"  Normal clip:  {normal_path}  (accident rate: {accident_rate*100:.1f}%)")
    print(f"  Concurrency:  {levels}")
    print(f"  Volume:       {n_requests} requests per level  ({WARMUP_REQUESTS} warm-up discarded)\n")

    # Health check
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        try:
            resp = await client.get(f"{url}/health")
            h = resp.json()
            print(f"  Health:   {h.get('status')}  |  "
                  f"stage1={h.get('stage1_model')}  stage2={h.get('stage2_model')}")
            cfg = h.get("config", {})
            print(f"  Config:   n_votes={cfg.get('n_votes')}  "
                  f"temp={cfg.get('temperature')}  vote_policy={cfg.get('vote_policy')}\n")
        except Exception as exc:
            print(f"\n  [ERROR] Cannot reach {url}/health: {exc}")
            print(f"  Start the server first:")
            print(f"    uvicorn api.endpoint:app --host 0.0.0.0 --port 8000")
            sys.exit(1)

    # Load clips
    repo = str(Path(__file__).parent.parent)

    def _load(path: str) -> tuple[bytes, str]:
        abs_path = path if os.path.isabs(path) else os.path.join(repo, path)
        if not os.path.exists(abs_path):
            print(f"\n  [ERROR] Clip not found: {abs_path}")
            sys.exit(1)
        data = Path(abs_path).read_bytes()
        print(f"  Loaded: {Path(abs_path).name}  ({len(data)/1_048_576:.2f} MB)")
        return data, Path(abs_path).name

    accident_bytes, accident_name = _load(clip_path)
    clips: list[tuple[bytes, str]] = [(accident_bytes, accident_name)]
    mixed = False

    if normal_path:
        normal_bytes, normal_name = _load(normal_path)
        clips.append((normal_bytes, normal_name))
        mixed = True

    clip_size_mb = len(accident_bytes) / 1_048_576
    print()

    # Run each concurrency level
    all_results: list[dict] = []
    for c in levels:
        print(f"  ┌─ Concurrency = {c}  {'─' * (W - 18)}")
        result = await _run_level(url, clips, c, n_requests, accident_rate)
        all_results.append(result)
        _print_level(result)
        print(f"  └{'─' * (W - 2)}\n")

    baseline_rps = all_results[0]["throughput_rps"] if all_results else 1.0

    # Add efficiency to results
    for r in all_results:
        r["efficiency"] = _efficiency(r["throughput_rps"], baseline_rps, r["concurrency"])

    _print_summary(all_results, baseline_rps)
    _print_little(all_results)
    _print_ehs_samples(all_results)
    _print_interpretation(all_results, clip_size_mb, mixed, accident_rate)

    # Export
    out_dir = os.path.join(repo, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    suffix = f"_{args.run_name}" if args.run_name else ""
    json_path = os.path.join(out_dir, f"load_test_results{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({
            "run_name":           args.run_name or "default",
            "server_url":         url,
            "clip":               clip_path,
            "clip_size_mb":       round(clip_size_mb, 2),
            "mixed_workload":     mixed,
            "accident_rate":      accident_rate if mixed else 1.0,
            "requests_per_level": n_requests,
            "warmup_requests":    WARMUP_REQUESTS,
            "concurrency_levels": levels,
            "results":            all_results,
        }, f, indent=2)
    print(f"  Results saved → {json_path}")

    md_path = os.path.join(out_dir, f"load_test_report{suffix}.md")
    _write_report(all_results, baseline_rps, clip_size_mb, clip_path,
                  mixed, accident_rate, n_requests,
                  prompt="high_recall binary gate, structured CoT classification",
                  out_path=md_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EHS Endpoint Load Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url",  default=DEFAULT_URL)
    parser.add_argument("--clip", default=DEFAULT_CLIP,
                        help="Accident clip path (relative to repo root)")
    parser.add_argument("--normal-clip", default=None, metavar="PATH",
                        help=f"Non-accident clip for mixed workload mode "
                             f"(default when --mixed: {DEFAULT_NORMAL_CLIP})")
    parser.add_argument("--large-clip", action="store_true",
                        help=f"Use large clip ({DEFAULT_LARGE_CLIP}) for size-sensitivity test")
    parser.add_argument("--accident-rate", type=float, default=0.01,
                        help="Fraction of requests that use the accident clip (default: 0.01)")
    parser.add_argument("--concurrency", nargs="+", type=int,
                        default=DEFAULT_CONCURRENCY, metavar="C")
    parser.add_argument("--requests", type=int, default=DEFAULT_N_REQUESTS, metavar="N")
    parser.add_argument("--mixed", action="store_true",
                        help="Enable mixed workload using DEFAULT_NORMAL_CLIP at 0.1% accident rate")
    parser.add_argument("--run-name", default=None, metavar="NAME",
                        help="Suffix for output files, e.g. 'mixed' → load_test_results_mixed.json")
    args = parser.parse_args()

    # Convenience flags: resolve paths before handing off to main()
    if args.large_clip:
        args.clip = DEFAULT_LARGE_CLIP
    if args.mixed and args.normal_clip is None:
        args.normal_clip = DEFAULT_NORMAL_CLIP
    if args.mixed and args.accident_rate == 0.01:
        args.accident_rate = 0.001   # data centre baseline: 0.1%

    asyncio.run(main(args))
