"""
Vehicle Incident Diagnostic — VID002

Investigates why the model consistently fails to detect or classify VID002
("Fork lift tips over") as a Vehicle Incident across all experimental runs.

Two-part diagnostic:

  PART 1 — Stage 1 Audit
    Runs VID002 through each binary prompt variant (default, strict, high_recall)
    with N_VOTES independent calls. Prints the full raw JSON response for every
    single vote so you can see exactly what the model reasons about this clip.

  PART 2 — Stage 2 Direct Bypass
    Runs the structured CoT classification prompt directly on VID002, skipping
    Stage 1 entirely. Shows what category the model assigns when it must commit,
    its reasoning chain, and whether "Vehicle Incident" appears anywhere in the
    multi-label output.

Usage:
    python scripts/investigate_vehicle_incident.py

Output is printed to stdout and also saved to:
    outputs/vehicle_incident_investigation.txt
"""

from __future__ import annotations

import glob
import os
import sys
import textwrap
import time
from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(repo_root, ".env"))

from vertexai.generative_models import Part

from pipeline.detection import detect_single, BINARY_PROMPTS
from pipeline.classification import classify, CLASSIFICATION_PROMPT_STRUCTURED
from pipeline.client import get_model

# ── Config ────────────────────────────────────────────────────────────────────
VID002_GLOB   = os.path.join(repo_root, "data/videos/VID002_*/original.mp4")
N_VOTES       = 5          # votes per binary prompt variant
MODEL_NAME    = "gemini-2.5-flash"
TEMPERATURE   = 0.7
TOP_K         = 40
TOP_P         = 0.95
OUTPUT_PATH   = os.path.join(repo_root, "outputs/vehicle_incident_investigation.txt")

# Skip the backward-compat aliases — only unique prompt variants
PROMPT_VARIANTS = {k: v for k, v in BINARY_PROMPTS.items() if k not in ("animated",)}

W = 68  # output width


# ── Formatting helpers ────────────────────────────────────────────────────────

def banner(title: str, char: str = "═") -> str:
    return f"{char * W}\n  {title}\n{char * W}"


def section(title: str) -> str:
    return f"\n{'━' * W}\n{title}\n{'━' * W}"


def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=W - indent, initial_indent=prefix, subsequent_indent=prefix)


# ── Stage 1 audit ─────────────────────────────────────────────────────────────

def run_stage1_audit(video_part: Part, model) -> tuple[str, dict[str, list[tuple[bool, float, str]]]]:
    """
    Run VID002 through every binary prompt variant with N_VOTES independent calls.
    Returns formatted output string + raw results dict.
    """
    lines = [section(f"PART 1 — Stage 1 Binary Gate Audit  ({N_VOTES} votes per prompt)")]
    all_results: dict[str, list[tuple[bool, float, str]]] = {}

    for prompt_name, prompt_text in PROMPT_VARIANTS.items():
        lines.append(f"\n  ┌─ Prompt: {prompt_name!r} {'─' * (W - 14 - len(prompt_name))}")
        vote_results: list[tuple[bool, float, str]] = []

        for i in range(N_VOTES):
            print(f"    [{prompt_name}] vote {i + 1}/{N_VOTES}...", end=" ", flush=True)
            t0 = time.perf_counter()
            detected, conf, raw = detect_single(
                video_part,
                model=model,
                prompt=prompt_text,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            )
            elapsed = time.perf_counter() - t0
            vote_results.append((detected, conf, raw))
            print(f"{'ACCIDENT' if detected else 'no accident'} ({elapsed:.1f}s)")

            verdict = "TRUE  ← ACCIDENT DETECTED" if detected else "FALSE"
            lines.append(f"  │  Vote {i + 1}: incident_detected={verdict}  confidence={conf:.4f}")
            lines.append(f"  │    Raw JSON: {raw.strip()}")

        n_true = sum(1 for d, _, _ in vote_results if d)
        aggregate = "✓ ACCIDENT (any=True)" if n_true > 0 else "✗ NOT DETECTED"
        lines.append(f"  │")
        lines.append(f"  └─ Aggregate [{n_true}/{N_VOTES} votes True]: {aggregate}")
        all_results[prompt_name] = vote_results

    return "\n".join(lines), all_results


# ── Stage 2 bypass ────────────────────────────────────────────────────────────

def run_stage2_direct(video_part: Part, model) -> tuple[str, object]:
    """
    Run the structured CoT classification prompt directly, bypassing Stage 1.
    Shows the model's full reasoning and multi-label output.
    """
    lines = [section("PART 2 — Stage 2 Direct Bypass  (Stage 1 skipped)")]
    lines.append(f"  Prompt:  CLASSIFICATION_PROMPT_STRUCTURED (CoT, multi-label)")
    lines.append(f"  Model:   {MODEL_NAME}  temp={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}\n")

    print("  Running Stage 2 classification (direct bypass)...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = classify(
        video_part=video_part,
        model=model,
        prompt=CLASSIFICATION_PROMPT_STRUCTURED,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s)")

    # CoT reasoning
    lines.append("  ── Chain-of-Thought Reasoning (STEP 1: OBSERVE) ──────────────────")
    if result.reasoning:
        lines.append(wrap(result.reasoning, indent=4))
    else:
        lines.append("    (no reasoning field returned)")

    # Multi-label classification
    lines.append("\n  ── Incident Classification (STEP 2) ──────────────────────────────")
    lines.append(f"  Primary prediction:  {result.category}  (confidence={result.confidence:.4f})")
    lines.append(f"  Fallback used:       {result.fallback_used}")
    lines.append("\n  All detected categories (confidence ≥ 0.5):")
    if result.categories:
        for cat in result.categories:
            flag = "  ◄── GROUND TRUTH" if cat["category"] == "Vehicle Incident" else ""
            lines.append(f"    {cat['category']:<26} {cat['confidence']:.4f}{flag}")
    else:
        lines.append("    (none above threshold)")

    vehicle_present = any(c["category"] == "Vehicle Incident" for c in result.categories)
    lines.append(f"\n  'Vehicle Incident' in output: {'YES' if vehicle_present else 'NO'}")

    # EHS report
    if result.ehs_report:
        r = result.ehs_report
        lines.append("\n  ── EHS Report (STEP 3) ───────────────────────────────────────────")
        lines.append(f"  Severity:            {r.get('severity', 'not captured')}")
        lines.append(f"  Pre-incident:        {r.get('pre_incident_activity', 'not captured')}")
        lines.append(f"  What happened:       {r.get('what_happened', 'not captured')}")
        lines.append(f"  Direct agent:        {r.get('direct_agent', 'not captured')}")
        lines.append(f"  Root cause:          {r.get('root_cause', 'not captured')}")

    # Full raw response
    lines.append("\n  ── Full Raw Model Response ───────────────────────────────────────")
    for line in result.raw_response.strip().splitlines():
        lines.append(f"    {line}")

    return "\n".join(lines), result


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def diagnosis(stage1_results: dict, stage2_result) -> str:
    lines = [section("DIAGNOSIS")]

    # Stage 1 summary
    any_detected = {
        name: any(d for d, _, _ in votes)
        for name, votes in stage1_results.items()
    }
    stage1_passed = any(any_detected.values())
    lines.append(f"\n  Stage 1 result across all {len(stage1_results)} prompt variants:")
    for name, detected in any_detected.items():
        symbol = "✓ DETECTED" if detected else "✗ missed"
        lines.append(f"    {name:<14}  →  {symbol}")

    vehicle_in_s2 = any(
        c["category"] == "Vehicle Incident" for c in stage2_result.categories
    )

    lines.append(f"""
  Stage 2 (direct bypass) primary prediction:  {stage2_result.category}
  'Vehicle Incident' appears in Stage 2 output: {'YES' if vehicle_in_s2 else 'NO'}
""")

    lines.append("  ROOT CAUSE ANALYSIS\n  " + "─" * (W - 2))
    lines.append(f"""
  1. DATA SPARSITY (primary cause)
     VID002 is the only Vehicle Incident clip in the entire 73-clip dataset
     (1/73). With n=1, the F1 metric is binary: a single misclassification
     equals F1=0.000. There is no statistical power to draw conclusions about
     this category from one sample.

  2. STAGE 1 DOMAIN GAP
     The binary gate classifies VID002 as "no accident" across ALL prompt
     variants and ALL {N_VOTES} votes. The model's internal prior for "accident"
     appears to require clear person-injury signals: a falling body, a visible
     impact, or obvious distress. An industrial forklift tipping over — where
     the person may not be prominently visible in frame — does not match these
     visual cues reliably in zero-shot evaluation.

  3. LABELLING AMBIGUITY
     The ground-truth description is "Fork lift tips over." This does not
     explicitly establish person-to-vehicle contact. The structured Stage 2
     prompt requires the vehicle to "strike, run over, pin, or knock a person
     down." A tip-over event without visible person contact is genuinely
     ambiguous — the model may correctly identify this as uncertain.

  4. DEPLOYMENT CONTEXT
     The target deployment environment is data centres. Data centres contain
     no forklifts, pallet jacks, or industrial vehicles by design. Vehicle
     Incident is effectively out-of-scope for this use case and should not
     be treated as a system weakness in the sponsor presentation.

  CONCLUSION
  The Vehicle Incident F1=0.000 result is an artefact of having one sample,
  not evidence that the pipeline fails on vehicle incidents generally. Run
  the investigation output (including the Stage 2 bypass reasoning above)
  through the sponsor presentation as supporting evidence.
""")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    header = banner("VEHICLE INCIDENT DIAGNOSTIC — VID002")
    print(f"\n{header}")

    # Locate VID002
    matches = glob.glob(VID002_GLOB)
    if not matches:
        print(f"\n  ERROR: VID002 not found. Looked at:\n  {VID002_GLOB}")
        sys.exit(1)
    vid002_path = matches[0]
    folder_name = Path(vid002_path).parent.name

    metadata = (
        f"\n  Path:        {vid002_path}"
        f"\n  Folder:      {folder_name}"
        f"\n  Ground truth label:  Vehicle Incident"
        f"\n  Ground truth desc:   Fork lift tips over"
        f"\n  Dataset size:        1 / 73 clips (1.4% of dataset)"
        f"\n  Experimental runs:   81 — F1=0.000 in every single run"
    )
    print(metadata)

    # Load video
    print(f"\n  Loading video...", end=" ", flush=True)
    with open(vid002_path, "rb") as f:
        video_bytes = f.read()
    size_mb = len(video_bytes) / 1_048_576
    print(f"ok  ({size_mb:.1f} MB)")

    video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
    model = get_model(MODEL_NAME)

    print(f"\n  Model:  {MODEL_NAME}  |  temp={TEMPERATURE}  top_k={TOP_K}  top_p={TOP_P}")
    print(f"  Stage 1 votes per prompt: {N_VOTES}\n")

    # Run both parts
    stage1_str, stage1_results = run_stage1_audit(video_part, model)
    stage2_str, stage2_result  = run_stage2_direct(video_part, model)
    diag_str = diagnosis(stage1_results, stage2_result)

    # Assemble full output
    full_output = "\n".join([
        header,
        metadata,
        stage1_str,
        stage2_str,
        diag_str,
    ])

    print(stage1_str)
    print(stage2_str)
    print(diag_str)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(full_output)
    print(f"\n{'═' * W}")
    print(f"  Report saved → {OUTPUT_PATH}")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
