"""
Direction 8 — Ablation study framework.

Systematically compare pipeline configurations across isolated axes.

Study groups (run independently — vary ONE axis at a time):

  Group A — Sampling parameters (temperature × top_p × top_k)
  Group B — Ensemble voting (n_votes × vote_policy)
  Group C — Prompt engineering (binary_prompt_variant × classification_prompt_variant)
  Group D — Stage 2 confidence threshold gating
  Group E — Model selection (stage1_model × model)
  Group F — Cost optimization (high-recall Stage 1 gate with cheap model)

Usage
-----
from experiments.ablation import run_study_group, compare_results

# Run a specific group
dirs = run_study_group("C")

# Run the full two-phase sweep (Phase 1 originals → select best → Phase 2 augmented)
from experiments.ablation import run_full_phased_sweep
run_full_phased_sweep(n_top=5)

# One-command CLI:
#   python -m experiments.ablation              # full sweep, 8 workers
#   python -m experiments.ablation --smoke      # 1 config/group smoke test
#   python -m experiments.ablation --group C    # single group
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any

from experiments.runner import ExperimentRunner

# ── Base config shared by all studies ─────────────────────────────────────────
# Override individual keys within each group definition.

BASE_CONFIG: dict[str, Any] = {
    "model": "gemini-2.5-flash",
    "stage1_model": "gemini-2.5-flash",   # same as model unless overridden
    "binary_prompt_variant": "default",
    "classification_prompt_variant": "default",
    "temperature": 0.1,
    "top_k": None,
    "top_p": None,
    "n_votes": 3,
    "vote_policy": "any",
    "confidence_threshold": 0.0,
    "use_frame_fallback": False,
    "frame_fallback_fps": 2.0,
    "frame_fallback_max_frames": 10,
    # Parallelism — set max_workers > 1 to process multiple videos concurrently.
    # n_votes calls within a single video are always parallelized automatically.
    # sleep_between_videos is ignored when max_workers > 1.
    "max_workers": 8,
    "sleep_between_videos": 0,
    "spreadsheet_source": "data/dataset_mapping.xlsx",
    "local_videos_dir": "data/videos",
    "originals_only": True,
    "outputs_dir": "outputs",
}


# ── Study group definitions ───────────────────────────────────────────────────
#
# Each group is a dict of axis → list[values].
# The grid is the Cartesian product of all axes in that group.
# All other parameters inherit from BASE_CONFIG.

STUDY_GROUPS: dict[str, dict[str, list[Any]]] = {

    # ── Group A: Sampling ─────────────────────────────────────────────────────
    # Isolate the effect of temperature, top_p, and top_k on output quality.
    # Note: top_p and top_k only have meaningful effect when temperature > 0.
    # For n_votes > 1 with temperature=0 all votes are identical (deterministic).
    "A": {
        "temperature": [0.0, 0.3, 0.7],
        "top_p":       [None, 0.85, 0.95],
        "top_k":       [None, 20, 40],
    },

    # ── Group B: Ensemble voting ──────────────────────────────────────────────
    # Fix temperature=0.3 (variance needed for ensemble to matter).
    # Sweep n_votes and vote_policy.
    # "any"      → incident if ANY vote is True  → higher recall
    # "majority" → incident if >50% are True     → balanced
    # "all"      → incident if ALL votes are True → higher precision
    "B": {
        "temperature": [0.3],
        "n_votes":     [1, 3, 5],
        "vote_policy": ["any", "majority", "all"],
    },

    # ── Group C: Prompt engineering ───────────────────────────────────────────
    # Fix sampling (temperature=0.1, no top_p/k, n_votes=3, vote=any).
    # Vary both stage prompts independently.
    # "default"     → standard workplace safety inspector framing
    # "strict"      → conservative — only flag clear physical harm
    # "high_recall" → aggressive — flag on any ambiguity (bias toward recall)
    "C": {
        "binary_prompt_variant":         ["default", "strict", "high_recall"],
        "classification_prompt_variant": ["default", "structured"],
    },

    # ── Group D: Confidence threshold (Stage 2 gating) ───────────────────────
    # Fix best prompt from Group C. Sweep Stage 2 gate threshold.
    # threshold=0.0 → always run Stage 2 (no gating)
    # threshold=0.5 → skip Stage 2 when Stage 1 confidence < 0.5
    # threshold=0.7 → aggressive gating: skip Stage 2 unless very confident
    # Key tradeoff: cost/latency savings vs. recall on ambiguous positives.
    "D": {
        "confidence_threshold": [0.0, 0.4, 0.5, 0.6, 0.7],
        "binary_prompt_variant": ["default", "high_recall"],
    },

    # ── Group E: Model selection ──────────────────────────────────────────────
    # Compare different model tiers for each stage.
    # Stage 1 (binary gate): should be fast and cheap with high recall.
    # Stage 2 (classifier):  benefits from more capable reasoning.
    "E": {
        "stage1_model": [
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",   # cheapest — is it good enough for gating?
        ],
        "model": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",          # most capable — worth the cost for Stage 2?
        ],
    },

    # ── Group F: Cost optimization — high-recall cheap gate ──────────────────
    # Core hypothesis: ~99% of surveillance frames have no accident.
    # Strategy: use cheapest model + high-recall prompt for Stage 1.
    # Stage 1 should have near-perfect recall (miss nothing) at low cost.
    # Stage 2 only runs on the ~1% of clips Stage 1 flags — so cost is minimal.
    #
    # Single-pass enforcement: frame fallback is explicitly off. Non-accident clips
    # must receive exactly ONE inference call (Stage 1 only).
    #
    # Metric to watch: Stage 1 recall (must stay near 1.0), cost per video.
    "F": {
        "stage1_model":            ["gemini-2.0-flash-lite", "gemini-2.5-flash"],
        "binary_prompt_variant":   ["high_recall"],
        "model":                   ["gemini-2.5-flash"],
        "confidence_threshold":    [0.0, 0.3],  # Low threshold → maximize recall
        "n_votes":                 [1, 3],       # n_votes=1 → cheapest Stage 1
        "use_frame_fallback":      [False],      # Explicit: never re-check non-accidents
    },
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def generate_configs(
    grid: dict[str, list[Any]],
    base_config: dict[str, Any] | None = None,
    max_configs: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate all Cartesian product combinations from a parameter grid.

    Each combination inherits from base_config (or BASE_CONFIG) and overrides
    the keys defined in the grid.
    """
    _base = base_config if base_config is not None else BASE_CONFIG

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    configs = []
    for combo in itertools.product(*value_lists):
        cfg = dict(_base)
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)

    if max_configs is not None:
        configs = configs[:max_configs]

    return configs


def run_study_group(
    group_id: str,
    base_config: dict[str, Any] | None = None,
    max_configs: int | None = None,
    outputs_dir: str = "outputs",
    name_prefix: str | None = None,
) -> list[str]:
    """
    Run all parameter combinations for a named study group.

    Parameters
    ----------
    group_id : str
        One of "A", "B", "C", "D", "E", "F".
    base_config : dict | None
        Override the shared BASE_CONFIG for this group.
    max_configs : int | None
        Cap number of configs (useful for smoke tests).
    outputs_dir : str
        Root directory for outputs.
    name_prefix : str | None
        Prefix for run names. Defaults to f"ablation_{group_id}".

    Returns
    -------
    List of output directories for completed runs.
    """
    if group_id not in STUDY_GROUPS:
        raise ValueError(f"Unknown study group: {group_id!r}. Use one of {list(STUDY_GROUPS)}")

    grid = STUDY_GROUPS[group_id]
    configs = generate_configs(grid, base_config=base_config, max_configs=max_configs)
    prefix = name_prefix or f"ablation_{group_id}"

    print(f"\n[Group {group_id}] {len(configs)} configurations")
    _describe_grid(group_id, grid)

    output_dirs: list[str] = []
    for i, cfg in enumerate(configs):
        run_name = f"{prefix}_{i:03d}"
        cfg["name"] = run_name
        cfg["outputs_dir"] = outputs_dir
        print(f"\n  [{i+1}/{len(configs)}] {run_name}")
        _print_diff(cfg)
        runner = ExperimentRunner(cfg, run_name=run_name)
        runner.run()
        output_dirs.append(runner.output_dir)

    print(f"\n[Group {group_id}] All runs complete.")
    compare_results(output_dirs)
    return output_dirs


def run_ablation(
    grid: dict[str, list[Any]] | None = None,
    base_config: dict[str, Any] | None = None,
    name_prefix: str = "ablation",
    max_configs: int | None = None,
    outputs_dir: str = "outputs",
) -> list[str]:
    """
    Run a custom parameter grid (not tied to a named group).
    Kept for backward compatibility and ad-hoc sweeps.
    """
    _grid = grid or {k: v for g in STUDY_GROUPS.values() for k, v in g.items()}
    configs = generate_configs(_grid, base_config=base_config, max_configs=max_configs)
    print(f"Running {len(configs)} custom ablation experiments ...")

    output_dirs: list[str] = []
    for i, cfg in enumerate(configs):
        run_name = f"{name_prefix}_{i:03d}"
        cfg["name"] = run_name
        cfg["outputs_dir"] = outputs_dir
        print(f"\n[{i+1}/{len(configs)}] {run_name}")
        runner = ExperimentRunner(cfg, run_name=run_name)
        runner.run()
        output_dirs.append(runner.output_dir)

    return output_dirs


# ── Recommended run order ─────────────────────────────────────────────────────
#
# Rationale:
#   C first — prompt quality is most impactful and cheapest to sweep (6 configs)
#   D next  — threshold tuning directly affects cost/recall tradeoff
#   F next  — validates the cost-optimization strategy (key result for the paper)
#   B next  — ensemble only matters if C showed variance; n_votes=5 is expensive
#   A last  — 27 configs; only run if B/C suggest sampling matters
#   E last  — Pro model is expensive; run only to confirm tier selection
#
SWEEP_ORDER: list[str] = ["C", "D", "F", "B", "A", "E"]


# ── Result comparison ─────────────────────────────────────────────────────────

def compare_results(output_dirs: list[str]) -> None:
    """
    Print a comparison table of key metrics across completed ablation runs.

    Columns: BinF1, MacF1, BinRec, BinPre, Lat(s), Cost($)
    Weak per-class F1 (Vehicle, Trip, Struck) printed when available.
    """
    _WEAK = ["Vehicle Incident", "Trip", "Struck by Object"]
    weak_cols = "  ".join(f"{c[:6]:>6}" for c in _WEAK)
    header = (
        f"{'Run':<45} {'BinF1':>6} {'MacF1':>6} {'BinRec':>7} {'BinPre':>7}"
        f" {'Lat(s)':>6} {'Cost$':>8}  {weak_cols}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for out_dir in output_dirs:
        metrics_path = os.path.join(out_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"  {os.path.basename(out_dir):<43}  (no metrics yet)")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        run_id = os.path.basename(out_dir)[-43:]
        per_class = m.get("per_class_f1", {})
        weak_vals = "  ".join(
            f"{per_class.get(c, float('nan')):>6.3f}" for c in _WEAK
        )
        print(
            f"  {run_id:<43} "
            f"{m.get('binary_f1', float('nan')):>6.3f} "
            f"{m.get('macro_f1', float('nan')):>6.3f} "
            f"{m.get('binary_recall', float('nan')):>7.3f} "
            f"{m.get('binary_precision', float('nan')):>7.3f} "
            f"{m.get('mean_latency_s', float('nan')):>6.1f} "
            f"{m.get('total_cost_usd', float('nan')):>8.6f}  "
            f"{weak_vals}"
        )


# ── Internal ──────────────────────────────────────────────────────────────────

def _describe_grid(group_id: str, grid: dict[str, list[Any]]) -> None:
    descriptions = {
        "A": "Sampling: temperature × top_p × top_k",
        "B": "Ensemble: n_votes × vote_policy (temperature=0.3 fixed)",
        "C": "Prompts: binary_prompt_variant × classification_prompt_variant",
        "D": "Threshold: confidence_threshold (Stage 2 gating aggressiveness)",
        "E": "Models: stage1_model × model (Stage 1 vs Stage 2 tier)",
        "F": "Cost optimization: cheap Stage 1 gate with high-recall prompt",
    }
    print(f"  Description: {descriptions.get(group_id, '')}")
    total = 1
    for k, v in grid.items():
        print(f"    {k}: {v}")
        total *= len(v)
    print(f"  Total combinations: {total}")


def _print_diff(cfg: dict[str, Any]) -> None:
    """Print only the keys that differ from BASE_CONFIG."""
    diffs = {k: v for k, v in cfg.items() if BASE_CONFIG.get(k) != v and k not in ("name", "outputs_dir")}
    print(f"    axes: {diffs}")


# ── Phased sweep ──────────────────────────────────────────────────────────────

def run_phase1_sweep(
    outputs_dir: str = "outputs",
    max_configs_per_group: int | None = None,
) -> dict[str, list[str]]:
    """
    Phase 1: Run all 6 study groups over the 73 original videos.

    Groups are executed in SWEEP_ORDER (C → D → F → B → A → E).
    Returns a dict mapping group_id → list of output directories.
    """
    results: dict[str, list[str]] = {}
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Originals sweep (order: {' → '.join(SWEEP_ORDER)})")
    print(f"{'='*60}")
    for group_id in SWEEP_ORDER:
        print(f"\n{'─'*60}")
        dirs = run_study_group(
            group_id,
            max_configs=max_configs_per_group,
            outputs_dir=outputs_dir,
            name_prefix=f"phase1_{group_id}",
        )
        results[group_id] = dirs
    return results


def select_best_configs(
    output_dirs: list[str],
    n: int = 5,
    primary_metric: str = "binary_f1",
    include_pareto: bool = True,
) -> list[dict[str, Any]]:
    """
    Select the top-N configs from a list of completed run directories.

    Loads each run's metrics.json and config_snapshot.yaml (if present),
    then ranks by primary_metric. When include_pareto=True, adds any
    Pareto-optimal configs on (binary_f1, cost) not already in the top-N.

    Returns a list of config dicts ready to pass to ExperimentRunner.
    """
    runs: list[dict[str, Any]] = []
    for out_dir in output_dirs:
        metrics_path = os.path.join(out_dir, "metrics.json")
        config_path = os.path.join(out_dir, "config_snapshot.yaml")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        cfg: dict[str, Any] = {}
        if os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        runs.append({
            "output_dir": out_dir,
            "metrics": m,
            "config": cfg,
            "binary_f1": m.get("binary_f1", 0.0),
            "total_cost_usd": m.get("total_cost_usd", float("inf")),
        })

    if not runs:
        print("[select_best_configs] No completed runs found.")
        return []

    # Sort by primary metric descending
    runs.sort(key=lambda r: r.get(primary_metric, 0.0), reverse=True)
    top_n = runs[:n]

    if include_pareto:
        pareto = get_pareto_front(runs)
        # Add any Pareto configs not already in top_n
        top_dirs = {r["output_dir"] for r in top_n}
        for p in pareto:
            if p["output_dir"] not in top_dirs:
                top_n.append(p)

    print(f"\n[select_best_configs] Selected {len(top_n)} configs for Phase 2:")
    hdr = f"  {'Run':<45} {'BinF1':>6} {'Cost$':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in top_n:
        run_id = os.path.basename(r["output_dir"])[-43:]
        print(f"  {run_id:<45} {r['binary_f1']:>6.3f} {r['total_cost_usd']:>8.6f}")

    return [r["config"] for r in top_n]


def get_pareto_front(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return the Pareto-optimal configs on the (binary_f1 ↑, cost ↓) frontier.

    A config is Pareto-optimal if no other config simultaneously has a higher
    binary_f1 AND a lower total_cost_usd.
    """
    pareto: list[dict[str, Any]] = []
    for candidate in runs:
        f1_c = candidate.get("binary_f1", 0.0)
        cost_c = candidate.get("total_cost_usd", float("inf"))
        dominated = any(
            other.get("binary_f1", 0.0) >= f1_c
            and other.get("total_cost_usd", float("inf")) <= cost_c
            and other is not candidate
            for other in runs
        )
        if not dominated:
            pareto.append(candidate)
    return pareto


def run_phase2_augmented(
    selected_configs: list[dict[str, Any]],
    outputs_dir: str = "outputs",
) -> list[str]:
    """
    Phase 2: Run the selected configs on augmented videos (type1/ and type2/).

    For each config, sets augmented_only=True and originals_only=False so the
    runner picks up only the augmented variants, not the originals again.

    Ground truth is still matched via the VID{n} prefix that video_id_from_path()
    embeds in composite augmented IDs (e.g. VID042_type1_aug → VID042 in GT).

    Returns list of output directories for all Phase 2 runs.
    """
    output_dirs: list[str] = []
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Augmented sweep ({len(selected_configs)} selected configs)")
    print(f"{'='*60}")

    for i, cfg in enumerate(selected_configs):
        run_name = cfg.get("name", f"phase2_{i:03d}")
        # Strip any Phase 1 prefix and add phase2 marker
        run_name = f"phase2_{run_name.lstrip('phase1_')}"

        aug_cfg = dict(cfg)
        aug_cfg["augmented_only"] = True
        aug_cfg["originals_only"] = False  # ensure augmented_only takes precedence
        aug_cfg["name"] = run_name
        aug_cfg["outputs_dir"] = outputs_dir

        print(f"\n  [{i+1}/{len(selected_configs)}] {run_name}")
        runner = ExperimentRunner(aug_cfg, run_name=run_name)
        runner.run()
        output_dirs.append(runner.output_dir)

    print(f"\n[Phase 2] All augmented runs complete.")
    compare_results(output_dirs)
    return output_dirs


def run_full_phased_sweep(
    n_top: int = 5,
    outputs_dir: str = "outputs",
    max_configs_per_group: int | None = None,
) -> dict[str, Any]:
    """
    Full two-phase ablation sweep.

    Phase 1: Run all 6 groups on originals (73 videos).
    Select:  Top-N + Pareto-optimal configs by binary_f1 vs cost.
    Phase 2: Run selected configs on augmented videos (type1/ + type2/).

    Parameters
    ----------
    n_top : int
        Number of top configs to carry into Phase 2.
    outputs_dir : str
        Root directory for all output folders.
    max_configs_per_group : int | None
        Cap per group — useful for smoke tests (e.g. max_configs_per_group=1).

    Returns
    -------
    Dict with keys "phase1", "selected_configs", "phase2" containing
    output directories for each phase.
    """
    # Phase 1
    phase1_dirs_by_group = run_phase1_sweep(
        outputs_dir=outputs_dir,
        max_configs_per_group=max_configs_per_group,
    )
    all_phase1_dirs = [d for dirs in phase1_dirs_by_group.values() for d in dirs]

    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE — Selecting best configs for Phase 2")
    print(f"{'='*60}")
    compare_results(all_phase1_dirs)

    selected = select_best_configs(all_phase1_dirs, n=n_top, include_pareto=True)

    # Phase 2
    phase2_dirs = run_phase2_augmented(selected, outputs_dir=outputs_dir)

    return {
        "phase1": phase1_dirs_by_group,
        "selected_configs": selected,
        "phase2": phase2_dirs,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    smoke = "--smoke" in args
    outputs = "outputs"

    # --group X  → run a single study group
    if "--group" in args:
        idx = args.index("--group")
        group_id = args[idx + 1].upper()
        run_study_group(
            group_id,
            max_configs=1 if smoke else None,
            outputs_dir=outputs,
        )
        sys.exit(0)

    # --compare <dir1> <dir2> ...  → compare result directories
    if "--compare" in args:
        idx = args.index("--compare")
        dirs = args[idx + 1:]
        compare_results(dirs)
        sys.exit(0)

    # Default: full phased sweep
    print("Starting full phased ablation sweep.")
    print("  Phase 1: originals (64 configs, C→D→F→B→A→E)")
    print("  Phase 2: augmented videos (top-5 Pareto configs)")
    if smoke:
        print("  [smoke] max_configs_per_group=1")
    run_full_phased_sweep(
        n_top=5,
        outputs_dir=outputs,
        max_configs_per_group=1 if smoke else None,
    )
