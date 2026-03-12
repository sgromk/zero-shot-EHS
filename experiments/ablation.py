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

# Compare across all completed runs
compare_results(dirs)
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
    "sleep_between_videos": 1.0,
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
    # Metric to watch: Stage 1 recall (must stay near 1.0), cost per video.
    "F": {
        "stage1_model":            ["gemini-2.0-flash-lite", "gemini-2.5-flash"],
        "binary_prompt_variant":   ["high_recall"],
        "model":                   ["gemini-2.5-flash"],
        "confidence_threshold":    [0.0, 0.3],  # Low threshold → maximize recall
        "n_votes":                 [1, 3],       # n_votes=1 → cheapest Stage 1
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


# ── Result comparison ─────────────────────────────────────────────────────────

def compare_results(output_dirs: list[str]) -> None:
    """Print a comparison table of key metrics across completed ablation runs."""
    header = f"{'Run':<45} {'BinF1':>6} {'MacF1':>6} {'BinRec':>7} {'BinPre':>7} {'Cost$':>7}"
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
        print(
            f"  {run_id:<43} "
            f"{m.get('binary_f1', float('nan')):>6.3f} "
            f"{m.get('macro_f1', float('nan')):>6.3f} "
            f"{m.get('binary_recall', float('nan')):>7.3f} "
            f"{m.get('binary_precision', float('nan')):>7.3f} "
            f"{m.get('total_cost_usd', float('nan')):>7.5f}"
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
