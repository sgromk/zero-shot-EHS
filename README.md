# EHS Video Analysis — Capstone Pipeline

Zero-shot and few-shot video classification pipeline for workplace safety incident detection using Google Vertex AI (Gemini 2.5 Flash).

---

## Project Structure

```
Capstone/
├── config/               # Categories, severity, global settings
├── pipeline/             # Core inference modules
│   ├── client.py         # Vertex AI + GCS init (lazy GCS client)
│   ├── ingestion.py      # Local video sourcing + spreadsheet loader
│   ├── detection.py      # Stage 1: binary accident detection
│   ├── classification.py # Stage 2: incident type classification
│   ├── postprocessing.py # JSON parsing, normalization, retry
│   ├── frame_fallback.py # ffmpeg frame extraction fallback
│   ├── structured_output.py # Decomposed component classification
│   └── near_miss.py      # Three-class: accident / near-miss / safe
├── experiments/          # Orchestration, sampling, ablation
│   ├── runner.py         # Main batch runner (reads YAML config)
│   ├── sampling.py       # Top-k/p probabilistic sweeps
│   ├── multi_agent.py    # Three-agent judge/jury pipeline
│   ├── ablation.py       # 6-group ablation + phased sweep orchestration
│   └── configs/          # YAML configs: attempt1–4 + ablation_grid
├── evaluation/           # Metrics, confusion analysis, plots
│   ├── metrics.py        # Binary + multiclass metrics, GT loader
│   ├── visualize.py      # generate_report() → single PDF output
│   ├── confusion.py      # Failure diagnostics (FP/FN analysis)
│   └── ehs_report.py     # OSHA category mapping, EHS report builder
├── run_logging/          # Per-video JSONL logger + cost estimator
├── notebooks/            # Reference + analysis notebooks
├── data/
│   ├── dataset_mapping.xlsx  # Ground truth (gitignored)
│   └── videos/               # Local dataset (gitignored)
│       └── VID{n}_<label>/
│           ├── original.mp4
│           ├── meta.json
│           ├── type1/    (augmented .mp4 files)
│           └── type2/    (augmented .mp4 files)
└── outputs/              # Experiment results (gitignored)
    └── <timestamp>_<run_id>/
        ├── predictions.jsonl
        ├── metrics.json      # binary_f1, macro_f1, per_class_f1, cost, latency
        ├── failed_videos.csv
        └── report.pdf
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
sudo apt install ffmpeg   # or: brew install ffmpeg on macOS
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — fill in your GCP credentials and paths
```

Required `.env` variables:

| Variable | Description |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Absolute path to service account JSON key |
| `GCP_PROJECT_ID` | GCP project ID (`ehs-incident-detection-blissey`) |
| `GCS_BUCKET_NAME` | GCS bucket name (leave blank if using local mode) |
| `LOCAL_GROUND_TRUTH` | Path to `data/dataset_mapping.xlsx` |

### 3. Dataset structure

Videos live under `data/videos/`, one folder per clip, 0-indexed and zero-padded:

```
data/videos/
├── VID000_<title>/
│   ├── original.mp4
│   ├── meta.json
│   ├── type1/    (augmented clips)
│   └── type2/    (augmented clips)
├── VID001_<title>/
...
└── VID072_<title>/
```

Ground truth: `data/dataset_mapping.xlsx` — 73 rows, columns:
`video_url`, `start_s`, `end_s`, `duration_s`, `incident_present`,
`near_miss_present`, `accident_present`, `incident_severity`, `incident_type`, `description`.

---

## Running Experiments

### Single experiment
```bash
# Run a specific experiment config
python -m experiments.runner experiments/configs/attempt2.yaml

# Auto-detect the next attempt (scans outputs/ for highest attempt{n})
python -m experiments.runner --next
```

After each run, `metrics.json` is automatically populated with binary_f1, macro_f1, per_class_f1, cost, and latency (requires `spreadsheet_source` to be set in the config).

### Ablation sweep — phased approach

Phase 1 sweeps all 64 configs over the 73 originals. Phase 2 runs the best-performing configs on augmented videos.

```python
# Full two-phase sweep (recommended)
from experiments.ablation import run_full_phased_sweep
run_full_phased_sweep(n_top=5)

# Smoke test — 1 config per group, skips nothing
run_full_phased_sweep(n_top=3, max_configs_per_group=1)

# Run phases independently
from experiments.ablation import run_phase1_sweep, select_best_configs, run_phase2_augmented

phase1 = run_phase1_sweep()                            # C → D → F → B → A → E
all_dirs = [d for dirs in phase1.values() for d in dirs]
best = select_best_configs(all_dirs, n=5)              # top-N + Pareto front
run_phase2_augmented(best)                             # type1/ + type2/ videos

# Run a single group
from experiments.ablation import run_study_group
run_study_group('C')
```

### Compare results across runs
```python
from experiments.ablation import compare_results
compare_results(["outputs/phase1_C_000", "outputs/phase1_C_001", ...])
# Prints: BinF1 | MacF1 | BinRec | BinPre | Lat(s) | Cost($) | weak-category F1s
```

### Generate a PDF report
```python
from evaluation.visualize import generate_report
generate_report(
    'outputs/<timestamp>_attempt2/predictions.jsonl',
    'data/dataset_mapping.xlsx',
    'outputs/<timestamp>_attempt2/report.pdf',
    run_id='attempt2',
)
```

---

## Experiment Configs

| Config | Description | Status |
|---|---|---|
| `attempt1.yaml` | Baseline — default prompt, originals only, no threshold | ✅ Done — binary F1=0.889, macro F1=0.681 |
| `attempt2.yaml` | Strict prompt, confidence threshold 0.6 | Planned |
| `attempt3.yaml` | Default prompt + frame fallback (10 frames @ 2 fps) | Planned |
| `attempt4_high_recall_gate.yaml` | Flash-Lite Stage 1 gate + high-recall prompt → Flash Stage 2 | Planned |
| `ablation_grid.yaml` | Full parameter space reference + group definitions | Reference |
| `experiment_template.yaml` | Fully documented template with all options | Reference |

---

## Pipeline Design

### Two-stage inference

```
Every video
    │
    ▼
Stage 1 (binary gate)
    model:  stage1_model   (default: gemini-2.5-flash, or gemini-2.0-flash-lite for cost opt.)
    prompt: binary_prompt_variant  (default | strict | high_recall)
    calls:  n_votes  (1–5 independent calls, combined by vote_policy)
    │
    ├─ "No Accident" ──→ STOP. Log prediction. No Stage 2 call.
    │                    (single-pass: critical for 99%-negative real-world footage)
    │
    └─ "Accident" ─────→ Stage 2 (classifier)
                            model:  model  (gemini-2.5-flash)
                            prompt: classification_prompt_variant
                            output: incident category + timing + RCA
```

### Cost optimization (single-pass for non-accidents)

In real surveillance deployment, ~99% of footage has no accident. The pipeline is designed so non-accident clips receive exactly **one cheap Stage 1 call** and stop. Accident clips receive Stage 1 + Stage 2.

Key config for cost optimization:
- `stage1_model: gemini-2.0-flash-lite` — cheapest model for gating
- `binary_prompt_variant: high_recall` — aggressive: flags any ambiguity, never misses a real accident
- `n_votes: 1` — single Stage 1 call per video (eliminates ensemble overhead on non-accidents)
- `use_frame_fallback: false` — frame fallback re-checks Stage 1 negatives; disable for cost opt.
- `confidence_threshold: 0.0` — pass all Stage 1 positives to Stage 2 (no additional gating)

Stage 1 recall must stay ≥ 0.97. Stage 1 precision is irrelevant — Stage 2 corrects false positives.

---

## Ablation Studies

Six study groups, 64 total configurations. Run in the order below (cheapest/most-impactful first):

| Order | Group | Axes | Configs | Key metric |
|---|---|---|---|---|
| 1 | **C** | binary_prompt × classification_prompt | 6 | binary_f1, macro_f1 |
| 2 | **D** | confidence_threshold × binary_prompt | 10 | binary_recall, cost |
| 3 | **F** | stage1_model × threshold × n_votes (high_recall, no fallback) | 8 | Stage 1 recall, cost |
| 4 | **B** | n_votes × vote_policy (temp=0.3 fixed) | 9 | binary_recall, cost |
| 5 | **A** | temperature × top_p × top_k | 27 | binary_f1, latency |
| 6 | **E** | stage1_model × model | 4 | recall (Stage 1), macro_f1 (Stage 2) |

`SWEEP_ORDER = ["C", "D", "F", "B", "A", "E"]` is enforced by `run_phase1_sweep()`.

### Config isolation rules
- **A**: keep n_votes=3, prompt=default, threshold=0.0
- **B**: keep temperature=0.3 (temp=0 makes all votes identical)
- **C**: keep temperature=0.1, n_votes=3, threshold=0.0
- **D**: use best binary_prompt from Group C
- **E**: fix everything else, vary only model tiers
- **F**: binary_prompt fixed to `high_recall`; `use_frame_fallback` fixed to `False`

### Phase 2 — Augmented videos

After Phase 1 identifies the top configs, `run_phase2_augmented()` reruns them on `type1/` and `type2/` augmented variants. Ground truth is matched by stripping the augmented suffix from the video ID (e.g., `VID042_type1_aug_bright` → `VID042`).

Use `augmented_only: true` in a YAML config to run augmented videos only without re-running originals.

---

## Output Format

Each run writes to `outputs/<timestamp>_<run_id>/`:

| File | Description |
|---|---|
| `predictions.jsonl` | Per-video predictions: stage1/stage2 latency, estimated cost, votes, category |
| `failed_videos.csv` | Videos that errored out |
| `metrics.json` | Full metrics: binary_f1, macro_f1, per_class_f1, binary_recall, binary_precision, cost, latency |
| `report.pdf` | Metrics table + binary CM + multiclass CM |

Cost is estimated per-video as: `n_votes × stage1_cost + (stage2_cost if accident_detected else 0)`. Stage 1 and Stage 2 may use different models with different per-token rates.

---

## Research Directions

| # | Direction | Module | Status |
|---|---|---|---|
| 1 | Prompt engineering (default / strict / high_recall) | `pipeline/detection.py`, `pipeline/classification.py` | ✅ impl |
| 2 | Structured decomposed output | `pipeline/structured_output.py` | ✅ impl |
| 3 | Voting ensemble (`n_votes` param) | `pipeline/detection.py` | ✅ impl |
| 4 | Top-k/p sampling distribution | `experiments/sampling.py` | ✅ impl |
| 5 | Multi-agent judge/jury | `experiments/multi_agent.py` | ✅ impl |
| 6 | Near-miss detection | `pipeline/near_miss.py` | ✅ impl |
| 7 | Failure mode diagnostics | `evaluation/confusion.py` | ✅ impl |
| 8 | 6-group ablation framework | `experiments/ablation.py` | ✅ impl |
| 9 | Augmentation robustness (originals + augmented) | `originals_only` / `augmented_only` config flags | ✅ impl |
| 10 | EHS report auto-population | `evaluation/ehs_report.py` | ✅ impl |
| 11 | High-recall cheap Stage 1 gate | `stage1_model` + `high_recall` prompt | ✅ impl |
| 12 | Auto-increment attempt number | `runner.py --next` / `next_attempt_number()` | ✅ impl |
| 13 | Phased sweep (originals → augmented) | `ablation.run_full_phased_sweep()` | ✅ impl |
| 14 | Pareto-optimal config selection (F1 vs cost) | `ablation.get_pareto_front()`, `select_best_configs()` | ✅ impl |

---

## Dataset

73 real YouTube workplace safety clips across 11 incident categories:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire, Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident.

Videos are identified by row order in `dataset_mapping.xlsx` → `VID000`–`VID072` (0-indexed, 3-digit zero-padded).
