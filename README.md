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
│   ├── ablation.py       # 6-group ablation framework
│   └── configs/          # YAML configs: attempt1–4 + ablation_grid
├── evaluation/           # Metrics, confusion analysis, plots
│   ├── metrics.py        # Binary + multiclass metrics, GT loader
│   └── visualize.py      # generate_report() → single PDF output
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
        ├── metrics.json
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

```bash
# Run a specific experiment config
python -m experiments.runner experiments/configs/attempt2.yaml

# Auto-detect the next attempt (scans outputs/ for highest attempt{n})
python -m experiments.runner --next

# Evaluate results against ground truth
python -m evaluation.metrics outputs/<timestamp>_attempt2/predictions.jsonl

# Generate PDF report (metrics table + confusion matrices)
python -c "
from evaluation.visualize import generate_report
generate_report(
    'outputs/<timestamp>_attempt2/predictions.jsonl',
    'data/dataset_mapping.xlsx',
    'outputs/<timestamp>_attempt2/report.pdf',
    run_id='attempt2',
)"

# Run an ablation study group
python -c "from experiments.ablation import run_study_group; run_study_group('F')"
```

Experiment configs in `experiments/configs/`:

| Config | Description | Status |
|---|---|---|
| `attempt1.yaml` | Baseline — default prompt, originals only, no threshold | ✅ Done — binary F1=0.889, macro F1=0.681 |
| `attempt2.yaml` | Strict prompt, confidence threshold 0.6 | Planned |
| `attempt3.yaml` | Default prompt + frame fallback (10 frames @ 2 fps) | Planned |
| `attempt4_high_recall_gate.yaml` | Flash-Lite Stage 1 gate + high-recall prompt → Flash Stage 2 | Planned |
| `ablation_grid.yaml` | Full parameter space reference + group definitions | Reference |
| `experiment_template.yaml` | Fully documented template with all options | Reference |

---

## Optimization Goals

The pipeline targets three dimensions for improvement:

### Latency
- Reduce prompt length (concise prompts → fewer input tokens → faster TTFT)
- Tune `temperature` and `top_p` to avoid long sampling tails
- Skip Stage 2 (classification) when Stage 1 confidence is below threshold
- Evaluate frame-fallback cost vs. latency tradeoff: full video vs. N sampled frames

### Cost
- **High-recall cheap gate (key strategy)**: Surveillance footage is highly imbalanced — in real deployment, ~99% of clips contain no accident. The pipeline exploits this with a two-model approach:  
  - `stage1_model` (`gemini-2.0-flash-lite` + `high_recall` prompt): runs on every video. Must have near-perfect recall. Missing an accident here is a critical failure. Tolerates false positives since Stage 2 corrects them.
  - `model` (`gemini-2.5-flash`): only runs on videos Stage 1 flags as accidents (~1% in real deployment). This is where accuracy matters.
  - Net effect: Stage 2 cost scales with incident rate, not video volume.
- The `confidence_threshold` parameter gates Stage 2 — only classify videos the model is confident are positive
- Compare per-video cost at `originals_only: true` vs. `false` (originals + augmented is ~11× more expensive)
- `stage1_model: gemini-2.0-flash-lite` is the primary cost lever for the gate

### Ablation Studies
See `experiments/configs/ablation_grid.yaml` for the full parameter space. Six study groups:

| Group | Axes | Combinations | Goal |
|---|---|---|---|
| **A** | temperature × top_p × top_k | 27 | Understand sampling stochasticity |
| **B** | n_votes × vote_policy (temp=0.3 fixed) | 9 | Optimal ensemble strategy |
| **C** | binary_prompt × classification_prompt | 6 | Prompt framing effect |
| **D** | confidence_threshold × binary_prompt | 10 | Stage 2 gating vs. recall tradeoff |
| **E** | stage1_model × model | 4 | Model tier selection |
| **F** | stage1_model × threshold × n_votes (high_recall fixed) | 8 | Cost optimization |

Run a group: `python -c "from experiments.ablation import run_study_group; run_study_group('A')"`

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
| 9 | Augmentation robustness | `originals_only: false` in config | ✅ impl |
| 10 | EHS report auto-population | `evaluation/ehs_report.py` | ✅ impl |
| 11 | High-recall cheap Stage 1 gate | `stage1_model` + `high_recall` prompt | ✅ impl |
| 12 | Auto-increment attempt number | `runner.py --next` / `next_attempt_number()` | ✅ impl |

---

## Output Format

Each run writes to `outputs/<timestamp>_<run_id>/`:

| File | Description |
|---|---|
| `predictions.jsonl` | Per-video predictions with confidence, latency, cost |
| `failed_videos.csv` | Videos that errored out |
| `metrics.json` | Aggregate binary + multiclass metrics |
| `report.pdf` | Metrics table + binary CM + multiclass CM |

---

## Dataset

73 real YouTube workplace safety clips across 11 incident categories:
Arc Flash, Caught In Machine, Electrocution, Fall, Fire, Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident.

Videos are identified by row order in `dataset_mapping.xlsx` → `VID000`–`VID072` (0-indexed, 3-digit zero-padded).
