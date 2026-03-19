# EHS Video Analysis — Workplace Safety Incident Detection

A two-stage zero-shot video classification pipeline for 24/7 workplace safety surveillance using Google Gemini 2.5 via Vertex AI. The system detects accidents in surveillance footage and generates structured EHS incident reports without any fine-tuning or labelled training data.

For full experimental results see [FINDINGS.md](FINDINGS.md).

---

## Why Two Stages

Surveillance footage has an approximately 1% accident rate in active industrial environments. A single-stage classifier would spend 99% of its budget on non-accident clips. The pipeline is explicitly designed around this asymmetry:

- **Stage 1** is cheap and aggressive — a short binary prompt asking only "is there an accident?" runs on every clip. Optimised for recall. Non-accident clips stop here with one API call.
- **Stage 2** is thorough and expensive — a longer structured prompt classifying incident type, reasoning through the scene, and generating an EHS report. Runs only on Stage 1 positives (~1% of clips in deployment). Its per-call cost is effectively negligible at scale.

This keeps the dominant cost as a single cheap Stage 1 call per clip, while delivering rich classification and reporting on the small fraction that matters.

---

## Key Modules

### `pipeline/detection.py` — Stage 1 binary gate

Implements `detect()`, which fires N independent Gemini Flash calls on a video and aggregates them by vote policy (`any` / `majority` / `all`). The `n_votes` parameter enables majority-vote ensembling; votes run in parallel via `ThreadPoolExecutor`. Parallelising within a single clip allows higher n_votes without adding wall-clock latency.

Three binary prompt variants are defined:
- `default` — balanced, flags any physical impact or consequence to a person
- `high_recall` — aggressive, instructs the model to flag any ambiguity and let Stage 2 decide; preferred for production
- `strict` — conservative, requires clear evidence of physical harm; useful for low-false-positive environments

### `pipeline/classification.py` — Stage 2 classifier and EHS report

Implements `classify()`, which calls Gemini Flash with the structured prompt and parses the response. Returns a `ClassificationResult` containing:
- `category` — primary (highest-confidence) incident type
- `categories` — full ranked list of predicted types with confidence scores (multi-label)
- `ehs_report` — structured dict with severity, immediate actions, root cause, contributing factors, corrective measures
- `incident_start_time` / `incident_end_time`, `description`, `raw_response`

The `structured` prompt uses a three-step chain-of-thought: observe the scene → classify all applicable types above 0.5 confidence → generate EHS report. The `default` prompt is single-label, returning one category with no EHS report.

Response parsing handles both old single-label format (for backward compat with baseline runs) and the new multi-label `incidents` list format.

### `pipeline/ingestion.py` — video sourcing

Loads video clips from local disk under `data/videos/VID{n}_{title}/`. Each VID folder contains `original.mp4` plus `type1/` and `type2/` subdirectories of augmented variants. The `originals_only` / `augmented_only` config flags control which subset is loaded. Returns `Part` objects ready for the Vertex AI API.

### `pipeline/postprocessing.py` — parsing and retry

`safe_parse_json()` handles malformed JSON in model responses (strips markdown fences, attempts partial parse). `generate_with_retry()` wraps API calls with exponential backoff on 429 rate-limit errors.

### `experiments/ablation.py` — sweep orchestration

Defines 8 study groups (A–H), each as a Cartesian product of parameter axes. `generate_configs()` expands any group into a flat list of configs. `run_study_group()` runs them sequentially, `compare_results()` prints a metric table. `run_phase1_sweep()` runs groups C→D→F→B→A→E in cost-impact order. `select_best_configs()` applies Pareto selection on (binary_F1, cost) and feeds into `run_phase2_augmented()` / `run_phase3_sweep()`.

### `experiments/runner.py` — single experiment

`ExperimentRunner` takes a config dict, loads videos, runs Stage 1 on all of them (parallelised by `max_workers`), runs Stage 2 only on positives, logs every prediction to JSONL, and calls `evaluation.metrics.evaluate()` at the end to write binary_F1, macro_F1, per_class_F1, cost, and latency into `metrics.json`.

### `evaluation/metrics.py` — evaluation

Loads ground truth from `dataset_mapping.xlsx` and predictions from `predictions.jsonl`, merges on VID key, and computes binary metrics (F1, precision, recall) and multiclass metrics (macro_F1, per-class F1). Also computes `any_match_recall` for multi-label runs: the fraction of accident clips where the ground-truth category appears anywhere in the predicted `categories` list.

### `run_logging/experiment_logger.py` — per-video logging

Thread-safe logger that appends one JSON record per video to `predictions.jsonl` immediately after inference, so partial runs are always recoverable. Estimates API cost using per-token Vertex AI pricing with separate rates for Stage 1 and Stage 2 models.

---

## Ablation Study Design

### Parameter Axes and Group Assignments

The ablation study is structured to vary one axis at a time, holding others at a principled baseline. Groups are run in cheapest/highest-impact order.

| Group | Axis | Configs | Rationale |
|---|---|---|---|
| **C** | binary_prompt × classification_prompt | 6 | Prompt quality is the highest-leverage lever and cheapest to sweep |
| **D** | confidence_threshold × binary_prompt | 10 | Tests whether gating Stage 2 by confidence saves cost without hurting recall |
| **F** | stage1_model × threshold × n_votes | 8 | Core cost-optimisation hypothesis: cheap gate + n_votes=1 |
| **B** | n_votes × vote_policy | 9 | Ensemble quality; n_votes=5 costs 5× on non-accidents, needs justification |
| **A** | temperature × top_p × top_k | 27 | Sampling space; most configs, run after cheaper groups confirm it matters |
| **E** | stage1_model × model | 4 | Model tier; Pro is expensive, run last once other axes are settled |
| **G** | binary_prompt × classification_prompt (Phase 3) | 6 | Re-evaluates prompts with best sampling locked in and working structured prompt |
| **H** | stage1_model × stage2_model (Phase 3) | 4 | Confirms model tier under CoT prompting |

### Why This Run Order

Groups C and F are run first because they answer the most practically important questions — does the prompt wording matter, and can we make Stage 1 dramatically cheaper — at the lowest experimental cost (6 and 8 configs respectively). Group A (27 configs) is deferred until sampling variance is confirmed to be meaningful. Group E (Pro model) is run last because it is the most expensive and Phase 1 already suggested the answer.

### Baseline Config Isolation

Each group holds all other parameters at their baseline to avoid confounding:
- **A**: n_votes=3, prompt=default, threshold=0.0 — sampling must be the only variable
- **B**: temperature=0.3 fixed — at temp=0, all votes are identical (greedy decoding)
- **C**: temperature=0.1, n_votes=3, threshold=0.0 — prompt is the only variable
- **F**: binary_prompt fixed to `high_recall`, `use_frame_fallback` fixed to False — cost-optimised group must never re-check Stage 1 negatives
- **G/H**: A_026 sampling locked in (temp=0.7, top_p=0.95, top_k=40) — Phase 3 varies only prompt and model

### Phased Sweep Design

**Phase 1** (Groups A–F) runs all 64 configs on the 73 original clips. After completion, `select_best_configs()` applies Pareto selection on binary_F1 vs cost, producing a shortlist of 3–5 configs on the efficient frontier.

**Phase 2** re-evaluates that shortlist on the augmented video set (`type1/` and `type2/` variants) to measure robustness under video quality variation. Augmented video IDs embed the original VID key (e.g., `VID042_type1_aug_brightness`) so ground-truth matching works without a separate augmented GT file.

**Phase 3** (Groups G–H) locks in the Phase 1 best sampling parameters and re-evaluates prompt variants with the working structured prompt, then re-confirms model tier under CoT prompting.

---

## Key Findings

Full results in [FINDINGS.md](FINDINGS.md). Summary:

| Finding | Result |
|---|---|
| Best binary detection | F1=0.941 (A_026: temp=0.7, top_p=0.95, top_k=40, n_votes=3, `any`) |
| Most robust config | A_020 (Δ=−0.009 under augmentation) |
| Best cost-optimised | F_006: F1=0.927, $0.011/73 clips (vs $0.021 for A_026) |
| Best multiclass (Phase 3) | G_005: macro_F1=0.671, any_match_recall=0.698 |
| Pro vs Flash | Pro: 4.5–7.4× cost, consistently lower F1 — not justified |
| Flash-Lite gate | binary_F1 drops to 0.805–0.843 — not acceptable for safety |
| CoT prompt improvement | macro_F1 +0.107 over default prompt (0.680 vs 0.573) |
| Production cost | ~$0.30/camera/day at 1% accident rate |
| Throughput (8 workers) | ~24 cameras supported in real time |
| EHS report | Full structured report generated per detected accident |

---

## Dataset

73 animated workplace safety videos sourced from YouTube, spanning 11 incident categories: Arc Flash, Caught In Machine, Electrocution, Fall, Fire, Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident. The evaluation set is dominated by Trip, Struck by Object, and Vehicle Incident. Ground truth is stored in `data/dataset_mapping.xlsx` with one row per clip.

Augmented variants (brightness, contrast, and noise transforms) in `type1/` and `type2/` subfolders are used for robustness evaluation in Phase 2 only.

---

## Output Format

Each experiment run writes to `outputs/<timestamp>_<run_id>/`:

| File | Contents |
|---|---|
| `predictions.jsonl` | One record per video: stage1/2 latency, cost estimate, votes, predicted category, `all_categories`, `ehs_report` |
| `metrics.json` | binary_F1, macro_F1, per_class_F1, any_match_recall, binary_recall, binary_precision, total cost, mean latency |
| `failed_videos.csv` | Videos that errored during inference |
| `config_snapshot.yaml` | Full config used for this run |
