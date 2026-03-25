# EHS Video Analysis — Workplace Safety Incident Detection

A zero-shot video classification pipeline for 24/7 workplace safety surveillance using Google Gemini 2.5 Flash via Vertex AI. The system detects accidents in surveillance footage and generates structured OSHA-aligned EHS incident reports — with no labelled training data and no fine-tuning.

Developed in collaboration with Google.

---

## The Problem

Industrial surveillance systems generate continuous footage across dozens or hundreds of cameras. Reviewing it for safety incidents is impractical at human scale. The goal is a system that can:

1. Watch all footage continuously and flag clips containing accidents
2. For each flagged clip, identify the incident type and produce an actionable EHS report
3. Do so at a cost and accuracy that justifies deployment over a human reviewer

---

## The Approach

### Why Zero-Shot

The dataset contains 73 labelled clips — far too few to train a reliable supervised classifier. A diagnostic study across three video architectures (FrameCNN, LRCN, CNN3D) confirmed this empirically: all three models overfit to training clips within 1–5 epochs, with best val accuracy peaking at epoch 1–3 and degrading immediately after. Five incident categories achieve F1=0.000 across all models and all cross-validation folds due to data scarcity. Gemini 2.5 Flash, operating zero-shot, outperforms the best trained model (CNN3D, F1=0.862) with a mean binary F1 of 0.933 — without any training data, GPU infrastructure, or retraining cycle. See [TRAINABLE_MODEL_FINDINGS.md](TRAINABLE_MODEL_FINDINGS.md) for the full diagnostic.

### Why Two Stages

Surveillance footage has a very low accident rate — roughly 1% in active industrial environments. Running a full classification and report generation call on every clip would be wasteful. The pipeline is designed around this asymmetry:

- **Stage 1** is a cheap binary gate: a short prompt asks only "is there an accident?" and runs on every clip. Optimised for recall so that accidents are not silently missed. Non-accident clips terminate here.
- **Stage 2** is a full classification and report generation call, running only on Stage 1 positives (~1% of clips in deployment). It classifies the incident type, reasons through the scene using a chain-of-thought prompt, and generates a structured EHS report.

This keeps the dominant cost as a single cheap call per clip, while delivering rich output on the small fraction that matters.

---

## Results

### Detection Performance

Best configuration: **G_005** — Gemini 2.5 Flash / Flash, structured chain-of-thought prompt, n_votes=3, majority vote.

| Metric | Value |
|---|---|
| Binary F1 | 0.933 |
| Recall | 0.977 |
| Precision | 0.894 |
| False Positive Rate | 0.167 |
| Multiclass Macro F1 | 0.671 |
| Any-match Recall | 0.698 |

**2.3% of accidents are missed** (false negative rate). The sponsor confirmed false positives are operationally acceptable — the system is only accessed when a real accident is confirmed by staff, at which point they check whether a report was auto-generated. False negatives are the only real operational loss.

### Multiclass Performance by Category

| Tier | Categories |
|---|---|
| Strong (F1 ≥ 0.80) | Electrocution, Lifting, No Accident, Arc Flash, Caught In Machine, Gas Inhalation |
| Moderate (0.60–0.79) | Fall, Slip |
| Weak (< 0.60) | Fire, Struck by Object, Trip |
| Zero | Vehicle Incident (data quality issue — incident not on screen) |

The strongest categories (Electrocution, Arc Flash) are the highest-priority incident types in a data centre deployment context.

### Key Findings from the Ablation Study

- **Chain-of-thought prompting** is the single highest-leverage improvement: +0.107 macro F1 over a default prompt (0.680 vs 0.573)
- **n_votes=3** with majority policy provides the best precision-recall balance; n_votes=1 saves cost at acceptable recall degradation for low-risk environments
- **Gemini Pro** costs 4.5–7.4× more than Flash and consistently underperforms it — not justified
- **Flash-Lite** at Stage 1 drops binary F1 to 0.805–0.843 — unacceptable for safety-critical detection
- Sampling parameters (temperature, top_p, top_k) have measurable but secondary impact; temp=0.7, top_p=0.95, top_k=40 is the best-found setting

Full results across all 64 ablation configurations in [FINDINGS.md](FINDINGS.md).

### EHS Report Quality

10 OSHA-aligned incident reports were evaluated across all 11 incident categories. Key findings:

- 100% field completion across all OSHA-required fields (severity, immediate actions, root cause, contributing factors, corrective measures, regulatory reference)
- Mean report specificity: ~68 words of substantive content per report
- Severity calibration is accurate: Electrocution and Arc Flash consistently rated Critical; Trip and Lifting rated Low/Medium
- Scene descriptions reference specific environmental details (wet surfaces, overhead cables, machine guard positions) rather than generic language in 8 of 10 cases
- All reports include applicable OSHA standard references (e.g. 29 CFR 1910.303 for electrical, 1910.212 for machine guarding)

Full quality analysis in [EHS_REPORT_FINDINGS.md](EHS_REPORT_FINDINGS.md) (gitignored).

---

## Cost

Full analysis in [COST_ANALYSIS.md](COST_ANALYSIS.md). Summary at the most relevant operating point (60-second clips, 0.1% accident rate):

| Scale | API cost/day | Cost/month | Cost per TP report |
|---|---|---|---|
| 1 camera | $0.36 | $11 | $0.26 |
| 10 cameras | $3.62 | $108 | $0.26 |
| 100 cameras | $36.17 | $1,085 | $0.26 |

Cloud Run infrastructure adds ~30% on top of Vertex AI at all accident rates.

**Breakeven vs a 24/7 human reviewer ($25/hr):** the pipeline is cost-equivalent at ~2,000 cameras. At 50 cameras, the annual saving is ~$212,000 — a 97% cost reduction.

The absence of a retraining cycle is a compounding advantage: the pipeline requires no periodic re-investment to maintain accuracy. Prompt updates replace retraining at a fraction of the cost.

---

## Limitations and Pre-Deployment Requirements

- **The model was evaluated on safety-training videos, not real site footage.** FPR in deployment may differ from the measured 16.7%. A validation run on actual site footage is a required pre-deployment step.
- **Vehicle Incident F1 = 0.000** due to a single mis-clipped sample where the incident never appears on screen. Not a model weakness.
- **Latency floor of 4–20 seconds per clip** is a hard API constraint. Sub-second detection requires a different architectural approach.
- **Vertex AI SDK migration required by June 24, 2026** (current SDK flagged for deprecation).
- Every clip is transmitted to Google's infrastructure for inference. A data processing agreement and data residency confirmation are required for non-US deployments.

---

## Dataset

73 animated workplace safety videos spanning 11 incident categories: Arc Flash, Caught In Machine, Electrocution, Fall, Fire, Gas Inhalation, Lifting, Slip, Struck by Object, Trip, Vehicle Incident. Augmented variants (brightness, contrast, noise) in `type1/` and `type2/` subfolders are used for robustness evaluation only and are never mixed into validation folds.
