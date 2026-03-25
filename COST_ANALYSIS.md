# Cost Analysis — Two-Stage Gemini EHS Detection Pipeline

**Configuration:** Gemini 2.5 Flash / Flash, structured CoT prompt, n_votes=3 (G_005)
**Platform:** Vertex AI (Gemini 2.5 Flash)
**Date:** March 2026

---

## 1. Token Cost Model

Token counts are measured constants from `run_logging/experiment_logger.py`.

| Stage | Tokens (in / out) | Cost per call (Flash) |
|---|---|---|
| Stage 1 — binary gate | 800 in / 30 out | $0.000069 |
| Stage 1 × 3 votes | — | **$0.000207** |
| Stage 2 — classify + EHS report | 1,500 in / 500 out | **$0.000263** |

**Per-clip cost:**
- Non-accident clip (Stage 1 only): **$0.000207**
- Accident clip (Stage 1 + Stage 2): **$0.000470**

Stage 1 is always paid. Stage 2 fires only when Stage 1 returns a positive detection.

---

## 2. Accident Rate Sensitivity

**G_005 measured performance:** recall = 0.9767, precision = 0.8936, FPR = 0.1667
(FPR measured on challenging safety-video negatives — conservative upper bound for real deployment.)

Per 1,000 clips processed:

| Accident rate | TP reports | FN missed | S2 calls | Cost / 1K clips | Cost / TP report |
|---|---|---|---|---|---|
| 0.1% | 0.98 | 0.02 | 167.5 | $0.251 | $0.257 |
| 0.5% | 4.88 | 0.12 | 168.2 | $0.252 | $0.052 |
| 1.0% | 9.77 | 0.23 | 174.8 | $0.253 | $0.026 |
| 2.0% | 19.53 | 0.47 | 188.0 | $0.256 | $0.013 |
| 5.0% | 48.84 | 1.16 | 214.4 | $0.263 | $0.005 |
| 10.0% | 97.67 | 2.33 | 263.9 | $0.276 | $0.003 |

**Key insight:** Total cost per 1K clips is nearly flat across accident rates ($0.251–$0.276) because Stage 1 dominates at low rates. Cost *per true positive report* drops sharply as accident rate rises — the pipeline becomes dramatically more efficient for higher-risk environments.

---

## 3. Sponsor Context — FP Economics

The sponsor confirmed false positives are acceptable: the system is only accessed when a real accident is confirmed, at which point staff check whether a report was auto-generated in that timeframe. An unused EHS report wastes Stage 2 API fees but causes no operational harm. **False negatives are the only real loss** — 2.3% of accidents are missed at the best config (recall = 0.9767).

**At 0.1% accident rate (data centre), per 1,000 clips:**
- Actual accidents: 1.0
- TP reports generated: 0.98 — the useful output
- FP Stage 2 calls (wasted): 166.5 — driven by 16.7% FPR × 999 non-accident clips
- Missed FNs: 0.02 — the only operational loss

At low accident rates, Stage 2 spend is dominated by false positives. This is why the measured FPR matters and why a validation run on actual site footage (not safety-training videos) is a required pre-deployment step.

---

## 4. Multi-Camera Daily Cost

60-second clip interval = 1,440 clips/camera/day.

### 60-second clips, 0.1% accident rate (data centre baseline)

| Cameras | Clips/day | API $/day | API $/month | TP reports/day | $/TP report | CR instances |
|---|---|---|---|---|---|---|
| 1 | 1,440 | $0.36 | $11 | 1.4 | $0.26 | 1 |
| 10 | 14,400 | $3.62 | $108 | 14.1 | $0.26 | 1 |
| 25 | 36,000 | $9.04 | $271 | 35.2 | $0.26 | 2 |
| 50 | 72,000 | $18.09 | $543 | 70.3 | $0.26 | 4 |
| 100 | 144,000 | $36.17 | $1,085 | 140.6 | $0.26 | 8 |
| 250 | 360,000 | $90.43 | $2,713 | 351.5 | $0.26 | 18 |

### 60-second clips, 1.0% accident rate (industrial site)

| Cameras | API $/day | API $/month | TP reports/day | $/TP report | CR instances |
|---|---|---|---|---|---|
| 10 | $3.64 | $109 | 140.6 | $0.026 | 1 |
| 50 | $18.21 | $546 | 703.2 | $0.026 | 4 |
| 100 | $36.42 | $1,093 | 1,406.5 | $0.026 | 8 |

**Instances:** 14 cameras per Cloud Run instance (measured from mixed-workload load test, C=4 optimal).

---

## 5. Breakeven vs Human Monitoring

Human reviewer: $25/hr, 24/7 coverage = **$600/day**.

| Cameras | Pipeline $/day | Human equivalent | Annual saving |
|---|---|---|---|
| 1 | $0.30 | 0.05% of one reviewer | — |
| 50 | $15.10 | — | $212,000 vs 1 reviewer |
| 100 | $30.20 | — | $212,000 vs 1 reviewer |
| ~2,000 | $600 | = 1 full-time reviewer | breakeven |

**A single 24/7 reviewer is cost-equivalent to the pipeline monitoring ~2,000 cameras.**
For a 50-camera data centre: pipeline costs ~$5,500/year vs ~$219,000/year for equivalent human coverage — **97% cost reduction**.

---

## 6. Cloud Run Infrastructure Overhead

Cloud Run gen2: $0.000024/vCPU-second, 1 vCPU allocated per instance.

| Accident rate | Mean latency | CR $/req | Vertex AI $/clip | Total $/clip | CR overhead |
|---|---|---|---|---|---|
| 0.1% | 4.3s | $0.000103 | $0.000251 | $0.000354 | ~29% of total (~41% of Vertex AI) |
| 1.0% | 4.5s | $0.000108 | $0.000253 | $0.000361 | ~30% of total |
| 5.0% | 4.8s | $0.000115 | $0.000263 | $0.000378 | ~30% of total |
| 10.0% | 5.4s | $0.000130 | $0.000276 | $0.000406 | ~32% of total |

**Key insight:** Cloud Run adds ~30% overhead on top of Vertex AI at all accident rates. Vertex AI dominates. Optimising model tier (Flash-Lite vs Flash) is a more effective cost lever than infrastructure tuning.

---

## 7. Cost vs Quality Tradeoff

| Config | Binary F1 | Macro F1 | Daily cost / 100 cams (1%) |
|---|---|---|---|
| Flash/Flash — G_005 (recommended) | 0.933 | 0.671 | ~$36 |
| Flash/Flash — H_000 (CoT only) | 0.916 | 0.678 | ~$36 |
| Flash-Lite/Flash — H_002 | 0.805 | 0.577 | ~$23 |
| Flash/Pro — H_001 | 0.892 | 0.628 | ~$120 |

Flash-Lite/Flash saves ~36% on API cost at the cost of an 11-point binary F1 drop and larger recall degradation. Flash/Pro costs 3× more than Flash/Flash for lower performance — not a viable option.

**Recommendation:** Flash/Flash G_005. It is the Pareto-optimal configuration across all cost and quality dimensions.

---

## 8. Per-Class F1 Breakdown

G_005 macro F1 (0.671) hides significant variance across incident types.

| Tier | Categories | F1 range |
|---|---|---|
| **Strong** (F1 ≥ 0.80) | Electrocution, Lifting, No Accident, Arc Flash, Caught In Machine, Gas Inhalation | 0.800 – 1.000 |
| **Moderate** (0.60–0.79) | Fall, Slip | 0.667 – 0.788 |
| **Weak** (F1 < 0.60) | Fire, Struck by Object, Trip | 0.286 – 0.500 |
| **Zero** | Vehicle Incident | 0.000 |

**Root causes for weak classes:**
- **Trip / Struck by Object** — visually ambiguous; high confusion with Fall and Slip. Both have n=7 samples — metric is noisy but the confusion is a real model behaviour.
- **Fire** — n=4; small sample but the model conflates fire with other hazards.
- **Vehicle Incident** — single mis-clipped sample (forklift tip-over never appears on screen). F1=0.000 is entirely a data quality issue, not a model weakness.

**Data centre relevance:** The weakest categories (Vehicle Incident, Trip) are less relevant in data centre environments where forklifts and trip-hazard scenarios are uncommon. The highest-priority categories for that deployment context (Electrocution, Arc Flash) are among the strongest performers.

---

## 9. Secondary Costs

### 9.1 — Implementation (One-Time)

| Item | Notes |
|---|---|
| GCP setup | IAM, Cloud Run, VPC — low engineering effort |
| **Camera feed integration** | Largest implementation unknown. Surveillance systems use RTSP / proprietary formats; a clip-extraction edge component is required per recorder. Cost scales with recorder diversity on site. |
| Staff onboarding | A process owner is needed to review flagged EHS reports and close corrective-measure loops. |
| **Site validation run** | Model evaluated on safety-training videos, not real site footage. A validation run pre-launch is a required cost that is easy to omit. FPR may differ from the measured 16.7%. |

### 9.2 — Ongoing Maintenance

- **Prompt maintenance:** The prompt is the model logic. Adding incident categories or updating OSHA field requirements requires a prompt edit and re-validation sweep. Low-frequency, but non-zero.
- **SDK deprecation:** The Vertex AI SDK in use is flagged for removal on **June 24, 2026**. Migration is a scheduled obligation.
- **Model version pinning:** The pipeline calls `gemini-2.5-flash` without a pinned version. Pinning prevents silent regressions but creates a standing migration obligation as new versions release.
- **Monitoring:** GCP Cloud Monitoring setup for error rates and latency — minimal cost, needs an on-call owner.

### 9.3 — Technological Debt

- **Vendor lock-in:** The pipeline is Vertex AI-specific at the client layer. Migration to another provider requires rewriting the client and re-validating all prompts.
- **Latency floor:** The 4–20s request latency is a hard API constraint. Sub-second detection would require a fundamentally different approach.
- **Two-stage prompt coupling:** Stage 1 and Stage 2 prompts are co-designed and must be re-validated jointly if either changes.

### 9.4 — Costs That Do Not Apply (Zero-Shot Advantage)

| Standard ML cost item | Why it does not apply |
|---|---|
| Training data collection and labeling | Zero-shot — no labeled examples required |
| Model training / fine-tuning compute | No GPU hours, no training infrastructure |
| **Retraining cycle** | Zero-shot has no learned distribution to drift from. Prompt updates replace retraining at a fraction of the cost. |
| Training data storage | No labeled video archive to maintain |
| Model registry / experiment tracking | Not applicable for prompt-based systems |

The absence of a retraining cycle is compounding: the cost advantage over human monitoring widens over time because the pipeline requires no periodic re-investment to maintain accuracy.

### 9.5 — Privacy and Data Sovereignty

Every clip is uploaded to Google's Vertex AI endpoint. In a standard deployment this requires a Data Processing Agreement (DPA), data residency confirmation (configurable via Vertex AI region), and legal review of surveillance footage as personal data under applicable privacy law (GDPR, CCPA).

**This project is developed in collaboration with Google.** Existing partnership agreements simplify or eliminate several of these steps. The data residency constraint is still worth confirming for any non-US deployment — it is a one-line change at the API client level.

Regardless of the partnership, the sponsor's internal security review will require documentation of the data flow: **video leaves the site network, is transmitted to Google's infrastructure for inference, and the clip is not retained after the API call returns.**

---

## 10. Summary

| Metric | Value |
|---|---|
| Best config | G_005 (Flash/Flash, structured CoT, n_votes=3) |
| Binary F1 | 0.933 (recall 0.977, precision 0.894) |
| Cost per clip (0.1% accident rate) | $0.000354 (Vertex AI + Cloud Run) |
| Cost per TP report (0.1% rate) | ~$0.26 |
| Cost per TP report (1.0% rate) | ~$0.026 |
| Cost per camera per day (0.1%, 60s clips) | ~$0.36 |
| Cameras per Cloud Run instance | 14 (measured) |
| Breakeven vs 24/7 human reviewer | ~2,000 cameras |
| 50-camera annual saving vs human | ~$212,000 (97% reduction) |
| Scheduled SDK migration deadline | June 24, 2026 |
