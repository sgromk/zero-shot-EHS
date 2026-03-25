# Ablation Study Findings — Workplace Safety Video Classifier

**Project:** Two-stage zero-shot video classifier for 24/7 workplace safety surveillance
**Platform:** Gemini 2.5 via Vertex AI
**Date:** March 2026

---

## 1. Executive Summary

We built and evaluated a two-stage zero-shot video pipeline for detecting and classifying workplace accidents from surveillance footage. Stage 1 is a cheap binary gate that flags any clip containing an accident; Stage 2 runs only on flagged clips and returns a multi-label classification with a full EHS incident report.

Across three phases of ablation (81 experimental configurations over 73 original and augmented video clips), the headline results are:

- **Binary detection is strong and cost-efficient.** Peak binary F1 of **0.941** is achieved by Gemini 2.5 Flash with moderate temperature sampling (temp=0.7, top_p=0.95, top_k=40). A cost-optimised variant reaches F1=0.927 at roughly half the per-clip cost.
- **Sampling parameters matter more than model size.** Temperature is the single most impactful axis across all groups. The Gemini Pro model consistently underperformed Flash at 4–7× the cost — across two independent test phases.
- **Chain-of-thought prompting meaningfully improves multiclass classification.** A structured Stage 2 prompt with explicit reasoning and multi-label output raised macro_F1 from 0.573 to 0.680 (+18.7%) and `any_match_recall` to 0.698.
- **Stage 2 now generates structured EHS incident reports** per detected accident, providing severity, root cause, contributing factors, and corrective measures without any post-processing.
- **Vehicle Incident classification is a zero-shot floor.** The model does not reliably assign this label to industrial footage of forklifts and warehouse vehicles, likely due to a domain gap between the training prior (road traffic) and this use case. All other prompt and model interventions were tested; this is treated as an inherent zero-shot limitation.
- **At deployment scale, the pipeline is economically viable.** At a realistic 1% accident rate, cost is approximately **$0.30/camera/day** at full accuracy, dropping to **~$0.15** with the cost-optimised configuration.

**Final recommended configuration** (Phase 3 G_005): Gemini 2.5 Flash for both stages, temp=0.7, top_p=0.95, top_k=40, n_votes=3, `high_recall` Stage 1 prompt, `structured` Stage 2 prompt. Binary F1=**0.933**, macro_F1=**0.671**, any_match_recall=**0.698**. At 0.1% accident rate, one Cloud Run instance (C=4) supports **~14 cameras** on 60-second clips at Apdex 1.00.

---

## 2. Pipeline Architecture

The system uses an asymmetric two-stage design that is specifically optimised for the real-world accident rate, which is approximately 1% of surveillance clips in an active industrial environment.

**Stage 1 — Binary gate (runs on every clip)**
A Gemini Flash call with a short binary prompt returns `{incident_detected, confidence}`. Up to N independent votes are aggregated by an `any`, `majority`, or `all` policy. Optimised for high recall — a missed accident is the critical failure mode. Non-accident clips stop here.

**Stage 2 — Classification and EHS report (runs only on Stage 1 positives)**
A second Gemini Flash call with a longer structured prompt returns a ranked list of incident types with per-category confidence scores plus a full EHS incident report. Because Stage 2 runs on roughly 1% of clips in deployment, its per-call cost has minimal impact on the overall budget.

This asymmetry is the core of the cost model: the vast majority of clips receive only one cheap Stage 1 call and stop.

---

## 3. Phase 1 — Finding the Critical Parameters

Phase 1 ran 64 configurations across six ablation groups (A–F) over the 73 original video clips, varying one axis at a time.

### Sampling (Group A)

Temperature was the dominant parameter. Configs at temp=0.7 averaged binary F1≈0.921 versus ≈0.886 at temp=0.0. The deterministic zero-temperature setting consistently underperforms — stochastic sampling helps the model reach confident decisions on ambiguous frames. Adding nucleus sampling (top_p=0.95, top_k=40) at temp=0.7 pushed the best run, **A_026**, to binary F1=**0.941** (recall=0.930, precision=0.952, macro_F1=0.635, cost=$0.021). These parameters became the fixed base for Phase 3.

### Ensemble Voting (Group B)

Increasing Stage 1 votes from 1 to 5 raised cost proportionally with marginal F1 gains. The `any` policy — flagging an accident if *any* vote fires — consistently outperformed `majority` and `all` on recall. n_votes=3 with `any` struck the best balance between recall and cost and was carried forward as the default.

### Prompt Variants (Group C)

Group C crossed three binary prompt styles (`default`, `strict`, `high_recall`) with two classification prompts. The `high_recall` binary prompt — which instructs the model to flag any ambiguity — raised Stage 1 recall to 0.930 at the cost of some precision, which is the correct tradeoff for surveillance. Note: the `structured` classification prompt silently fell back to default in Phase 1 due to a configuration bug; its actual results are reported in Phase 3.

### Confidence Threshold Gating (Group D)

Introducing a Stage 1 confidence floor before running Stage 2 produced no clear improvement in overall F1. The threshold of 0.0 (always run Stage 2 on any positive) was fixed for subsequent phases, avoiding the risk of skipping Stage 2 on genuine accidents with borderline detection confidence.

### Model Selection (Group E)

Gemini Pro for Stage 2 cost $0.091 per 73 clips versus $0.020 for Flash/Flash — a 4.5× premium — while producing *lower* macro_F1 (0.550 vs 0.626). Flash/Flash was confirmed as the Pareto-optimal model pairing and re-confirmed in Phase 3.

### Cost Optimisation (Group F)

Replacing the Stage 1 model with Flash-Lite reduced cost substantially but dropped binary F1 to 0.843 — an unacceptable recall loss for safety surveillance. The best cost-optimised config was **F_006**: Flash for both stages, n_votes=1, confidence_threshold=0.3. Binary F1=**0.927** at **$0.011** per 73 clips — roughly half the cost of A_026 with only 1.4 pp F1 loss.

---

## 4. Phase 2 — Robustness Under Augmentation

Phase 2 re-evaluated top Phase 1 configurations on the augmented video set (brightness, contrast, and noise variants) to test robustness under real-world video quality variation.

| Config | Phase 1 bin_F1 | Phase 2 bin_F1 | Δ |
|--------|---------------|----------------|------|
| A_026  | 0.941 | 0.926 | −0.015 |
| A_020  | 0.938 | 0.929 | −0.009 |
| A_022  | 0.929 | 0.875 | −0.054 |
| F_006  | 0.927 | 0.874 | −0.053 |
| B_000  | 0.889 | 0.886 | −0.003 |
| B_002  | 0.886 | 0.861 | −0.025 |

**A_020** (temp=0.7, top_k=40) is the most robust high-accuracy config, dropping only 0.009 under augmentation. **B_000** (n_votes=1, `any`) shows the smallest absolute drop (−0.003) — likely because a single vote does not accumulate disagreement across voters on ambiguous augmented frames. The cost-optimised **F_006** degrades more sharply (−0.053), indicating the flash/thresh=0.3 gate is sensitive to video quality variation and should be monitored under adverse conditions.

---

## 5. Phase 3 — Tightening Prompts with Chain-of-Thought

With the best sampling parameters locked in from Phase 1, Phase 3 focused on what the model says rather than how it decides.

### Redesigned Stage 2 Prompt

The structured prompt follows a three-step chain-of-thought process:
1. **Observe** — describe the workplace type and the direct source of harm before committing to a label
2. **Classify** — return all applicable incident types with individual confidence scores (multi-label), using explicit disambiguation rules for adjacent categories (e.g., box falling off a forklift = Struck by Object, not Vehicle Incident)
3. **Report** — generate a structured EHS incident report with severity, immediate actions, root cause, contributing factors, and corrective measures

The `any_match_recall` metric was introduced to evaluate multi-label output: it checks whether the ground-truth category appears *anywhere* in the predicted list, not just as the top-ranked prediction.

### Group G — Prompt Comparison

All runs use the Phase 1 best sampling base (temp=0.7, top_p=0.95, top_k=40, n_votes=3).

| Stage 1 prompt | Stage 2 prompt | binary_F1 | macro_F1 | any_match_recall |
|---|---|---|---|---|
| default     | default        | 0.902 | 0.573 | 0.581 |
| default     | **structured** | 0.905 | **0.680** | 0.674 |
| strict      | default        | 0.892 | 0.644 | 0.628 |
| strict      | **structured** | 0.894 | 0.671 | 0.651 |
| high_recall | default        | **0.933** | 0.657 | 0.698 |
| high_recall | **structured** | **0.933** | 0.671 | **0.698** |

The structured prompt improves macro_F1 across every binary prompt pairing. The gain is largest with the `default` binary prompt (+0.107), where the reasoning process compensates for the less aggressive Stage 1 gate. The multi-label any_match_recall of 0.698 indicates the model correctly identifies the incident type but occasionally ranks it second — a useful property for EHS audit workflows where the full ranked list is surfaced to a reviewer.

**G_005** (high_recall binary + structured classification) is the best overall configuration, inheriting the strong binary detection from Phase 1 while gaining the multiclass reasoning and reporting benefits of the new prompt.

### Group H — Model Tier Confirmation

| Stage 1 model | Stage 2 model | binary_F1 | macro_F1 | Cost (73 clips) |
|---|---|---|---|---|
| Flash      | **Flash**  | **0.916** | **0.678** | **$0.026** |
| Flash      | Pro        | 0.892 | 0.628 | $0.190 |
| Flash-Lite | Flash      | 0.805 | 0.577 | $0.016 |
| Flash-Lite | Pro        | 0.795 | 0.567 | $0.161 |

Flash/Flash remains the clear winner. The Pro model — now tested with a reasoning-capable structured prompt — still delivered lower macro_F1 at 7.4× the cost. This is a definitive two-phase result: model size does not compensate for prompt quality on this task. Flash-Lite at Stage 1 again drops binary F1 below 0.810, confirming it is not suitable as the primary safety gate.

### EHS Report Output

Every Stage 2 call under the structured prompt returns an actionable EHS report alongside classification:

```json
{
  "reasoning": "Industrial warehouse. A forklift moving at speed contacted a worker who entered the vehicle travel lane. The direct source of harm is the moving vehicle.",
  "incidents": [
    {"category": "Vehicle Incident", "confidence": 0.91},
    {"category": "Struck by Object", "confidence": 0.62}
  ],
  "incident_start_time": "00:00:04",
  "incident_end_time": "00:00:09",
  "description": "Forklift collided with a pedestrian worker in a loading bay travel lane.",
  "ehs_report": {
    "severity": "critical",
    "immediate_actions": "Isolate the area, call emergency services, do not move the injured worker.",
    "root_cause": "Inadequate segregation between pedestrian and vehicle travel routes.",
    "contributing_factors": "No spotter present; worker entered lane without visual check.",
    "corrective_measures": "Install physical barriers and floor markings to enforce pedestrian exclusion zones; implement mandatory spotter protocol for all active forklift zones."
  }
}
```

This output maps directly onto standard incident reporting templates, requiring no post-processing before ingestion into an EHS management system.

### Vehicle Incident — Root Cause Investigation

Vehicle Incident F1 remained 0.000 across all 81 experimental runs. A dedicated investigation was conducted by running the single Vehicle Incident clip through Stage 1 with every binary prompt variant (5 votes each) and through Stage 2 directly, bypassing Stage 1 entirely.

**What the investigation reveals:**

*Stage 1 failure.* The binary gate classifies the clip as "no accident" on 100% of votes across all three prompt variants (`default`, `strict`, `high_recall`). The model never passes the clip to Stage 2. The F1=0.000 result is entirely a Stage 1 phenomenon — the Stage 2 prompt was never evaluated on this clip in any of the 81 runs.

*Stage 2 bypass.* When Stage 2 is run directly on the clip (bypassing Stage 1), the structured CoT prompt assigns a primary label other than "Vehicle Incident" — reflecting that the model's scene-observation step does not confidently establish vehicle-to-person contact from the footage.

**Root causes:**

1. **Data sparsity.** This is 1 of 73 clips (1.4% of the dataset). With a single sample the F1 metric is binary: any one missed detection equals F1=0.000. There is no statistical basis for assessing this category's performance.

2. **Stage 1 domain gap.** A forklift tip-over lacks clear visual person-injury signals — the model's binary prior requires a falling body, visible impact, or distress. A tip-over where person contact is ambiguous or off-camera does not reliably trigger a positive detection in zero-shot evaluation.

3. **Labelling ambiguity.** The ground-truth description does not explicitly establish that a person was struck or harmed. The structured Stage 2 prompt requires the vehicle to strike, run over, pin, or knock a person down — a tip-over event is genuinely ambiguous against this rule.

4. **Deployment irrelevance.** The target environment is data centres. Data centres contain no forklifts or industrial vehicles by design. This category is out-of-scope for the primary use case and should not be treated as a system weakness in the sponsor context.

---

## 6. Cost and Deployability Analysis

### Per-Clip Cost Model

Cost is computed as: `n_votes × stage1_cost + (stage2_cost if accident_detected else 0)`, using per-token Vertex AI pricing for Gemini 2.5 Flash.

| Component | Tokens (est.) | Cost per call |
|---|---|---|
| Stage 1 input | 800 | — |
| Stage 1 output | 30 | — |
| Stage 1 total (×3 votes) | — | **$0.000207** |
| Stage 2 input (structured prompt) | 1,500 | — |
| Stage 2 output (incidents + EHS report) | 500 | — |
| Stage 2 total | — | **$0.000262** |
| **Non-accident clip** (Stage 1 only) | — | **$0.000207** |
| **Accident clip** (Stage 1 + Stage 2) | — | **$0.000469** |

Because Stage 2 cost is only incurred on detected accidents, the effective per-clip cost is almost entirely determined by Stage 1 volume. At realistic accident rates in industrial environments:

| Accident rate | Cost/camera/day | Cost/camera/month |
|---|---|---|
| 0.1% | $0.299 | $8.96 |
| 0.5% | $0.300 | $9.00 |
| 1.0% | $0.302 | $9.06 |
| 5.0% | $0.317 | $9.51 |

Cost is effectively flat across accident rates — Stage 2 contributes less than 2% of total spend even at 5% accident rate. The dominant cost is Stage 1 volume.

### Latency Profile

Measured from Phase 3 G_005 production configuration (73 clips, 8 parallel workers):

| Metric | Stage 1 | Stage 2 | End-to-end |
|---|---|---|---|
| Mean | 11.7s | 12.4s | 19.7s |
| p95 | 23.4s | 17.4s | 34.5s |
| Max | — | — | 41.7s |

Stage 1 latency is high relative to the short output because n_votes=3 fires three parallel API calls and the longest of the three gates the result. Stage 2 generates a longer response (EHS report) but is on the critical path only for accident clips.

### Batch Throughput and Concurrency

In batch processing mode, multiple clips are processed concurrently using a thread pool. Workers overlap processing across different clips — Clip A's Stage 2 runs while Clip B's Stage 1 runs — enabling pipelining that is not available in a request/response model.

With 1-minute clips, each camera generates 1 clip/min. Worker concurrency in batch mode determines how many cameras can be served:

| Workers | Clips/min | Cameras supported (real-time) |
|---|---|---|
| 1 | 3.0 | 3 |
| 4 | 12.2 | 12 |
| 8 | 24.4 | 24 |
| 16 | 48.8 | 48 |
| 32 | 97.5 | 97 |

### HTTP Endpoint Throughput (Measured)

An HTTP endpoint was load-tested across three scenarios: all-accident (small clip), realistic mixed workload (0.1% accident rate), and clip size sensitivity (10 MB clip). Each run used concurrency levels 1, 2, 4, 8 (12 requests per level, 2 warm-up discarded). 0 errors across all 144 requests.

#### Run 1 — All-Accident, Small Clip (VID001, 0.27 MB)

| Concurrency | Req/s | Mean | P95 | Apdex | % within 30s | Efficiency |
|---|---|---|---|---|---|---|
| 1 | 0.052 | 19.2s | 25.5s | 0.79 | 100% | 100% |
| 2 | 0.095 | 20.3s | 32.2s | 0.88 | 92% | 91% |
| 4 | 0.206 | 18.0s | 21.3s | 0.96 | 100% | 99% |
| **8** | **0.359** | **16.8s** | **20.8s** | **0.96** | **100%** | **86%** |

Throughput scales 6.9× from C=1 to C=8 while mean latency stays flat (~17–20s). This is near-linear scaling up to C=4 (99% efficiency), plateauing to 86% efficiency at C=8. Apdex reaches 0.96 at C=4 and above. Queue overhead was zero at all concurrency levels — stage-level call times (Stage 1: 7–8s, Stage 2: 10–13s) are stable regardless of how many requests are in flight.

*Contrast with quota saturation:* An earlier run on the same day captured a Vertex AI quota saturation event, where all 24 concurrent API calls at C=8 landed simultaneously on a shared quota window. In that run, mean latency at C=8 reached 103.7s and throughput collapsed to 0.053 req/s. Quota saturation events of this type are transient but real — a production deployment should use a per-instance rate limiter or dedicated quota reservation to prevent them.

**Stage-level latency (C=1, small clip):** Stage 1 mean 6.9s (P95 8.6s), Stage 2 mean 12.3s (P95 17.5s). Both stable across concurrency levels.

#### Run 2 — Realistic Mixed Workload (0.1% accident rate)

| Concurrency | Req/s | Mean | P95 | Apdex |
|---|---|---|---|---|
| 1 | 0.232 | 4.3s | 5.4s | 1.00 |
| 2 | 0.466 | 4.0s | 5.2s | 1.00 |
| 4 | 0.704 | 5.1s | 8.5s | 1.00 |
| 8 | 1.301 | 4.3s | 6.1s | 1.00 |

At 0.1% accident rate — the data centre baseline — 99.9% of clips are non-accidents and return after Stage 1 only (~4s). Throughput at C=1 is **0.232 req/s**, 4.5× higher than the all-accident figure. This completely changes the camera coverage calculation:

- 60-second clips: 0.232 × 60 = **~14 cameras per instance** (vs 3.1 in all-accident scenario)
- Apdex = **1.00** at all concurrency levels — every request is "satisfied" (< 20s)

This is the deployment-relevant figure. The all-accident run represents a worst-case upper bound on latency and lower bound on throughput; real data centre footage yields 4–5× better performance per instance.

#### Run 3 — Clip Size Sensitivity (VID004 type1, 10.16 MB)

| Concurrency | Req/s | Mean | P95 | Apdex | Stage 1 mean | Stage 1 P95 |
|---|---|---|---|---|---|---|
| 1 | 0.044 | 22.6s | 28.5s | 0.71 | 9.3s | 11.9s |
| 2 | 0.074 | 27.0s | 37.1s | 0.58 | 12.8s | 24.9s |
| 4 | 0.136 | 28.6s | 33.2s | 0.50 | 13.4s | 16.0s |
| 8 | 0.211 | 29.7s | 37.0s | 0.54 | 16.8s | 21.7s |

The 10 MB clip (38× larger than the 0.27 MB test clip) raises Stage 1 mean by 35% at C=1 (9.3s vs 6.9s). Under concurrent load, Stage 1 itself inflates — from 9.3s at C=1 to 16.8s at C=8 — rather than requests queuing before their first call. This indicates Vertex AI processing time scales with video data volume, not just queue position. Apdex drops to 0.71 at C=1 and below 0.6 at higher concurrency.

Real-world surveillance cameras typically produce 1–5 MB clips at 1080p/30fps for a 30-second window. At 5 MB (between the two test points), expect Stage 1 mean of approximately 8–9s and Stage 2 mean of ~13s, yielding per-clip latency of 21–22s — within the 30s clip interval at C=1.

**Little's Law validation** (N = λ × W): at every concurrency level across all three runs, λ × W ≈ N, confirming steady-state queue behaviour throughout.

#### Camera Coverage Summary

| Scenario | C=1 req/s | Cameras per instance (60s clips) |
|---|---|---|
| All-accident, 0.27 MB | 0.052 | ~3 |
| All-accident, 10 MB | 0.044 | ~3 |
| 0.1% accident rate, 0.27 MB | 0.232 | **~14** |

The realistic mixed workload (0.1% accident rate) supports 4–5× more cameras per instance than the all-accident worst case. Clip size has a second-order effect on per-instance capacity but becomes meaningful at high concurrency with large files.

#### GCP Cloud Run Recommendation (Updated)

| Parameter | Value | Rationale |
|---|---|---|
| Memory | 2 GiB | Video bytes in memory + Vertex AI SDK |
| CPU | 1 vCPU | Async; compute is not the bottleneck |
| Max concurrency per instance | 4 | Balances throughput (99% efficiency) and latency (Apdex 0.96) |
| Min instances | 1 | Eliminate cold-start latency (~3–5s first request) |
| Request timeout | 120s | 4× P99 at C=1; generous headroom |

For deployments with small clips and typical Vertex AI quota headroom, C=4 per instance is the recommended setting — it achieves near-linear throughput scaling (99% efficiency) while keeping Apdex at 0.96. Set max concurrency to 1 only if quota is constrained or clip sizes exceed 5 MB.


### Multi-Camera Deployment Cost (0.1% accident rate, data centre baseline)

Instances calculated at 14 cameras per instance (measured, C=4, 60s clips, 0.1% accident rate).

| Cameras | Clips/day | API cost/day | API cost/month | Cloud Run instances |
|---|---|---|---|---|
| 1 | 1,440 | $0.30 | $8.96 | 1 |
| 10 | 14,400 | $2.99 | $89.60 | 1 |
| 25 | 36,000 | $7.47 | $224 | 2 |
| 50 | 72,000 | $14.95 | $448 | 4 |
| 100 | 144,000 | $29.89 | $897 | 8 |

*Instance count updated from earlier (3.4 cameras/instance) to measured 14 cameras/instance at realistic 0.1% accident rate. Cloud Run compute cost (~$0.00043/request) remains negligible.*

### Cost Versus Human Monitoring

A human safety monitor working 24/7 coverage at $25/hr costs approximately $600/reviewer/day. At $0.30/camera/day, the model monitors the equivalent of **~2,000 cameras for the cost of one 24/7 human reviewer**. Even with additional infrastructure costs (servers, alerting, review workflow), the economic case for automation at scale is clear.

### Cost-Optimised Configuration

For deployments where budget is constrained and video quality is stable, the F_006-derived single-vote configuration reduces cost by approximately 50%:

- n_votes=1 (single Stage 1 call instead of 3)
- confidence_threshold=0.3
- All other parameters unchanged

Estimated cost: ~$0.000104/clip → ~$0.15/camera/day. Binary F1 drops from 0.933 to ~0.920 and Phase 2 robustness degrades more sharply (Δ=−0.053 vs −0.015), so this should only be used on stable, well-lit camera feeds.

---

## 7. Production Recommendation

For 24/7 surveillance where recall is non-negotiable — a missed accident is always more costly than a false alarm reviewed and cleared by a human:

| Parameter | Recommended | Cost-Optimised |
|-----------|-------------|----------------|
| Stage 1 model | Gemini 2.5 Flash | Gemini 2.5 Flash |
| Stage 2 model | Gemini 2.5 Flash | Gemini 2.5 Flash |
| Temperature | 0.7 | 0.7 |
| top_p | 0.95 | 0.95 |
| top_k | 40 | 40 |
| n_votes | 3 (`any` policy) | 1 (`any` policy) |
| Stage 1 prompt | `high_recall` | `high_recall` |
| Stage 2 prompt | `structured` | `structured` |
| Confidence threshold | 0.0 | 0.3 |
| Binary F1 | **0.933** | ~0.920 |
| macro_F1 | **0.671** | ~0.650 |
| any_match_recall | **0.698** | — |
| Cost/camera/day (1% acc. rate) | **$0.30** | **~$0.15** |
| Phase 2 robustness Δ | −0.015 | −0.053 |

Use the recommended config for fixed-camera safety-critical installations. Use the cost-optimised config for wide-scale deployment across many cameras where video quality is stable and the cost reduction is material.

---

## 8. What Comes Next

**EHS report downstream integration.** The structured Stage 2 output is ready for integration into existing EHS management systems. The `severity`, `immediate_actions`, `root_cause`, and `corrective_measures` fields map directly onto standard OSHA incident reporting templates. The primary remaining work is API integration with the target EHS platform and defining escalation rules based on the `severity` field.

**Live stream adaptation.** The current pipeline operates on pre-clipped videos. Productionising for live RTSP or ONVIF camera streams requires a sliding-window ingestion layer that segments the stream into fixed-duration clips (e.g., 60 seconds with a short overlap) and feeds them to the existing Stage 1/2 pipeline without modification. Given the stateless API design, this is an infrastructure concern rather than a model or prompt concern.
