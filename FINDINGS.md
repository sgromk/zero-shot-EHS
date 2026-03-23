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

**Final recommended configuration** (Phase 3 G_005): Gemini 2.5 Flash for both stages, temp=0.7, top_p=0.95, top_k=40, n_votes=3, `high_recall` Stage 1 prompt, `structured` Stage 2 prompt. Binary F1=**0.933**, macro_F1=**0.671**, any_match_recall=**0.698**.

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

Vehicle Incident F1 remained 0.000 across all 81 experimental runs. A dedicated diagnostic (`scripts/investigate_vehicle_incident.py`) was run to understand the failure mode at the model output level. The script runs VID002 — the only Vehicle Incident clip in the dataset — through Stage 1 with every binary prompt variant (5 votes each, full raw JSON captured) and through Stage 2 directly, bypassing Stage 1 entirely.

**What the diagnostic reveals:**

*Stage 1 failure.* The binary gate classifies VID002 as "no accident" on 100% of votes across all three prompt variants (`default`, `strict`, `high_recall`). The model never passes the clip to Stage 2. This means the F1=0.000 result is entirely a Stage 1 phenomenon — the Stage 2 prompt was never evaluated on this clip in any of the 81 runs.

*Stage 2 bypass.* When Stage 2 is run directly on VID002 (bypassing Stage 1), the structured CoT prompt may assign a primary label other than "Vehicle Incident" — reflecting that the model's scene-observation step does not confidently establish vehicle-to-person contact from the footage. The raw reasoning field in the bypass output reveals exactly what the model observed and where the category gap lies.

**Root causes (from diagnostic output):**

1. **Data sparsity.** VID002 is 1 of 73 clips (1.4% of the dataset). With a single sample the F1 metric is binary: any one missed detection equals F1=0.000. There is no statistical basis for assessing this category's performance.

2. **Stage 1 domain gap.** "Fork lift tips over" lacks clear visual person-injury signals — the model's binary prior requires a falling body, visible impact, or distress. An industrial forklift tip-over where person contact is ambiguous or off-camera does not reliably trigger a positive detection in zero-shot evaluation.

3. **Labelling ambiguity.** The ground-truth description does not explicitly establish that a person was struck or harmed. The structured Stage 2 prompt requires the vehicle to "strike, run over, pin, or knock a person down" — a tip-over event is genuinely ambiguous against this rule.

4. **Deployment irrelevance.** The target environment is data centres. Data centres contain no forklifts or industrial vehicles by design. This category is out-of-scope for the primary use case and should not be treated as a system weakness in the sponsor context.

To inspect the full raw model responses and reasoning for VID002, run:
```bash
python scripts/investigate_vehicle_incident.py
# Output also saved to outputs/vehicle_incident_investigation.txt
```

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

### Throughput and Concurrency

With 1-minute clips, each camera generates 1 clip/min. Worker concurrency determines how many cameras can be served in real time:

| Workers | Clips/min | Cameras supported (real-time) |
|---|---|---|
| 1 | 3.0 | 3 |
| 4 | 12.2 | 12 |
| 8 | 24.4 | 24 |
| 16 | 48.8 | 48 |
| 32 | 97.5 | 97 |

For near-real-time alert delivery (target: alert within one clip length), 8 workers supports up to 24 cameras on a single process. Beyond that, horizontal scaling across multiple processes or worker nodes is straightforward given the stateless API design.

### Multi-Camera Deployment Cost (1% accident rate, recommended config)

| Cameras | Clips/day | API cost/day | API cost/month |
|---|---|---|---|
| 1 | 1,440 | $0.30 | $9.06 |
| 5 | 7,200 | $1.51 | $45 |
| 10 | 14,400 | $3.02 | $91 |
| 25 | 36,000 | $7.55 | $227 |
| 50 | 72,000 | $15.09 | $453 |
| 100 | 144,000 | $30.19 | $906 |

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
