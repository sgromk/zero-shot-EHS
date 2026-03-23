# Ablation Study Charts

Six PNG files in this directory. All produced by `notebooks/ablation_sweep.ipynb`.

---

## `pareto_front.png`
**Phase 1 — Cost vs Binary F1 (Pareto Efficient Frontier)**

Scatter plot of all Phase 1/2 configurations (Groups A–F), plotting total inference cost (USD) against binary F1. The dashed staircase traces the Pareto frontier — configs that are not dominated on both axes. Key labelled points: **A_026** (top performer, binary F1=0.941, ~$0.02), **F_006** (cheapest Pareto point), and several B-group configs. Most configs cluster around $0.02. Group E (Gemini Pro for Stage 1) is a clear outlier at >$0.08 with no accuracy benefit.

---

## `parameter_impact.png`
**Parameter Impact on Binary F1 — Phase 1**

Six bar charts (one per ablation group), each showing mean binary F1 ± std across the swept values for that group's parameter. Groups: C (binary_prompt), D (confidence_threshold), F (stage1_model), B (n_votes), A (temperature), E (stage1_model with Pro). Key takeaways: temperature and n_votes have modest effect; using Flash-Lite for Stage 1 drops F1 by ~5 points; the `strict` binary prompt slightly underperforms `default` and `high_recall`.

---

## `robustness_comparison.png`
**Phase 1 vs Phase 2 — Robustness on Augmented Videos**

Side-by-side grouped bar charts for the top 7 Phase 1 configs, comparing performance on original clips (blue) vs augmented clips with lighting/compression/blur distortions (red). Left panel: binary F1 (small drops of 0.01–0.05). Right panel: macro F1 (larger spread; A_020 uniquely *improves* by +0.08 on augmented data). Annotations show the delta. Confirms A_026 as the most consistently robust config.

---

## `phase3_group_g.png`
**Phase 3 Group G — Prompt Engineering (A_026 sampling base)**

Two panels testing all 6 binary × classification prompt combinations, with A_026 sampling params fixed. Left panel: macro_F1 and any_match_recall side by side per combo — the structured classification prompt lifts both metrics for every binary prompt variant (default+structured: macro=0.680, recall=0.674). Right panel: binary F1 per combo — the `high_recall` binary prompt matches the Phase 1 best (0.933, shown as dashed line) regardless of classification prompt. The `default+structured` combo is the best balanced option.

---

## `phase3_group_h.png`
**Phase 3 Group H — Model Tier with Structured Prompt**

Compares four Stage 1 × Stage 2 model combinations, all using the structured CoT prompt. Left bar chart ranks by macro F1: **Flash/Flash** (0.678) beats all others including Flash/Pro (0.628), Flash-Lite/Flash (0.577), and Flash-Lite/Pro (0.567). Right scatter shows cost vs macro F1: Flash/Flash sits at ~$0.025 for 73 videos while Flash/Pro costs ~$0.13 — a 5× cost increase for *lower* performance. Flash-Lite combos save nothing because Stage 2 dominates cost at high accident rates.

---

## `phase3_comparison.png`
**Phase 1 Best vs Phase 3 Best Configurations**

Summary comparison of four configurations across three metrics (binary F1, macro F1, any_match_recall): the Phase 1 baseline (A_026), and three Phase 3 configs (G_001 best macro, G_005 best binary+macro balance, H_000 Flash/Flash). Phase 3 configs all improve macro F1 from 0.635 → ~0.67–0.68 and introduce any_match_recall (~0.67–0.70). Binary F1 trades off slightly (0.941 → 0.905–0.933). H_000 (Flash/Flash) achieves macro=0.678 at the lowest cost, making it the recommended deployment config.
