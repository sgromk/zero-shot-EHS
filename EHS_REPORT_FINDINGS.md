# EHS Report Quality Findings

**Pipeline:** Gemini 2.5 Flash, structured CoT prompt, temperature=0.7
**Sample:** 10 accident clips, random seed=42
**Stage:** Stage 2 direct (no Stage 1 gate — all clips are known accidents)
**Date:** March 2026

---

## 1. Summary Statistics

| Metric | Value |
|---|---|
| Reports generated | 10 / 10 (100% success) |
| Mean field completeness | 9.0 / 9 fields (100%) |
| Reports with scene assessment (CoT reasoning) | 10 / 10 |
| Mean specificity (root_cause + corrective_measures word count) | ~68 words |
| Severity distribution | Low: 2 · Medium: 6 · High: 1 · Critical: 1 |

All 10 reports populated all 9 OSHA 300-series fields without any generic placeholder values. Every report also included a free-form `SCENE ASSESSMENT` block drawn from the model's chain-of-thought reasoning pass before classification.

---

## 2. Per-Report Detail

| Video | Predicted Category | Confidence | Severity | Specificity (WC) | Notes |
|---|---|---|---|---|---|
| VID040 | Arc Flash | 100% (+ Fire 90%) | Hospitalisation | ~86 | LOTO violation identified; detailed PPE gap analysis |
| VID007 | Fall | 100% | Medical Treatment | ~72 | Belt-buckle rule cited; spotter recommendation |
| VID001 | Fall | 100% | Medical Treatment | ~72 | Platform edge identified as root cause; guardrail recommendation |
| VID017 | Fall | 100% | **Life-Threatening/Fatal** | ~77 | Makeshift platform failure; highest severity in sample |
| VID015 | Struck by Object (+ Trip 70%) | 90% | Medical Treatment | ~57 | PPE focus (foot protection); dual-category detection |
| VID014 | Trip | 95% | Medical Treatment | ~79 | Housekeeping root cause; material storage corrective plan |
| VID008 | Struck by Object | 95% | First Aid Only | ~77 | Near-miss correctly classified; tool tethering recommendation |
| VID006 | Trip | 100% | First Aid Only | ~53 | Cable management root cause; shortest corrective section |
| VID034 | Struck by Object | 100% | Medical Treatment | ~52 | Dropped wrench from height; exclusion zone recommendation |
| VID005 | Slip | 100% | Medical Treatment | ~56 | Warning sign present but ignored — noted in contributing factors |

---

## 3. Field Completeness

All 9 fields were populated across all 10 reports:

| Field | Notes on quality |
|---|---|
| `pre_incident_activity` | Consistently specific — describes task, posture, and equipment in use |
| `what_happened` | Strong narrative; matches visual scene description in scene assessment |
| `injury_description` | Appropriately hedged ("potential for", "not directly observed") in near-miss cases |
| `direct_agent` | Correctly identifies the physical object (wrench, wet floor, cable, arc flash) |
| `immediate_actions` | Weakest field — VID008 explicitly stated "Not shown in the video"; others are generic protocols |
| `root_cause` | Strong — consistently identifies the systemic failure (LOTO, housekeeping, PPE) |
| `contributing_factors` | Good — distinguishes systemic from proximal causes |
| `corrective_measures` | Strongest field — actionable, specific to incident type, references named standards (LOTO, OSHA) |
| `severity` | Appropriate calibration (see Section 4) |

---

## 4. Severity Calibration

| Level | Count | Clips |
|---|---|---|
| First Aid Only | 2 | VID008 (hammer near-miss), VID006 (cable trip) |
| Medical Treatment Required | 6 | VID007, VID001, VID015, VID014, VID034, VID005 |
| Hospitalisation | 1 | VID040 (arc flash) |
| Life-Threatening / Fatal | 1 | VID017 (fall from makeshift platform) |

**Assessment:** Calibration is qualitatively sensible. VID017's "Life-Threatening/Fatal" rating matches the animated scenario (significant fall from improvised elevated platform). VID008's "First Aid Only" correctly reflects the near-miss nature of the incident. The arc flash receiving "Hospitalisation" rather than "Critical" is arguably conservative — arc flash PPE failures typically carry critical injury potential — but is defensible given the video is a training animation.

---

## 5. Specificity Analysis

Mean combined word count for `root_cause` + `corrective_measures` ≈ **68 words** across the sample.

**High specificity (≥75 words):**
- VID040 (~86): Named-standard detail ("LOTO", "arc flash rated PPE", "arc flash hazard assessment")
- VID014 (~79): Precise housekeeping protocol with material storage specifics
- VID017 (~77): Scaffold certification and makeshift equipment prohibition
- VID008 (~77): Tool tethering policy, toe boards, netting — distinct corrective items

**Moderate specificity (50–74 words):**
- VID007 (~72): Belt-buckle rule is a named best practice — above-average signal
- VID001 (~72): Guardrail installation and operating procedure update
- VID015 (~57): Mandatory footwear policy — actionable but narrow
- VID005 (~56): Barricading and non-slip footwear — standard slip guidance

**Lower specificity (<55 words):**
- VID006 (~53): Cable management policy is correct but brief
- VID034 (~52): Correct recommendations but concise — exclusion zone + tool tethering only

No report produced a generic or vacuous corrective section ("train employees on safety"). All recommendations were grounded in the specific incident mechanism.

---

## 6. Scene Assessment (CoT Reasoning) Quality

Every report included a `SCENE ASSESSMENT` block, which is the model's free-form observation of the video before classifying. This is the most differentiating aspect of the structured CoT prompt.

**Strengths observed:**
- Correctly identifies the precipitating action in all 10 cases (e.g., "The worker's foot became entangled on an unsecured plank", "An unexpected electrical arc flash occurred")
- Describes spatial relationships accurately (worker position relative to ladder, platform edge, other workers)
- VID034 captures the two-worker dynamic (worker dropping wrench above; worker struck in boom lift below) — this level of multi-person scene understanding is notable
- VID005 specifically notes the ignored warning sign: "walking past a 'CAUTION WET FLOOR' sign and subsequently slips" — directly informs contributing factors field

**Limitations observed:**
- VID008 (near-miss) generates a scene assessment consistent with near-miss classification but then assigns "First Aid Only" severity — slight tension between narrative ("worker reacts defensively, indicating a near-miss") and the injury fields, which correctly log "no direct injury"
- Scene assessments for the VID015/VID014 pair (same Woodlift series) are contextually distinct but both anchor on foot/trip hazards — correct but shows the model relies on visible signage ("Risk of Foot Injury" label visible in VID015)

---

## 7. Category Prediction Observations

Predicted categories across the 10-clip sample:

| Category | Count | Clips |
|---|---|---|
| Struck by Object | 3 | VID015, VID008, VID034 |
| Fall | 3 | VID007, VID001, VID017 |
| Trip | 2 | VID014, VID006 |
| Arc Flash | 1 | VID040 |
| Slip | 1 | VID005 |

**VID001 flag:** VID001 is "Electric Forklift Safety Accident" — the ground-truth label is likely `Vehicle Incident`, but the model predicted `Fall` (the worker falls backward off a raised platform while operating a pallet jack). The OSHA report is coherent and the corrective measures are appropriate, but the category is wrong. This is consistent with the G_005 per-class breakdown where `Vehicle Incident` F1 = 0.000 due to the category's small and inconsistently-clipped sample.

**VID015 flag:** Ground-truth is likely `Lifting` (Woodlift series), but the model predicted `Struck by Object`. The incident involves wooden planks and a foot injury — visually closer to struck/trip than a lifting manoeuvre. The report's content is accurate and actionable regardless of the category label.

---

## 8. Key Takeaways

1. **Field completeness is not a limiting factor.** The structured prompt reliably populates all 9 OSHA fields. This is a resolved capability.

2. **Specificity is the quality differentiator.** Reports vary more in depth of root cause analysis than in whether fields are present. The corrective measures field is consistently the strongest; `immediate_actions` is the weakest (often defaults to "call EMS, secure the area").

3. **The CoT scene assessment substantially improves report coherence.** By anchoring all downstream fields in a free-form visual observation, the model avoids generic OSHA boilerplate. The scene description feeds directly into accurate direct agent identification and contributing factors.

4. **Severity calibration is appropriate for the sample.** No systematic over- or under-severity observed. The near-miss (VID008) correctly lands at First Aid Only.

5. **Category accuracy at Stage 2 direct does not match ablation metrics.** The Stage 2 prompt is designed to work downstream of the Stage 1 gate, which filters to confirmed accidents. Running Stage 2 directly on raw clips (as here) is the correct use case for report generation, but category accuracy may be slightly lower than the gated pipeline produces. The two misclassifications observed (VID001 → Fall, VID015 → Struck by Object) are consistent with the known weak categories in the G_005 per-class F1 breakdown.

6. **No hallucinated fields.** Where the model lacked visual evidence (e.g., exact time of incident in VID006), it either used placeholder values or noted it was not visible — it did not fabricate specifics.
