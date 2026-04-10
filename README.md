# Machine Learning for Postoperative Outcome Prediction
## A Clinical Decision Support System for Orthopaedic Rehabilitation

> LightGBM · SHAP · MICE imputation · temporal data shift · cohort mixing · clinical ML · Master's Thesis

A research-grade ML pipeline developed as my Master's Thesis in Data Science, in collaboration with **Istituto Ortopedico Galeazzi** (Milan). The system predicts three postoperative outcomes for patients undergoing hip or knee fracture surgery, with the goal of supporting clinical decision-making, personalising rehabilitation pathways, and optimising hospital resource allocation.

Supervised by **Prof. Federico Cabitza** and **Andrea Campagner, PhD** — Università degli Studi di Milano-Bicocca, A.Y. 2025/2026.

> ⚠️ The raw clinical dataset is not publicly available due to patient privacy constraints. This repository contains code, methodology, and results documentation.

---

## Clinical Targets

Three outcomes modelled simultaneously — each with a distinct task type and clinical interpretation:

| Outcome | Task | Clinical Meaning |
|---------|------|-----------------|
| Length of Stay (LoS) | Regression | Predicted duration of inpatient rehabilitation |
| Social Discharge | Classification | Discharge destination — home vs. institutional care |
| Functional Recovery | Regression | Post-operative autonomy via Barthel Index (BI) |

---

## Dataset & Preprocessing

77 clinical and demographic predictors collected at Galeazzi across two temporal cohorts (2018, 2019).

**Missing data:** 21.8% of pharmaceutical variables missing — handled via **MICE (Multiple Imputation by Chained Equations)**, which models each missing variable as a function of all others. Compared against simpler strategies; MICE improved model R² by approximately +0.05 across outcomes.

---

## Models

Ridge/Lasso · Random Forest · LightGBM · SVM · Multi-Layer Perceptron (MLP)

**Final models:** SVR and LightGBM for LoS; LightGBM with cost-sensitive learning for Social Discharge.

Cost-sensitive learning was applied to Social Discharge to address severe class imbalance — misclassifying a patient who needs institutional care as ready for home discharge has direct clinical consequences, so the loss function is weighted accordingly.

---

## The Central Challenge — Temporal Data Shift

The most technically significant contribution of this thesis. A structural discrepancy was identified between the 2018 and 2019 patient cohorts — not random noise, but a systematic shift in the data-generating process.

**Detection:**
- **Adversarial Validation** — a classifier trained to distinguish 2018 from 2019 records achieved accuracy of **1.0**, confirming the cohorts are statistically separable
- **Kolmogorov-Smirnov tests** — identified specific features driving the distributional divergence

**Impact:** naive training on 2018 data and testing on 2019 caused model collapse — R² dropped to **−0.027** for LoS, indicating the model had learned patterns that did not transfer.

**Mitigation — Cohort Mixing:** strategic blending of both temporal cohorts during training to expose models to distributional variation. Results after mitigation:

| Outcome | Metric | Score |
|---------|--------|-------|
| Length of Stay | R² | 0.253 |
| Length of Stay | MAE | 2.20 days |
| Social Discharge | AUC | 0.84 |
| Social Discharge | Balanced Accuracy | 0.856 |
| Social Discharge | F2-score | 0.656 |

F2-score was chosen over F1 for Social Discharge — in discharge planning, false negatives (patients incorrectly sent home) carry greater risk than false positives, so recall is weighted more heavily.

---

## Explainability

SHAP values and Permutation Importance applied to all final models — not as a post-hoc addition, but as a validation step. The question is not only *does the model perform well*, but *does it perform well for the right reasons*.

**Top SHAP predictors:** surgery duration, age, haemoglobin levels, and baseline functional scores (Barthel Index at admission) — all consistent with established clinical literature on orthopaedic recovery. Alignment between model-identified drivers and domain knowledge is a necessary condition for clinical adoption.

---

## Green AI

Carbon footprint of the training process monitored throughout experimentation — an increasingly relevant consideration in applied ML research and one explicitly tracked in this work.

---

## Key Design Decisions

- **MICE over simple imputation** — preserves multivariate relationships in pharmaceutical data; mean/median imputation would introduce systematic bias in correlated clinical variables
- **Adversarial validation for shift detection** — more sensitive than univariate distributional tests alone; a classifier that perfectly separates two cohorts is unambiguous evidence of structural shift
- **Cohort mixing over reweighting** — directly exposes the model to temporal variation during training rather than correcting for it post-hoc
- **F2 over F1 for Social Discharge** — encodes the asymmetric clinical cost of the two error types into the evaluation metric itself
- **SHAP as clinical validation** — interpretability is not optional in a CDSS; if the model's reasoning cannot be explained to clinicians, it will not be used regardless of performance

---

## Academic Context

| | |
|-|-|
| Degree | MSc in Data Science |
| Institution | Università degli Studi di Milano-Bicocca |
| Supervisor | Prof. Federico Cabitza |
| Co-supervisor | Andrea Campagner, PhD |
| Academic Year | 2025/2026 |
| Research Site | Istituto Ortopedico Galeazzi, Milan |

---

## Stack

`Python` `LightGBM` `scikit-learn` `SHAP` `MICE` `adversarial validation` `clinical ML` `CDSS` `temporal data shift` `cost-sensitive learning`
