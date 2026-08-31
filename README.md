# PLF-OGT (DRCM): Auditable concept–residual dynamics for multi-organ trajectory reconstruction

Code repository for the manuscript "用于重症多器官轨迹重构的可审计概念—残差动态表征
(Auditable concept–residual dynamics for multi-organ trajectory reconstruction in critical illness)".

The framework (**DRCM**: Dynamic Representation with Concept–Residual Modeling) is implemented
as **PLF-OGT** (Patient-Level Flow-aware Organ-Grounded Transformer). It decomposes the patient
state into clinically anchored concept states S and a complementary residual state R, and is
audited for proxy verifiability, information complementarity, patient specificity, and
cross-training semantic stability.

## Repository structure

```
figures/           Figure generation scripts (Figs 2–5): frozen-data assertions (tol 1e-3),
                   unified style block, 300 dpi PNG/PDF output, per-figure data archives.
analysis_frozen/   Frozen analysis pipeline (53 scripts, V14 snapshot): cohort construction,
                   label generation, model training, evaluation, bootstrap CIs, ablations.
editorial/         Manuscript/table surgery tooling (docx/xlsx programmatic edits).
requirements.txt   Python dependencies.
```

## Data availability

MIMIC-IV is available via PhysioNet (credentialed access). GMUICU patient-level data are not
publicly shareable (de-identification / ethics conditions). This repository contains code and
aggregated result summaries only — no patient-level data.

## Reproducibility notes

- All plotting values are asserted against frozen result files (tolerance 0.001) before rendering.
- See the manuscript Supplementary Data (Supplementary_Data_1.xlsx) for per-analysis summaries.
- Tag `v1.0-nc-submission` corresponds to the submitted manuscript version.
