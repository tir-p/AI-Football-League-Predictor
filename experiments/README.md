# Experiments

This folder contains the planned experiment suite for the football match prediction project.

Implemented studies:

- feature expansion on top of the current feature matrix
- ablation analysis of feature families
- model comparison and soft-voting ensemble testing
- probability calibration analysis
- rolling out-of-sample validation across future seasons
- feature importance analysis for XGBoost

Outputs are written to `experiments/results`.

Run everything:

```bash
python experiments/run_all_experiments.py
```

Run individual studies:

```bash
python experiments/run_feature_expansion.py
python experiments/run_ablation.py
python experiments/run_model_comparison.py
python experiments/run_calibration_analysis.py
python experiments/run_out_of_sample_validation.py
python experiments/run_feature_importance.py
```

Notes:

- The current Understat collection in this repository only includes match results and xG. Because of that, the feature-expansion study focuses on defensive form, venue-specific strength, head-to-head context, and missing-xG handling instead of shots or goalkeeper statistics.
- The experiment scripts reuse the existing processed files in `data/processed` and do not overwrite the production model artifacts in `models`.
