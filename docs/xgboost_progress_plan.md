# XGBoost Progress and Improvement Plan

Last updated: 2026-06-20

## Current State

The active predictor is the saved XGBoost model at `models/xgboost.json`.
The best aggregate-metric experiment is the feature-v1 XGBoost model at
`models/experiments/xgboost_features_v1.json`.
The best behavioral-sanity experiment is the feature-v2 XGBoost model at
`models/experiments/xgboost_features_v2.json`.

Current command paths:

- CLI prediction, active legacy model: `.venv\Scripts\python.exe -m models.ev_cli`
- CLI prediction, v1 experiment: `.venv\Scripts\python.exe -m models.ev_cli --features v1`
- CLI prediction, v2 experiment: `.venv\Scripts\python.exe -m models.ev_cli --features v2`
- Full legacy evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --out models\evaluation_current.txt`
- Full v1 evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --features v1`
- Full v2 evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --features v2`
- Monotonic sanity checks: `.venv\Scripts\python.exe -m models.monotonic_checks --features v2`
- Legacy training: `.venv\Scripts\python.exe -m models.xgboost_model`
- V1 training: `.venv\Scripts\python.exe -m models.xgboost_model --features v1`
- V2 training: `.venv\Scripts\python.exe -m models.xgboost_model --features v2`

Output convention: keep generated reports, experiment models, and smoke-test artifacts inside this repository. Do not write project outputs to `C:\tmp`.

## Current Model Behavior

The current model predicts a residual over the "game ended now" baseline:

```text
target = (final_score + final_uma) / 1000
baseline = (start_score + start_uma) / 1000
model target = target - baseline
final EV = baseline + predicted_residual - 25
```

The four predicted seat values are recentered so the implied `(score + uma) / 1000` total is exactly `100.0`, making EVs zero-sum.

Current feature shape is 9 columns:

```text
[wind_id, round, honba_bucket, riichi_bucket, seat, rotated_score_0, rotated_score_1, rotated_score_2, rotated_score_3]
```

Feature-v1 is 29 columns. It keeps the 9 legacy columns and adds:

```text
seat score, seat baseline EV, score gaps to opponents, gaps to first/fourth/next higher/next lower,
sorted score gaps, rotated current places, and rotated current uma values
```

Feature-v2 is 23 columns. It keeps the 9 legacy columns and adds smooth score-gap
and phase/dealer features, but intentionally does not include current place,
current uma, or sorted placement-gap features because those create hard jumps
around ties.

Feature-v2 also changes the target mode:

```text
model target = final EV directly, in thousands relative to 25k
final EV = predicted EV, recentered so all four seats sum to zero
```

V2 uses XGBoost monotone constraints for the clearest directional score features:
own score and own score gaps are constrained positive; rotated opponent scores
are constrained negative.

Known data detail: the current `data/rounds.db` stores winds as integers (`0`, `1`, `2`). The XGBoost path now treats `0/E` as East and `1/S` as South while skipping West continuation rounds.

## Latest Evaluation

Report: `models/evaluation_current.txt`

Evaluation setup:

- Model: `models/xgboost.json`
- DB: `data/rounds.db`
- Validation slice: last 10% of DB rows
- Calibration buckets: 20

Headline metrics:

- Rounds evaluated: `147357`
- Rounds skipped (W+): `817`
- Model MSE: `6224.7852`
- Model RMSE: `78.8973`
- Model MAE, diagnostic only: `60.6755`
- Correlation with true EV: `0.5780`
- Avg abs calibration-bucket diff: `2.031`
- Zero-sum check: `0.0000` average sum, `0.0000` max absolute sum

Special E1 all-25k check:

- Rounds in exact state: `13960`
- Actual average EV: `+1.763, -0.612, -0.917, -0.281`
- Model prediction: `+0.417, -0.447, +0.130, -0.100`

## Feature-v1 Experiment

Artifacts:

- Model: `models/experiments/xgboost_features_v1.json`
- Full report: `models/experiments/evaluation_features_v1.txt`
- Summary JSON: `models/experiments/evaluation_features_v1_summary.json`

Training/evaluation setup:

- Training matrix: `5,893,900` rows x `29` features
- DB: `data/rounds.db`
- Validation slice: last 10% of DB rows
- Calibration buckets: 20

Headline metrics:

- Rounds evaluated: `147357`
- Rounds skipped (W+): `817`
- Model MSE: `6085.1030`
- Model RMSE: `78.0071`
- Model MAE, diagnostic only: `60.1296`
- Correlation with true EV: `0.5904`
- Avg abs calibration-bucket diff: `0.594`
- Zero-sum check: `0.0000` average sum, `0.0000` max absolute sum

Change versus current legacy model:

- RMSE improved by `0.8902` thousand points, about `1.13%`.
- MAE improved by `0.5459` thousand points, about `0.90%`.
- Correlation improved from `0.5780` to `0.5904`.
- Avg abs calibration-bucket diff improved from `2.031` to `0.594`, about a `70.7%` reduction.

Special E1 all-25k check:

- Actual average EV: `+1.763, -0.612, -0.917, -0.281`
- V1 model prediction: `+0.298, -0.056, -0.262, +0.020`

## Feature-v2 Experiment

Artifacts:

- Model: `models/experiments/xgboost_features_v2.json`
- Full report: `models/experiments/evaluation_features_v2.txt`
- Summary JSON: `models/experiments/evaluation_features_v2_summary.json`
- Monotonic checks: `models/experiments/monotonic_checks_v2.txt`

Training/evaluation setup:

- Training matrix: `5,893,900` rows x `23` features
- Target mode: direct EV
- DB: `data/rounds.db`
- Validation slice: last 10% of DB rows
- Calibration buckets: 20

Headline metrics:

- Rounds evaluated: `147357`
- Rounds skipped (W+): `817`
- Model MSE: `6114.9849`
- Model RMSE: `78.1984`
- Model MAE, diagnostic only: `60.2954`
- Correlation with true EV: `0.5878`
- Avg abs calibration-bucket diff: `0.839`
- Zero-sum check: `0.0000` average sum, `0.0000` max absolute sum

Change versus v1:

- RMSE is worse by `0.1913` thousand points.
- MAE is worse by `0.1658` thousand points.
- Correlation is lower by `0.0026`.
- Avg abs calibration-bucket diff is worse by `0.245`.
- Monotonic sanity violations improved from `4` to `0`.

Monotonic sanity checks:

- Legacy: `13` violations
- V1: `4` violations
- V2: `0` violations

Specific sanity cases:

- `S 4 0 0 40000 30000 14400 15600`, increasing seat 2 by 100-point steps while decreasing seat 3, is monotone for seat 2 under v2.
- `S 1 0 0 35000 35000 14900 15100`, increasing seat 2 by 100-point steps while decreasing seat 3, is monotone for seat 2 under v2.
- `E 2 0 0 25000 25000 25000 25000` v2 prediction: `-0.954, -0.285, -0.011, +1.250`.

## Main Read

The current legacy model is not behaviorally reliable around close placement
boundaries because the residual target is built on a discontinuous "game ended
now" uma baseline. Calibration buckets did not catch this. Monotonic sanity
checks are now a required model-selection metric alongside RMSE, correlation,
calibration, and zero-sum checks.

V1 is currently best on aggregate predictive metrics. V2 is currently best on
the score-transfer monotonicity checks that exposed the strange behavior.

The current feature set gives the model raw rotated scores, but it does not directly expose several concepts that matter for placement EV:

- current placement
- score gaps between placements
- score gaps from the predicted seat to each opponent
- dealer-relative information
- remaining hands / game phase
- all-last and near-all-last structure
- current baseline EV as an explicit feature

## Improvement Plan

### 1. Lock Down Evaluation Comparisons

Before changing features, make evaluation easier to compare across experiments.

Status: done.

- Add a compact machine-readable summary output, e.g. `models/evaluation_current_summary.json`.
- Keep the full text report for human inspection.
- Add report fields for model path, DB path, git commit if available, feature version, and timestamp.
- Avoid overwriting canonical reports unless explicitly requested.
- Store experiment reports under `models/experiments/`.

Success condition: every experiment can be compared by RMSE, calibration-bucket average diff, zero-sum checks, and key group metrics.

### 2. Centralize Feature Encoding

Create one feature builder used by both training and inference.

Status: done in `models/features.py`.

Goals:

- Training and prediction must always use the exact same feature order.
- The feature list should have a version/name.
- The evaluator should record the feature version.
- Feature generation should be unit-testable on a few fixed states.

This prevents silent mismatches as features are added.

### 3. Add Placement and Gap Features

First feature experiment should stay simple and high-signal.

Status: done as feature version `v1`.

Candidate additions:

- seat current score
- current rank/place of each seat
- predicted seat current place
- current uma for each seat
- baseline EV for predicted seat
- score gap from predicted seat to each opponent
- score gap to next higher placement
- score gap to next lower placement
- score gap to first
- score gap to fourth
- sorted score gaps: first-second, second-third, third-fourth

Train this as a new model artifact, not over `models/xgboost.json`, until it wins clearly.

Suggested artifact names:

- `models/experiments/xgboost_features_v1.json`
- `models/experiments/evaluation_features_v1.txt`
- `models/experiments/evaluation_features_v1_summary.json`

### 4. Add Round Phase and Dealer Features

Second feature experiment should encode game context more directly.

Status: done as feature version `v2`, with direct EV targets and monotone constraints.

Candidate additions:

- dealer seat
- whether predicted seat is dealer
- dealer current place
- hands remaining until normal S4 end
- is East round
- is South round
- is South 3
- is South 4
- is all-last
- is dealer leading in all-last
- honba and riichi buckets retained

These features should help the model learn that the same score gap means different things in E1, S3, and S4.

### 5. Tune XGBoost After Feature Gains

Do hyperparameter tuning after feature engineering, not before.

Candidates:

- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `learning_rate`
- `n_estimators`
- `reg_lambda`
- `reg_alpha`

Use early stopping with a validation split from the training slice, while preserving the final held-out evaluation slice for reporting.

### 6. Consider Specialized Models Only After That

Only after feature-enhanced XGBoost plateaus:

- one model per broad phase: early, mid, S3, S4
- one model per exact round group
- post-model calibration correction
- neural network / MLP

A neural network is not the next best step because this is structured tabular data and the current model likely lacks direct EV-relevant features more than raw model capacity.

## Recommended Next Step

Keep `models/xgboost.json` unchanged until an experimental model wins clearly and is intentionally promoted.

The next experiment should improve v2 without reintroducing discontinuities:

- expand monotonic checks to randomly sampled score-transfer states
- tune XGBoost hyperparameters for v2 direct EV targets
- test phase-specific models for early, mid, S3, and S4 states
- consider a smooth post-model calibrator only if it preserves monotonic checks
