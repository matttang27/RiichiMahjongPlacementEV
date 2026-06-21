# XGBoost Progress and Improvement Plan

Last updated: 2026-06-21

## Current State

The active predictor is the saved XGBoost model at `models/xgboost.json`.
The best aggregate-metric experiment is the feature-v1 XGBoost model at
`models/experiments/xgboost/v1/model.json`.
The best behavioral-sanity experiment is the feature-v2 XGBoost model at
`models/experiments/xgboost/v2/model.json`.

Current command paths:

- CLI prediction, active legacy model: `.venv\Scripts\python.exe -m models.ev_cli`
- CLI prediction, v1 experiment: `.venv\Scripts\python.exe -m models.ev_cli --features v1`
- CLI prediction, v2 experiment: `.venv\Scripts\python.exe -m models.ev_cli --features v2`
- Full legacy evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --out models\evaluation_current.txt`
- Full v1 evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --features v1`
- Full v2 evaluation: `.venv\Scripts\python.exe -m models.evaluate_model --features v2`
- Oracle diagnostics are included in evaluations; defaults are `--oracle-min-rounds 20 --oracle-score-bucket 1000`
- Monotonic sanity checks: `.venv\Scripts\python.exe -m models.monotonic_checks --features v2`
- Legacy training: `.venv\Scripts\python.exe -m models.xgboost_model`
- V1 training: `.venv\Scripts\python.exe -m models.xgboost_model --features v1`
- V2 training: `.venv\Scripts\python.exe -m models.xgboost_model --features v2`
- NN smoke training: `.venv\Scripts\python.exe -m models.nn_model --max-rows 5000 --max-iter 5 --model models\experiments\sklearn_nn\joint_v1_smoke\model.joblib`
- NN full training: `.venv\Scripts\python.exe -m models.nn_model`
- NN evaluation: `.venv\Scripts\python.exe -m models.evaluate_nn`
- NN monotonic checks: `.venv\Scripts\python.exe -m models.monotonic_checks_nn`
- PyTorch NN smoke training: `.venv\Scripts\python.exe -m models.torch_nn_model --max-rows 5000 --epochs 3 --batch-size 1024 --hidden-dim 128 --layers 3 --model models\experiments\torch_nn\joint_v1_smoke\model.pt --training-log models\experiments\torch_nn\joint_v1_smoke\training_log.json`
- PyTorch NN full training: `.venv\Scripts\python.exe -m models.torch_nn_model`
- PyTorch NN evaluation: `.venv\Scripts\python.exe -m models.evaluate_torch_nn`
- PyTorch NN monotonic checks: `.venv\Scripts\python.exe -m models.monotonic_checks_torch_nn --random-cases 100`

Output convention: keep generated reports, experiment models, and smoke-test artifacts inside this repository. Do not write project outputs to `C:\tmp`.

Experiment artifacts are organized by model family and experiment name:

```text
models/experiments/
  xgboost/<version>/
  sklearn_nn/<experiment>/
  torch_nn/<experiment>/
```

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

The first neural-net experiment is `nn_joint_v1`. It uses one row per game
state, predicts all four seat EVs jointly, and recenters outputs so the four
predictions sum to zero. It is implemented with scikit-learn's `MLPRegressor`
inside a scaling pipeline, avoiding a PyTorch/TensorFlow dependency for the
first pass.

The first custom neural-net experiment is `torch_joint_v1`. It uses the same
full-state feature shape as `nn_joint_v1`, predicts all four seat EVs jointly,
and enforces zero-sum in the network forward pass. Training uses direct final-EV
targets plus an optional monotonicity penalty from random score-transfer pairs.

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

- Model: `models/experiments/xgboost/v1/model.json`
- Full report: `models/experiments/xgboost/v1/evaluation.txt`
- Summary JSON: `models/experiments/xgboost/v1/summary.json`

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

- Model: `models/experiments/xgboost/v2/model.json`
- Full report: `models/experiments/xgboost/v2/evaluation.txt`
- Summary JSON: `models/experiments/xgboost/v2/summary.json`
- Monotonic checks: `models/experiments/xgboost/v2/monotonic_checks.txt`

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

## Joint NN Experiment

Status: smoke-tested only.

Artifacts from smoke run:

- Model: `models/experiments/sklearn_nn/joint_v1_smoke/model.joblib`
- Full report: `models/experiments/sklearn_nn/joint_v1_smoke/evaluation.txt`
- Summary JSON: `models/experiments/sklearn_nn/joint_v1_smoke/summary.json`
- Monotonic checks: `models/experiments/sklearn_nn/joint_v1_smoke/monotonic_checks.txt`

Implemented commands:

- Training: `.venv\Scripts\python.exe -m models.nn_model`
- CLI prediction: `.venv\Scripts\python.exe -m models.nn_cli`
- Evaluation: `.venv\Scripts\python.exe -m models.evaluate_nn`
- Monotonic checks: `.venv\Scripts\python.exe -m models.monotonic_checks_nn`

Design:

- Input: one full game-state row, not one seat row.
- Output: four EVs jointly.
- Target: direct final EV for all four seats.
- Inference: subtract mean prediction so outputs are zero-sum.
- Default training fraction: first 90% of DB rows, preserving the evaluator's
  last-10% validation slice.

Smoke run:

- Training matrix: `4,969` rows x `38` features
- Hidden layers: `128,64`
- Max iterations: `5`
- Monotonic sanity violations: `0`
- The smoke model is not a quality result; it only verifies the pipeline.

## PyTorch Joint NN Experiment

Status: smoke-tested only.

Artifacts from smoke run:

- Model: `models/experiments/torch_nn/joint_v1_smoke/model.pt`
- Training log: `models/experiments/torch_nn/joint_v1_smoke/training_log.json`
- Full report: `models/experiments/torch_nn/joint_v1_smoke/evaluation.txt`
- Summary JSON: `models/experiments/torch_nn/joint_v1_smoke/summary.json`
- Monotonic checks: `models/experiments/torch_nn/joint_v1_smoke/monotonic_checks.txt`

Implemented commands:

- Training: `.venv\Scripts\python.exe -m models.torch_nn_model`
- CLI prediction: `.venv\Scripts\python.exe -m models.torch_nn_cli`
- Evaluation: `.venv\Scripts\python.exe -m models.evaluate_torch_nn`
- Monotonic checks: `.venv\Scripts\python.exe -m models.monotonic_checks_torch_nn`

Default full-training configuration:

- Hidden dimension: `256`
- Layers: `4`
- Dropout: `0.05`
- Batch size: `4096`
- Epochs: `20`
- Optimizer: `AdamW`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Monotonic penalty weight: `0.05`
- Monotonic transfer delta: `0.1` thousand points

Design:

- Input: one full game-state row, not one seat row.
- Output: four EVs jointly.
- Target: direct final EV for all four seats.
- Network forward pass subtracts the mean output, so the model is zero-sum by construction.
- Target scaling uses one shared EV scale across all seats, preserving zero-sum in scaled space.
- Training can add monotonicity penalty by transferring points from a random donor to a random recipient and penalizing recipient-EV decreases.
- Monotonic check command can include random sampled states with `--random-cases`.

Smoke run:

- Training matrix: `4,969` rows x `38` features
- Hidden dimension/layers: `128 x 3`
- Epochs: `3`
- Validation scaled MSE: `0.6609`
- Evaluation RMSE on 500 held-out rounds: `80.9834`
- Avg abs calibration-bucket diff on 500 held-out rounds: `6.672`
- Monotonic sanity violations: `0`, including `5` random sampled transfer states
- The smoke model is not a quality result; it only verifies the pipeline.

## Main Read

The current legacy model is not behaviorally reliable around close placement
boundaries because the residual target is built on a discontinuous "game ended
now" uma baseline. Calibration buckets did not catch this. Monotonic sanity
checks are now a required model-selection metric alongside RMSE, correlation,
calibration, and zero-sum checks.

Evaluations now include an empirical oracle / noise-floor diagnostic. It groups
validation states two ways:

- exact state: `(wind, round, honba, riichi, exact start scores)`
- coarse state: `(wind, round, honba bucket, riichi bucket, start scores rounded to nearest score bucket)`

For each eligible group, the oracle prediction is the leave-one-out mean actual
EV of the other rounds in that group. This prevents singleton groups from
looking artificially perfect. It is not the true perfect RMSE; it is a
repeated/similar-state variance diagnostic.

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

- `models/experiments/xgboost/v1/model.json`
- `models/experiments/xgboost/v1/evaluation.txt`
- `models/experiments/xgboost/v1/summary.json`

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
- train and evaluate `nn_joint_v1` on a larger sample or full training slice
- train and evaluate `torch_joint_v1` on the full training slice
- consider a smooth post-model calibrator only if it preserves monotonic checks
