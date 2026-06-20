# Riichi Mahjong Placement EV (Agent Notes)

This repo models **expected value at the start of a round** from Tenhou (Houou) logs.

## Big picture
- Data pipeline: `data/unzip.py` parses Tenhou `.mjson(.gz)` inside yearly `YYYY.zip` files and builds a SQLite DB with one row per *kyoku start*, labeled by the *final game scores*.
- Model: `ev_model.py` trains an XGBoost regressor that predicts a **residual over a baseline** (current score + current uma), then projects predictions to be **zero-sum**.
- Evaluation: `evaluate_model.py` scores predictors on held-out DB rows (RMSE is the correct scalar metric for estimating the mean).
- UI sandbox: `ev_ui.py` is a Streamlit app that calls `ev_model.estimate_all_values`.

## Definitions & units (be consistent)
- Scores in the DB are **raw points** (e.g. `25000`).
- Model API uses `scores_thousands` (e.g. `[25.0, 25.0, 25.0, 25.0]`).
- EV output is **in thousands**, relative to 25k:
  `EV = (score + uma)/1000 - 25`.
- Uma scheme is Tenhou: `90 / 45 / 0 / -135` (ties broken by lower seat index).

## SQLite schema (rounds)
Created by `data/unzip.py` in `rounds.db`:
- Keys/ids: `round_key` (primary key), `log_id`.
- State: `wind` (`E`/`S`/etc), `round` (kyoku number), `honba`, `riichi` (kyotaku).
- Scores: `s1_start..s4_start`, `s1_final..s4_final`.
- Labels: `s*_y_residual` is `(final_score+final_uma) - (start_score+start_uma)` in **points** (integer).

## Model behavior (what to preserve)
- Features (`ev_model._encode_state_row`): `[wind_id, round, honba_bucket(5+), riichi_bucket(5+), seat, rotated_scores_thousands]`.
- Target (`ev_model.build_training_matrix`): residual in **thousands** over the “no change” baseline.
- `ev_model.estimate_all_values` enforces zero-sum by shifting predicted `(score+uma)/1000` so the four seats sum to `100.0` (because total points are 100k).

## Workflows (typical commands)
- Build DB from zips: `python data/unzip.py` (expects `YYYY.zip` in repo root; writes `data/rounds.db`).
- Train model: `python ev_model.py` (reads `ROUNDS_DB_PATH`, writes `ev_model1.json`).
- Evaluate model: `python evaluate_model.py --db data/rounds.db` (default uses last 10% as validation).
- Run UI: `streamlit run ev_ui.py`.

## Gotchas / conventions
- Default DB location is `data/rounds.db`.
- Keep generated reports, model smoke-test outputs, and other run artifacts inside this repo; do not write project outputs to `C:\tmp`.
- West rounds are skipped for training/eval unless `wind in ("E","S")`.
- For EV correctness: prefer **calibration by EV buckets** as the primary check; RMSE is the proper loss for conditional mean; MAE is diagnostic only.
