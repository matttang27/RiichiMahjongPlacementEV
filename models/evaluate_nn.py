"""Evaluate the joint-output NN EV model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from .evaluate_model import (
        DEFAULT_CALIBRATION_BUCKETS,
        DEFAULT_DB_PATH,
        DEFAULT_ORACLE_MIN_ROUNDS,
        DEFAULT_ORACLE_SCORE_BUCKET,
        DEFAULT_VALIDATE_SPLIT,
        evaluate_model_ev,
    )
    from .nn_model import (
        EVALUATION_PATH,
        FEATURE_VERSION,
        MODEL_PATH,
        SUMMARY_PATH,
        TARGET_MODE,
        estimate_all_values,
        load_model,
    )
except ImportError:  # Allows `python models/evaluate_nn.py`.
    from evaluate_model import (
        DEFAULT_CALIBRATION_BUCKETS,
        DEFAULT_DB_PATH,
        DEFAULT_ORACLE_MIN_ROUNDS,
        DEFAULT_ORACLE_SCORE_BUCKET,
        DEFAULT_VALIDATE_SPLIT,
        evaluate_model_ev,
    )
    from nn_model import (
        EVALUATION_PATH,
        FEATURE_VERSION,
        MODEL_PATH,
        SUMMARY_PATH,
        TARGET_MODE,
        estimate_all_values,
        load_model,
    )


def _predict(model, wind, round_num, honba, riichi, scores_pts):
    return estimate_all_values(
        model,
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=[s / 1000.0 for s in scores_pts],
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the joint-output NN EV model")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite rounds.db")
    p.add_argument("--model", default=str(MODEL_PATH), help="Path to NN joblib model")
    p.add_argument(
        "--validate-split",
        type=float,
        default=DEFAULT_VALIDATE_SPLIT,
        help="Evaluate on the last fraction of DB rows",
    )
    p.add_argument("--max-rounds", type=int, default=None, help="Optional cap for quick evaluation")
    p.add_argument(
        "--calibration-buckets",
        type=int,
        default=DEFAULT_CALIBRATION_BUCKETS,
        help="Number of predicted-EV quantile buckets",
    )
    p.add_argument(
        "--oracle-min-rounds",
        type=int,
        default=DEFAULT_ORACLE_MIN_ROUNDS,
        help="Minimum rounds in an exact/coarse state group for oracle diagnostics",
    )
    p.add_argument(
        "--oracle-score-bucket",
        type=int,
        default=DEFAULT_ORACLE_SCORE_BUCKET,
        help="Score bucket size in points for coarse oracle diagnostics",
    )
    p.add_argument("--out", default=str(EVALUATION_PATH), help="Where to write the full text report")
    p.add_argument("--summary-out", default=str(SUMMARY_PATH), help="Where to write compact JSON metrics")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    print(f"Using NN model: {args.model}")
    return evaluate_model_ev(
        db_path=str(args.db),
        model_path=str(args.model),
        feature_version=FEATURE_VERSION,
        target_mode=TARGET_MODE,
        validate_split=float(args.validate_split),
        max_rounds=args.max_rounds,
        calibration_buckets=int(args.calibration_buckets),
        out_path=str(args.out),
        summary_out_path=str(args.summary_out),
        oracle_min_rounds=int(args.oracle_min_rounds),
        oracle_score_bucket=int(args.oracle_score_bucket),
        model_loader=load_model,
        predictor=_predict,
    )


if __name__ == "__main__":
    main()
