"""evaluate_model.py

Evaluate EV predictors against held-out rounds from `data/rounds.db`.

Uses the default JSON/XGBoost model (via `ev_model.estimate_all_values`).

Predictions are 4 EVs (one per seat), in *thousands*, relative to 25k:

        EV = (final_score + final_uma)/1000 - 25

Notes on metrics:
    - Calibration by EV buckets is the primary correctness check for EV-as-mean.
    - RMSE is the proper scalar metric for estimating the conditional mean.
    - MAE is reported as a diagnostic only (it targets the median).
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence

import numpy as np

from helper import UMA, compute_uma, _get_round_count
from xgboost_model import estimate_all_values, load_model


# Last 10% of database is for validation by default
DEFAULT_VALIDATE_SPLIT = 0.1
DEFAULT_DB_PATH = "../data/rounds.db"
DEFAULT_MODEL_PATH = "xgboost.json"
DEFAULT_CALIBRATION_BUCKETS = 20

model = load_model(DEFAULT_MODEL_PATH)
# CHANGE YOUR MODEL HERE:
def predict(wind, round_num, honba, riichi, scores_pts) -> tuple[float, float, float, float]:

    scores_div = [s / 1000.0 for s in scores_pts]

    evs = estimate_all_values(
        model=model,
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_div,
    )
    return evs

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))

def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    # Handle constant vectors (np.corrcoef would return nan)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _print_calibration_by_buckets(
    actual: np.ndarray, model: np.ndarray, buckets: int
) -> None:
    if buckets <= 0:
        return
    if actual.size == 0:
        return

    buckets = int(buckets)
    buckets = max(1, buckets)
    buckets = min(buckets, int(actual.size))

    order = np.argsort(model, kind="mergesort")
    actual_sorted = actual[order]
    model_sorted = model[order]

    # Split into approximately-equal-count buckets (quantile buckets).
    splits = np.array_split(np.arange(model_sorted.size), buckets)

    print("\n--- Calibration by predicted-EV buckets (primary check) ---")
    print("bucket\tcount\tmean_pred\tmean_actual\tdiff")
    for idx, s in enumerate(splits, start=1):
        if s.size == 0:
            continue
        mp = float(np.mean(model_sorted[s]))
        ma = float(np.mean(actual_sorted[s]))
        diff = ma - mp
        print(f"{idx:02d}\t{s.size}\t{mp:+.3f}\t\t{ma:+.3f}\t\t{diff:+.3f}")


def evaluate_model_ev(
    *,
    db_path: str,
    model_path: str,
    validate_split: float,
    max_rounds: int | None,
    calibration_buckets: int,
) -> dict:

    if not (0.0 < validate_split < 1.0):
        raise ValueError("validate_split must be in (0, 1)")

    total_count = _get_round_count(db_path)
    start_index = int(total_count * (1 - validate_split))

    model_obj = load_model(model_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final,
                     s1_place, s2_place, s3_place, s4_place
            FROM rounds
            ORDER BY rowid
            LIMIT -1 OFFSET ?
            """,
            (start_index,),
        )

        actual_evs: list[float] = []
        model_evs: list[float] = []

        count_rounds = 0
        skipped_rounds = 0
        total_sum_model = 0.0
        abs_sum_model_max = 0.0
        sum_model_squares = 0.0

        for row in cur:
            if max_rounds is not None and count_rounds >= max_rounds:
                break

            wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])

            # Match repo convention: evaluate only East/South games.
            if wind not in ("E", "S"):
                skipped_rounds += 1
                continue

            s_start_pts = row[4:8]
            s_final_pts = row[8:12]
            s_final_places = row[12:16]
            final_uma = [UMA[place - 1] for place in s_final_places]
            pred = predict(wind,rnd,honba,riichi,s_start_pts)
            if len(pred) != 4:
                raise ValueError(f"Model must return 4 EVs. Got {len(pred)}")

            actual_evs.extend(
                [
                    (s_final_pts[i] + final_uma[i]) / 1000.0 - 25.0
                    for i in range(4)
                ]
            )
            model_evs.extend([float(x) for x in pred])

            round_sum = float(sum(pred))
            total_sum_model += round_sum
            abs_sum_model_max = max(abs_sum_model_max, abs(round_sum))
            sum_model_squares += round_sum * round_sum

            count_rounds += 1
            if (count_rounds % 1000) == 0:
                print(f"Processed {count_rounds} rounds...")
    finally:
        conn.close()

    actual = np.array(actual_evs, dtype=np.float32)
    model = np.array(model_evs, dtype=np.float32)

    if count_rounds == 0:
        print("\n=== EV ACCURACY RESULTS ===")
        print("Rounds evaluated      : 0")
        print(f"Rounds skipped (W+)   : {skipped_rounds}")
        print("No rounds to evaluate (check DB / validate split).")
        return {
            "model_mse": float("nan"),
            "model_rmse": float("nan"),
            "model_mae": float("nan"),
            "model_corr": float("nan"),
            "avg_sum_model": float("nan"),
            "sum_model_rmse": float("nan"),
            "max_abs_sum_model": float("nan"),
            "rounds": 0,
            "skipped_rounds": skipped_rounds,
        }

    model_mse = _mse(actual, model)
    model_rmse = float(np.sqrt(model_mse))
    model_mae = float(np.mean(np.abs(actual - model)))
    model_corr = _safe_corrcoef(actual, model)

    avg_sum_model = total_sum_model / max(count_rounds, 1)
    sum_model_rmse = float(np.sqrt(sum_model_squares / max(count_rounds, 1)))

    print("\n=== EV ACCURACY RESULTS ===")
    print(f"Rounds evaluated      : {count_rounds}")
    print(f"Rounds skipped (W+)   : {skipped_rounds}")
    print(f"Avg sum(model EVs)    : {avg_sum_model:.4f} (should be ~0)")
    print(f"RMSE(sum model EVs)   : {sum_model_rmse:.4f} (should be small)")
    print(f"Max |sum model EVs|   : {abs_sum_model_max:.4f}")

    print("\n--- MSE / RMSE (thousands of points² / thousands) ---")
    print(f"Model MSE             : {model_mse:.4f}")
    print(f"Model RMSE            : {model_rmse:.4f}")
    print(f"Model MAE (diagnostic): {model_mae:.4f}")

    print("\n--- Correlation with true EV ---")
    print(f"Model corr            : {model_corr:.4f}")

    _print_calibration_by_buckets(
        actual=actual, model=model, buckets=int(calibration_buckets)
    )

    return {
        "model_mse": model_mse,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "model_corr": model_corr,
        "avg_sum_model": avg_sum_model,
        "sum_model_rmse": sum_model_rmse,
        "max_abs_sum_model": abs_sum_model_max,
        "rounds": count_rounds,
        "skipped_rounds": skipped_rounds,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EV predictor on held-out rounds")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite rounds.db")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to default JSON model",
    )
    p.add_argument(
        "--validate-split",
        type=float,
        default=DEFAULT_VALIDATE_SPLIT,
        help="Fraction of rows used for validation (uses last N%%)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional cap on number of validation rounds",
    )
    p.add_argument(
        "--calibration-buckets",
        type=int,
        default=DEFAULT_CALIBRATION_BUCKETS,
        help="Number of quantile buckets for calibration table (0 to disable)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    print(f"Using default JSON model: {args.model}")
    return evaluate_model_ev(
        db_path=str(args.db),
        model_path=str(args.model),
        validate_split=float(args.validate_split),
        max_rounds=args.max_rounds,
        calibration_buckets=int(args.calibration_buckets),
    )


if __name__ == "__main__":
    main()
