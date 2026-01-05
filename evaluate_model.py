"""evaluate_model.py

Evaluate EV predictors against held-out rounds from `data/rounds.db`.

Supports either:
    - the default JSON/XGBoost model (via `ev_model.estimate_all_values`), or
    - any Python callable you pass with `--predictor module:callable`.

Predictors must output 4 EVs (one per seat), in *thousands*, relative to 25k:

        EV = (final_score + final_uma)/1000 - 25
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ev_model import compute_uma


# Last 10% of database is for validation by default
DEFAULT_VALIDATE_SPLIT = 0.1
DEFAULT_DB_PATH = "data/rounds.db"
DEFAULT_MODEL_PATH = "ev_model1.json"


class EVPModel:
    """Protocol-like base: model must implement predict(...) and return 4 EVs."""

    def predict(
        self,
        wind: str,
        round_num: int,
        honba: int,
        riichi: int,
        scores_thousands: Sequence[float],
    ) -> Sequence[float]:
        raise NotImplementedError


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


class DefaultJSONModel(EVPModel):
    """Adapter so the existing JSON/XGBoost model works with this evaluator."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        # Lazy import so users can still import this module without xgboost installed.
        from ev_model import load_model

        self._model = load_model(model_path)

    def predict(
        self,
        wind: str,
        round_num: int,
        honba: int,
        riichi: int,
        scores_thousands: Sequence[float],
    ) -> Sequence[float]:
        from ev_model import estimate_all_values

        return estimate_all_values(
            model=self._model,
            wind=wind,
            round_num=int(round_num),
            honba=int(honba),
            riichi=int(riichi),
            scores_thousands=list(scores_thousands),
        )


@dataclass(frozen=True)
class EvalConfig:
    db_path: str
    validate_split: float
    max_rounds: int | None


def _get_round_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rounds")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def evaluate_model_ev(model: EVPModel, cfg: EvalConfig) -> dict:
    if not (0.0 < cfg.validate_split < 1.0):
        raise ValueError("validate_split must be in (0, 1)")

    total_count = _get_round_count(cfg.db_path)
    start_index = int(total_count * (1 - cfg.validate_split))

    conn = sqlite3.connect(cfg.db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final
            FROM rounds
            LIMIT -1 OFFSET ?
            """,
            (start_index,),
        )

        actual_evs: list[float] = []
        model_evs: list[float] = []

        count_rounds = 0
        total_sum_model = 0.0

        while True:
            if cfg.max_rounds is not None and count_rounds >= cfg.max_rounds:
                break

            row = cur.fetchone()
            if row is None:
                break

            wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])
            s_start_pts = list(row[4:8])
            s_final_pts = list(row[8:12])

            final_uma_pts = compute_uma(s_final_pts)

            s_start_th = [s / 1000.0 for s in s_start_pts]

            pred = list(model.predict(wind, rnd, honba, riichi, s_start_th))
            if len(pred) != 4:
                raise ValueError(
                    f"Predictor must return 4 values (one per seat). Got {len(pred)}."
                )

            actual_round: list[float] = []

            for seat in range(4):
                actual = (s_final_pts[seat] + final_uma_pts[seat]) / 1000.0 - 25.0
                actual_round.append(actual)

            actual_evs.extend(actual_round)
            model_evs.extend([float(x) for x in pred])

            total_sum_model += float(sum(pred))

            count_rounds += 1
            if (count_rounds % 1000) == 0:
                print(f"Processed {count_rounds} rounds...")
    finally:
        conn.close()

    actual = np.array(actual_evs, dtype=np.float32)
    model = np.array(model_evs, dtype=np.float32)

    model_mse = _mse(actual, model)
    model_rmse = float(np.sqrt(model_mse))
    model_corr = _safe_corrcoef(actual, model)

    avg_sum_model = total_sum_model / max(count_rounds, 1)

    print("\n=== EV ACCURACY RESULTS ===")
    print(f"Rounds evaluated      : {count_rounds}")
    print(f"Avg sum(model EVs)    : {avg_sum_model:.4f} (should be ~0)")

    print("\n--- MSE / RMSE (thousands of points² / thousands) ---")
    print(f"Model MSE             : {model_mse:.4f}")
    print(f"Model RMSE            : {model_rmse:.4f}")

    print("\n--- Correlation with true EV ---")
    print(f"Model corr            : {model_corr:.4f}")

    return {
        "model_mse": model_mse,
        "model_rmse": model_rmse,
        "model_corr": model_corr,
        "avg_sum_model": avg_sum_model,
        "rounds": count_rounds,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EV predictor on held-out rounds")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite rounds.db")
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
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    cfg = EvalConfig(
        db_path=args.db,
        validate_split=float(args.validate_split),
        max_rounds=args.max_rounds,
    )

    model = DefaultJSONModel(DEFAULT_MODEL_PATH)
    print(f"Using default JSON model: {DEFAULT_MODEL_PATH}")
    return evaluate_model_ev(model, cfg)


if __name__ == "__main__":
    main()
