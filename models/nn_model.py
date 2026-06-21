"""Small joint-output neural net EV model.

This model predicts all four seats from one game-state row, then recenters the
four outputs so the predicted EVs are zero-sum.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .features import final_evs_thousands, is_supported_wind, normalize_wind_id
    from .helper import _get_round_count
except ImportError:  # Allows `python models/nn_model.py`.
    from features import final_evs_thousands, is_supported_wind, normalize_wind_id
    from helper import _get_round_count


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUNDS_DB_PATH = REPO_ROOT / "data" / "rounds.db"
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "sklearn_nn" / "joint_v1"
MODEL_PATH = EXPERIMENT_DIR / "model.joblib"
EVALUATION_PATH = EXPERIMENT_DIR / "evaluation.txt"
SUMMARY_PATH = EXPERIMENT_DIR / "summary.json"
MONOTONIC_PATH = EXPERIMENT_DIR / "monotonic_checks.txt"

FEATURE_VERSION = "nn_joint_v1"
TARGET_MODE = "joint_direct_ev"

JOINT_FEATURE_NAMES = [
    "wind_id",
    "round",
    "honba_bucket",
    "riichi_bucket",
    "round_index",
    "hands_until_normal_end",
    "dealer_seat",
    "is_east_round",
    "is_south_round",
    "is_all_last",
    "score_total_th",
    "score_mean_th",
    "score_0_th",
    "score_1_th",
    "score_2_th",
    "score_3_th",
    "score_centered_0_th",
    "score_centered_1_th",
    "score_centered_2_th",
    "score_centered_3_th",
    "gap_to_leader_0_th",
    "gap_to_leader_1_th",
    "gap_to_leader_2_th",
    "gap_to_leader_3_th",
    "gap_to_last_0_th",
    "gap_to_last_1_th",
    "gap_to_last_2_th",
    "gap_to_last_3_th",
    "is_dealer_0",
    "is_dealer_1",
    "is_dealer_2",
    "is_dealer_3",
    "gap_0_1_th",
    "gap_0_2_th",
    "gap_0_3_th",
    "gap_1_2_th",
    "gap_1_3_th",
    "gap_2_3_th",
]


def get_feature_names() -> list[str]:
    return list(JOINT_FEATURE_NAMES)


def encode_joint_state(
    *,
    wind: str | int,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands: Sequence[int] | Sequence[float],
) -> np.ndarray:
    if len(scores_thousands) != 4:
        raise ValueError("scores_thousands must have length 4")

    wind_id = normalize_wind_id(wind)
    round_num = int(round_num)
    honba_bucket = min(int(honba), 5)
    riichi_bucket = min(int(riichi), 5)
    scores = [float(s) for s in scores_thousands]

    score_total = float(sum(scores))
    score_mean = score_total / 4.0
    leader = max(scores)
    last = min(scores)
    round_index = wind_id * 4 + round_num
    dealer_seat = (round_num - 1) % 4

    values = [
        float(wind_id),
        float(round_num),
        float(honba_bucket),
        float(riichi_bucket),
        float(round_index),
        float(max(0, 8 - round_index)),
        float(dealer_seat),
        1.0 if wind_id == 0 else 0.0,
        1.0 if wind_id == 1 else 0.0,
        1.0 if wind_id == 1 and round_num == 4 else 0.0,
        score_total,
        score_mean,
        *scores,
        *[s - score_mean for s in scores],
        *[s - leader for s in scores],
        *[s - last for s in scores],
        *[1.0 if seat == dealer_seat else 0.0 for seat in range(4)],
        scores[0] - scores[1],
        scores[0] - scores[2],
        scores[0] - scores[3],
        scores[1] - scores[2],
        scores[1] - scores[3],
        scores[2] - scores[3],
    ]
    return np.asarray(values, dtype=np.float32)


def _training_scan_limit(
    *,
    db_path: str | Path,
    train_fraction: float,
    max_rows: int | None,
) -> int:
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")

    total = _get_round_count(str(db_path))
    scan_limit = max(1, int(total * train_fraction))
    if max_rows is not None:
        scan_limit = min(scan_limit, int(max_rows))
    return scan_limit


def build_training_matrix(
    db_path: str | Path = ROUNDS_DB_PATH,
    *,
    train_fraction: float = 0.9,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    scan_limit = _training_scan_limit(
        db_path=db_path,
        train_fraction=train_fraction,
        max_rows=max_rows,
    )

    X = np.empty((scan_limit, len(JOINT_FEATURE_NAMES)), dtype=np.float32)
    y = np.empty((scan_limit, 4), dtype=np.float32)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final
            FROM rounds
            LIMIT ?
            """,
            (scan_limit,),
        )

        out_idx = 0
        while True:
            row = cur.fetchone()
            if row is None:
                break

            wind = row[0]
            if not is_supported_wind(wind):
                continue

            start_scores_pts = list(row[4:8])
            final_scores_pts = list(row[8:12])
            X[out_idx, :] = encode_joint_state(
                wind=wind,
                round_num=int(row[1]),
                honba=int(row[2]),
                riichi=int(row[3]),
                scores_thousands=[s / 1000.0 for s in start_scores_pts],
            )
            y[out_idx, :] = np.asarray(final_evs_thousands(final_scores_pts), dtype=np.float32)
            out_idx += 1

            if out_idx >= scan_limit:
                break
    finally:
        conn.close()

    if out_idx == 0:
        raise RuntimeError("No supported East/South rounds loaded from rounds.db.")

    X = X[:out_idx, :]
    y = y[:out_idx, :]
    print(
        f"Built NN training matrix: X.shape={X.shape}, y.shape={y.shape}, "
        f"feature_version={FEATURE_VERSION}, target_mode={TARGET_MODE}"
    )
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_layers: tuple[int, ...] = (128, 64),
    max_iter: int = 30,
    random_state: int = 42,
) -> TransformedTargetRegressor:
    regressor = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    batch_size=2048,
                    learning_rate_init=0.001,
                    alpha=0.0001,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=random_state,
                    verbose=True,
                ),
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )
    model.fit(X, y)
    return model


def save_model(model, path: str | Path = MODEL_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "target_mode": TARGET_MODE,
            "feature_names": JOINT_FEATURE_NAMES,
        },
        p,
    )
    print(f"Saved NN model to {p}")


def load_model(path: str | Path = MODEL_PATH):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def estimate_all_values(
    model,
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands: Sequence[int] | Sequence[float],
) -> tuple[float, float, float, float]:
    X = encode_joint_state(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
    ).reshape(1, -1)
    pred = np.asarray(model.predict(X)[0], dtype=np.float64)
    pred = pred - float(np.mean(pred))
    return tuple(float(x) for x in pred)


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("hidden layers must be comma-separated integers") from e
    if not layers or any(v <= 0 for v in layers):
        raise argparse.ArgumentTypeError("hidden layers must be positive integers")
    return layers


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the joint-output NN EV model")
    p.add_argument("--db", default=str(ROUNDS_DB_PATH), help="Path to rounds.db")
    p.add_argument("--model", default=str(MODEL_PATH), help="Where to save the NN joblib model")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of DB rows to scan for training; default excludes the last 10%.",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests")
    p.add_argument("--max-iter", type=int, default=30, help="MLP max training epochs")
    p.add_argument(
        "--hidden-layers",
        type=_parse_hidden_layers,
        default=(128, 64),
        help="Comma-separated hidden layer sizes, e.g. 128,64",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    X, y = build_training_matrix(
        args.db,
        train_fraction=float(args.train_fraction),
        max_rows=args.max_rows,
    )
    model = train_model(
        X,
        y,
        hidden_layers=args.hidden_layers,
        max_iter=int(args.max_iter),
    )
    save_model(model, args.model)

    ex_vals = estimate_all_values(
        model,
        wind="S",
        round_num=4,
        honba=0,
        riichi=0,
        scores_thousands=[40.0, 30.0, 14.4, 15.6],
    )
    print("Example S4 EVs [40,30,14.4,15.6]:", ex_vals)


if __name__ == "__main__":
    main()
