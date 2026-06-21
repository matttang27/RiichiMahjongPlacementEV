"""XGBoost EV model training and inference."""

import argparse
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUNDS_DB_PATH = REPO_ROOT / "data" / "rounds.db"
# Keep model path relative to this module so it works in deployments.
MODEL_PATH = Path(__file__).resolve().parent / "xgboost.json"
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
SUPPORTED_TARGET_MODES = ("residual_uma", "direct_ev")
DEFAULT_TARGET_MODE_BY_FEATURE = {
    "legacy": "residual_uma",
    "v1": "residual_uma",
    "v2": "direct_ev",
}

try:
    from .features import (
        baseline_targets_thousands,
        encode_state_row,
        encode_state_rows,
        final_evs_thousands,
        feature_count,
        get_feature_names,
        is_supported_wind,
    )
    from .helper import compute_uma
except ImportError:  # Allows `python models/xgboost_model.py`.
    from features import (
        baseline_targets_thousands,
        encode_state_row,
        encode_state_rows,
        final_evs_thousands,
        feature_count,
        get_feature_names,
        is_supported_wind,
    )
    from helper import compute_uma


# ---------- Feature encoding ----------

def _encode_state_row(
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands,
    seat: int,
    feature_version: str = "legacy",
) -> np.ndarray:
    """
    Encode a single game state as features.

    wind: 'E'/'S' or 0/1
    round_num: 1..4
    honba, riichi: ints (we bucket both at 5: 5+ -> 5)
    scores_thousands: [s0, s1, s2, s3] in *thousands*
        e.g. [25.0, 25.0, 25.0, 25.0] for all 25000
    seat: whose perspective (0..3)

    Returns shape (1, num_features).
    """
    return encode_state_row(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
        seat=seat,
        feature_version=feature_version,
    ).reshape(1, -1)


def default_model_path_for_features(feature_version: str) -> Path:
    get_feature_names(feature_version)
    if feature_version == "legacy":
        return MODEL_PATH
    return EXPERIMENTS_DIR / "xgboost" / feature_version / "model.json"


def default_evaluation_path_for_features(feature_version: str) -> Path:
    get_feature_names(feature_version)
    if feature_version == "legacy":
        return Path(__file__).resolve().parent / "evaluation_current.txt"
    return EXPERIMENTS_DIR / "xgboost" / feature_version / "evaluation.txt"


def default_summary_path_for_features(feature_version: str) -> Path:
    get_feature_names(feature_version)
    if feature_version == "legacy":
        return Path(__file__).resolve().parent / "evaluation_current_summary.json"
    return EXPERIMENTS_DIR / "xgboost" / feature_version / "summary.json"


def default_target_mode_for_features(feature_version: str) -> str:
    get_feature_names(feature_version)
    return DEFAULT_TARGET_MODE_BY_FEATURE[feature_version]


def validate_target_mode(target_mode: str) -> str:
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(
            f"Unsupported target mode: {target_mode!r}. "
            f"Use one of {SUPPORTED_TARGET_MODES}."
        )
    return target_mode


def monotone_constraints_for_features(feature_version: str) -> tuple[int, ...] | None:
    if feature_version != "v2":
        return None

    positive_features = {
        "rot_score_0_th",
        "self_score_th",
        "gap_to_rel1_th",
        "gap_to_rel2_th",
        "gap_to_rel3_th",
        "self_vs_max_other_th",
        "self_vs_min_other_th",
    }
    negative_features = {
        "rot_score_1_th",
        "rot_score_2_th",
        "rot_score_3_th",
    }

    constraints = []
    for name in get_feature_names(feature_version):
        if name in positive_features:
            constraints.append(1)
        elif name in negative_features:
            constraints.append(-1)
        else:
            constraints.append(0)
    return tuple(constraints)


def _count_rounds_to_scan(db_path: str | Path, max_rows: int | None) -> int:
    if max_rows is not None:
        return int(max_rows)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rounds")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def _resize_training_arrays(
    X: np.ndarray,
    y: np.ndarray,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    new_rows = max(min_rows, int(X.shape[0] * 1.5), X.shape[0] + 4096)
    X_new = np.empty((new_rows, X.shape[1]), dtype=np.float32)
    y_new = np.empty((new_rows,), dtype=np.float32)
    X_new[: X.shape[0], :] = X
    y_new[: y.shape[0]] = y
    return X_new, y_new


def _store_training_round(
    *,
    X: np.ndarray,
    y: np.ndarray,
    out_idx: int,
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    start_scores_pts: Sequence[int] | Sequence[float],
    final_scores_pts: Sequence[int] | Sequence[float],
    feature_version: str,
    target_mode: str,
) -> int:
    scores_thousands = [s / 1000.0 for s in start_scores_pts]
    X_rows = encode_state_rows(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
        feature_version=feature_version,
    )

    if target_mode == "residual_uma":
        baseline_thousands = baseline_targets_thousands(start_scores_pts)
        final_uma_pts = compute_uma(final_scores_pts)
        for seat in range(4):
            target_thousands = (
                float(final_scores_pts[seat]) + float(final_uma_pts[seat])
            ) / 1000.0
            X[out_idx, :] = X_rows[seat]
            y[out_idx] = target_thousands - baseline_thousands[seat]
            out_idx += 1
        return out_idx

    if target_mode == "direct_ev":
        final_evs = final_evs_thousands(final_scores_pts)
        for seat in range(4):
            X[out_idx, :] = X_rows[seat]
            y[out_idx] = final_evs[seat]
            out_idx += 1
        return out_idx

    validate_target_mode(target_mode)
    return out_idx


# ---------- Dataset building from rounds.db ----------

def build_training_matrix(
    db_path: str | Path = ROUNDS_DB_PATH,
    max_rows: int | None = None,
    feature_version: str = "legacy",
    target_mode: str | None = None,
):
    """
    Read rounds from SQLite and build (X, y):

    Features:
      [wind_id, round, honba_bucket, riichi_bucket, seat,
       s0_thousands, s1_thousands, s2_thousands, s3_thousands]

    Target (for each seat):
      y = (final_score + final_uma)/1000 - (start_score + start_uma)/1000
        = residual over the "no change" baseline, in thousands.
    """
    if target_mode is None:
        target_mode = default_target_mode_for_features(feature_version)
    validate_target_mode(target_mode)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final
            FROM rounds
            """
        )

        n_features = feature_count(feature_version)
        initial_round_capacity = max(_count_rounds_to_scan(db_path, max_rows), 1)
        X = np.empty((initial_round_capacity * 4, n_features), dtype=np.float32)
        y = np.empty((initial_round_capacity * 4,), dtype=np.float32)

        count_rows = 0
        out_idx = 0

        while True:
            row = cur.fetchone()
            if row is None:
                break

            wind = row[0]
            round_num = int(row[1])
            honba = int(row[2])
            riichi = int(row[3])

            # Skip West rounds for simplicity (continuation hands)
            if not is_supported_wind(wind):
                continue

            s_start_pts = list(row[4:8])
            s_final_pts = list(row[8:12])

            if out_idx + 4 > X.shape[0]:
                X, y = _resize_training_arrays(X, y, out_idx + 4)

            out_idx = _store_training_round(
                X=X,
                y=y,
                out_idx=out_idx,
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                start_scores_pts=s_start_pts,
                final_scores_pts=s_final_pts,
                feature_version=feature_version,
                target_mode=target_mode,
            )

            count_rows += 1
            if max_rows is not None and count_rows >= max_rows:
                break
    finally:
        conn.close()

    if out_idx == 0:
        raise RuntimeError("No data loaded from rounds.db. Is the table empty?")

    X = X[:out_idx, :]
    y = y[:out_idx]

    print(
        f"Built training matrix: X.shape={X.shape}, y.shape={y.shape}, "
        f"feature_version={feature_version}, target_mode={target_mode}"
    )
    return X, y



# ---------- Model training / saving / loading ----------

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_version: str = "legacy",
    target_mode: str | None = None,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor on the given data.
    """
    if target_mode is None:
        target_mode = default_target_mode_for_features(feature_version)
    validate_target_mode(target_mode)

    constraints = monotone_constraints_for_features(feature_version)
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        monotone_constraints=constraints,
    )

    model.fit(X, y)
    model.get_booster().set_attr(
        feature_version=feature_version,
        target_mode=target_mode,
    )
    return model


def save_model(model: xgb.XGBRegressor, path: str | Path = MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"Saved model to {path}")


def load_model(path: str | Path = MODEL_PATH) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


# ---------- EV estimation API ----------

def estimate_value_for_seat(
    model: xgb.XGBRegressor,
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands,
    seat: int,
    feature_version: str = "legacy",
    target_mode: str | None = None,
) -> float:
    """
    Return EV in thousands for one seat:

        EV = E[ (final_score + final_uma)/1000 - 25 ]

    where model predicts the residual over the baseline:
        residual = target_thousands - baseline_thousands
    """
    if target_mode is None:
        target_mode = default_target_mode_for_features(feature_version)
    validate_target_mode(target_mode)

    # Reconstruct start scores in points from thousands
    s_start_pts = [int(round(s * 1000)) for s in scores_thousands]

    # Baseline EV (in thousands) from "no change"
    start_uma_pts = compute_uma(s_start_pts)
    baseline_thousands = (
        s_start_pts[seat] + start_uma_pts[seat]
    ) / 1000.0

    # Features for this seat
    x = _encode_state_row(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
        seat=seat,
        feature_version=feature_version,
    )

    # Model predicts residual over baseline, in thousands
    prediction = float(model.predict(x)[0])

    if target_mode == "direct_ev":
        return prediction

    # Final predicted target in thousands
    y_thousands = baseline_thousands + prediction

    # EV relative to 25k
    value = y_thousands - 25.0
    return value


def estimate_all_values(
    model: xgb.XGBRegressor,
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands,
    feature_version: str = "legacy",
    target_mode: str | None = None,
):
    """
    Return (EV0, EV1, EV2, EV3) in thousands for all players.

    Model predicts residuals over the "no change" baseline.
    We then:
      - add the baseline back, and
      - recenter so the four predicted (score+uma)/1000 sum to 100,
        ensuring EVs are zero-sum.
    """
    if target_mode is None:
        target_mode = default_target_mode_for_features(feature_version)
    validate_target_mode(target_mode)

    # Reconstruct start scores and baseline for all seats
    s_start_pts = [int(round(s * 1000)) for s in scores_thousands]
    start_uma_pts = compute_uma(s_start_pts)
    baseline_thousands = [
        (s_start_pts[i] + start_uma_pts[i]) / 1000.0
        for i in range(4)
    ]

    X = encode_state_rows(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
        feature_version=feature_version,
    )

    predictions = model.predict(X)  # shape (4,)

    if target_mode == "direct_ev":
        evs_raw = [float(predictions[i]) for i in range(4)]
        shift = sum(evs_raw) / 4.0
        return tuple(ev - shift for ev in evs_raw)

    # Add baseline back for residual-over-current-uma models.
    y_thousands = [
        baseline_thousands[i] + float(predictions[i]) for i in range(4)
    ]

    # Enforce sum(y) = 100 exactly (zero-sum EVs)
    total = sum(y_thousands)
    shift = (total - 100.0) / 4.0
    y_adj = [y - shift for y in y_thousands]

    # Convert to EVs (relative to 25k)
    evs = [y - 25.0 for y in y_adj]
    return tuple(evs)


# ---------- Script entrypoint ----------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the XGBoost EV model")
    p.add_argument("--db", default=str(ROUNDS_DB_PATH), help="Path to rounds.db")
    p.add_argument(
        "--features",
        default="legacy",
        choices=["legacy", "v1", "v2"],
        help="Feature version to train",
    )
    p.add_argument(
        "--target-mode",
        default=None,
        choices=list(SUPPORTED_TARGET_MODES),
        help="Training target. Defaults by feature version.",
    )
    p.add_argument("--model", default=None, help="Where to save the XGBoost JSON model")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick training smoke tests")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)
    model_path = Path(args.model) if args.model is not None else default_model_path_for_features(args.features)
    target_mode = args.target_mode or default_target_mode_for_features(args.features)

    print(f"Building training matrix from {args.db}...")
    X, y = build_training_matrix(
        args.db,
        max_rows=args.max_rows,
        feature_version=args.features,
        target_mode=target_mode,
    )

    print("Training model...")
    model = train_model(X, y, feature_version=args.features, target_mode=target_mode)

    print("Saving model...")
    save_model(model, model_path)

    # Tiny sanity check example (you can delete this later):
    ex_vals = estimate_all_values(
        model,
        wind=1,  # South
        round_num=4,
        honba=0,
        riichi=0,
        scores_thousands=[0.0, 15.0, 35.0, 50.0],
        feature_version=args.features,
        target_mode=target_mode,
    )
    print("Example S4 EVs [0,15,35,50]:", ex_vals)


if __name__ == "__main__":
    main()
