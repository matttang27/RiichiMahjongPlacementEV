# ev_model.py

import sqlite3
from pathlib import Path

import numpy as np
import xgboost as xgb

ROUNDS_DB_PATH = "rounds.db"
MODEL_PATH = "ev_model2.json"


# ---------- UMA / scoring ----------

def compute_uma(final_scores, uma_scheme=(90000, 45000, 0, -135000)):
    """
    final_scores: list/tuple of 4 ints in points (e.g. [45000, 25000, ...])
    returns: list of 4 ints in points (e.g. [90000, 0, 45000, -135000])
    """
    s = list(final_scores)
    idxs = list(range(4))
    # higher score is better, break ties by smaller seat index
    order = sorted(idxs, key=lambda i: (-s[i], i))

    uma = [0] * 4
    for rank, seat in enumerate(order):
        uma[seat] = uma_scheme[rank]
    return uma


# ---------- Feature encoding ----------

def _encode_state_row(
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands,
    seat: int,
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
    wind_map = {"E": 0, "S": 1, 0: 0, 1: 1}
    try:
        w = wind_map[wind]
    except KeyError:
        raise ValueError(f"Unsupported wind: {wind!r}. Use 'E' or 'S' or 0/1.")

    h_b = min(int(honba), 5)
    r_b = min(int(riichi), 5)

    if len(scores_thousands) != 4:
        raise ValueError("scores_thousands must have length 4")

    # Rotate so that this seat is position 0
    p = [float(scores_thousands[(seat + k) % 4]) for k in range(4)]

    row = np.array(
        [
            float(w),
            float(round_num),
            float(h_b),
            float(r_b),
            float(seat),
            p[0],
            p[1],
            p[2],
            p[3],
        ],
        dtype=np.float32,
    )
    return row.reshape(1, -1)


# ---------- Dataset building from rounds.db ----------

def build_training_matrix(db_path: str = ROUNDS_DB_PATH, max_rows: int | None = None):
    """
    Read rounds from SQLite and build (X, y):

    Features:
      [wind_id, round, honba_bucket, riichi_bucket, seat,
       s0_thousands, s1_thousands, s2_thousands, s3_thousands]

    Target (for each seat):
      y = (final_score + final_uma)/1000 - (start_score + start_uma)/1000
        = residual over the "no change" baseline, in thousands.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT wind, round, honba, riichi,
               s1_start, s2_start, s3_start, s4_start,
               s1_final, s2_final, s3_final, s4_final
        FROM rounds
        """
    )

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []

    count_rows = 0

    while True:
        row = cur.fetchone()
        if row is None:
            break

        wind = row[0]
        round_num = int(row[1])
        honba = int(row[2])
        riichi = int(row[3])

        # Skip West rounds for simplicity (continuation hands)
        if wind not in ("E", "S"):
            continue

        s_start_pts = list(row[4:8])
        s_final_pts = list(row[8:12])

        # Convert start scores to thousands for features
        scores_thousands = [s / 1000.0 for s in s_start_pts]

        # Uma based on start + final
        start_uma_pts = compute_uma(s_start_pts)
        final_uma_pts = compute_uma(s_final_pts)

        for seat in range(4):
            x = _encode_state_row(
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=scores_thousands,
                seat=seat,
            )

            # Baseline: "no change" EV target in thousands
            baseline_thousands = (
                s_start_pts[seat] + start_uma_pts[seat]
            ) / 1000.0

            # True final target in thousands
            target_thousands = (
                s_final_pts[seat] + final_uma_pts[seat]
            ) / 1000.0

            # Residual over baseline
            y_residual = target_thousands - baseline_thousands

            X_rows.append(x[0])
            y_vals.append(y_residual)

        count_rows += 1
        if max_rows is not None and count_rows >= max_rows:
            break

    conn.close()

    if not X_rows:
        raise RuntimeError("No data loaded from rounds.db. Is the table empty?")

    X = np.stack(X_rows, axis=0)
    y = np.array(y_vals, dtype=np.float32)

    print(f"Built training matrix: X.shape={X.shape}, y.shape={y.shape}")
    return X, y



# ---------- Model training / saving / loading ----------

def train_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor on the given data.
    """
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X, y)
    return model


def save_model(model: xgb.XGBRegressor, path: str = MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"Saved model to {path}")


def load_model(path: str = MODEL_PATH) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
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
) -> float:
    """
    Return EV in thousands for one seat:

        EV = E[ (final_score + final_uma)/1000 - 25 ]

    where model predicts the residual over the baseline:
        residual = target_thousands - baseline_thousands
    """
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
    )

    # Model predicts residual over baseline, in thousands
    residual_thousands = float(model.predict(x)[0])

    # Final predicted target in thousands
    y_thousands = baseline_thousands + residual_thousands

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
):
    """
    Return (EV0, EV1, EV2, EV3) in thousands for all players.

    Model predicts residuals over the "no change" baseline.
    We then:
      - add the baseline back, and
      - recenter so the four predicted (score+uma)/1000 sum to 100,
        ensuring EVs are zero-sum.
    """
    import numpy as np

    # Reconstruct start scores and baseline for all seats
    s_start_pts = [int(round(s * 1000)) for s in scores_thousands]
    start_uma_pts = compute_uma(s_start_pts)
    baseline_thousands = [
        (s_start_pts[i] + start_uma_pts[i]) / 1000.0
        for i in range(4)
    ]

    # Build features for all seats
    X_rows = []
    for seat in range(4):
        x = _encode_state_row(
            wind=wind,
            round_num=round_num,
            honba=honba,
            riichi=riichi,
            scores_thousands=scores_thousands,
            seat=seat,
        )
        X_rows.append(x[0])

    X = np.stack(X_rows, axis=0)  # shape (4, num_features)

    # Model predicts residuals in thousands
    residuals = model.predict(X)  # shape (4,)

    # Add baseline back
    y_thousands = [
        baseline_thousands[i] + float(residuals[i]) for i in range(4)
    ]

    # Enforce sum(y) = 100 exactly (zero-sum EVs)
    total = sum(y_thousands)
    shift = (total - 100.0) / 4.0
    y_adj = [y - shift for y in y_thousands]

    # Convert to EVs (relative to 25k)
    evs = [y - 25.0 for y in y_adj]
    return tuple(evs)


# ---------- Script entrypoint ----------

def main():
    print("Building training matrix from rounds.db...")
    X, y = build_training_matrix(ROUNDS_DB_PATH)

    print("Training model...")
    model = train_model(X, y)

    print("Saving model...")
    save_model(model, MODEL_PATH)

    # Tiny sanity check example (you can delete this later):
    ex_vals = estimate_all_values(
        model,
        wind=1,  # South
        round_num=4,
        honba=0,
        riichi=0,
        scores_thousands=[0.0, 15.0, 35.0, 50.0],
    )
    print("Example S4 EVs [0,15,35,50]:", ex_vals)


if __name__ == "__main__":
    main()
