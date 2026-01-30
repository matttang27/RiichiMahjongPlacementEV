import sqlite3

import numpy as np

DB_PATH = "data/rounds.db"

LIMIT = 5000
K = 2000
WEIGHTS = [1000,1000,300,1000,1,1,1,1]

_X8: np.ndarray | None = None       # (N, 8) float32: wind, round, honba, riichi, s1_start..s4_start
_Y4: np.ndarray | None = None       # (N, 4) float32: s1_final_s..s4_final_s

def _ensure_loaded(db_path: str = DB_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load (wind, round, honba, riichi, s*_start) and (s*_final_s) into NumPy once."""
    global _X8, _Y4
    if _X8 is not None:
        return _X8, _Y4  # type: ignore[return-value]

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rounds")
        n = int(cur.fetchone()[0])
        if n <= 0:
            raise RuntimeError("rounds table is empty")

        X8 = np.empty((n, 8), dtype=np.float32)
        Y4 = np.empty((n, 4), dtype=np.float32)

        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final_s, s2_final_s, s3_final_s, s4_final_s
            FROM rounds
            """
        )

        i = 0
        fetch = cur.fetchmany
        while True:
            rows = fetch(100_000)
            if not rows:
                break
            for row in rows:
                # Keep representation exactly as stored (no extra encoding).
                X8[i, 0] = float(row[0])
                X8[i, 1] = float(row[1])
                X8[i, 2] = float(row[2])
                X8[i, 3] = float(row[3])
                X8[i, 4] = float(row[4])
                X8[i, 5] = float(row[5])
                X8[i, 6] = float(row[6])
                X8[i, 7] = float(row[7])

                Y4[i, 0] = float(row[8])
                Y4[i, 1] = float(row[9])
                Y4[i, 2] = float(row[10])
                Y4[i, 3] = float(row[11])
                i += 1

        if i != n:
            X8 = X8[:i, :]
            Y4 = Y4[:i, :]

        _X8 = X8
        _Y4 = Y4
        return X8, Y4
    finally:
        conn.close()


# a contains wind, round, honba, riichi, s1_start, s2_start, s3_start, s4_start
def distance(a, b, weights):
    return sum(weights[i] * (a[i] - b[i]) ** 2 for i in range(8)) ** 0.5

def predict(wind,round,honba,riichi,scores):
    X8, Y4 = _ensure_loaded()
    base = [wind, round, honba, riichi] + scores

    # LIMIT prefilter (same semantics as the previous SQL window).
    s0, s1, s2, s3 = int(scores[0]), int(scores[1]), int(scores[2]), int(scores[3])
    start_scores = X8[:, 4:8]  # float32 but values are integral
    mask = (
        (start_scores[:, 0] > s0 - LIMIT)
        & (start_scores[:, 0] < s0 + LIMIT)
        & (start_scores[:, 1] > s1 - LIMIT)
        & (start_scores[:, 1] < s1 + LIMIT)
        & (start_scores[:, 2] > s2 - LIMIT)
        & (start_scores[:, 2] < s2 + LIMIT)
        & (start_scores[:, 3] > s3 - LIMIT)
        & (start_scores[:, 3] < s3 + LIMIT)
    )

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    # Vectorized weighted distance on all 8 features (same as distance()).
    base8 = np.asarray(base, dtype=np.float32)
    d = X8[idx, :] - base8[np.newaxis, :]
    w = np.asarray(WEIGHTS, dtype=np.float32)
    dist2 = np.sum(w[np.newaxis, :] * (d * d), axis=1)  # squared distance; sqrt not needed

    k = int(min(max(K, 1), dist2.size))
    nn_local = np.argpartition(dist2, kth=k - 1)[:k]
    nn_idx = idx[nn_local]

    avg_final = np.mean(Y4[nn_idx, :], axis=0)
    return [float(avg_final[0]), float(avg_final[1]), float(avg_final[2]), float(avg_final[3])]

print(predict(1,0,0,0,[23000,25000,25000,27000]))
                
