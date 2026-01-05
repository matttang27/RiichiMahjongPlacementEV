"""models/knn_model.py

KNN EV predictor.

Goal: estimate per-seat EV (in thousands, relative to 25k) from a round-start state.

Distance: weighted Euclidean on (wind, round, honba, riichi) with *customizable* weights,
plus unweighted Euclidean distance on the 4 start scores (in thousands).

This intentionally matches the repo conventions:
- Skip West (continuation) rounds: only wind in ("E", "S").
- Bucket honba/riichi at 5 (5+).
- Enforce zero-sum EVs by recentering to sum to 0.

Typical usage (from repo root):

    from models.knn_model import KNNModel
    m = KNNModel.from_db("data/rounds.db", k=200, w_wind=5.0, w_round=1.0, w_honba=0.2, w_riichi=0.2)
    evs = m.predict("E", 1, 0, 0, [25000, 25000, 25000, 25000])

"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    # When imported as a package module: models.knn_model
    from .helper import compute_uma
except ImportError:  # pragma: no cover
    # When run directly from within models/
    from helper import compute_uma


def _wind_to_id(wind: str | int) -> int:
    wind_map = {"E": 0, "S": 1, 0: 0, 1: 1}
    try:
        return int(wind_map[wind])
    except KeyError as e:
        raise ValueError(f"Unsupported wind: {wind!r}. Use 'E'/'S' or 0/1.") from e


def _bucket_5(x: int) -> int:
    return min(int(x), 5)


def _encode_state(
    wind: str | int,
    round_num: int,
    honba: int,
    riichi: int,
    scores_pts: Sequence[int] | Sequence[float],
) -> np.ndarray:
    if len(scores_pts) != 4:
        raise ValueError("scores must have length 4")

    # Scores in distance space are in thousands (consistent magnitude).
    scores_th = np.asarray(scores_pts, dtype=np.float32) / 1000.0

    return np.array(
        [
            float(_wind_to_id(wind)),
            float(int(round_num)),
            float(_bucket_5(honba)),
            float(_bucket_5(riichi)),
            float(scores_th[0]),
            float(scores_th[1]),
            float(scores_th[2]),
            float(scores_th[3]),
        ],
        dtype=np.float32,
    )


def _recenter_zero_sum(evs: np.ndarray) -> np.ndarray:
    # EVs are relative to 25k, so true sum should be 0.
    shift = float(np.sum(evs)) / 4.0
    return evs - shift


@dataclass(frozen=True)
class KNNWeights:
    w_wind: float = 1.0
    w_round: float = 1.0
    w_honba: float = 1.0
    w_riichi: float = 1.0

    def __post_init__(self) -> None:
        for name, v in (
            ("w_wind", self.w_wind),
            ("w_round", self.w_round),
            ("w_honba", self.w_honba),
            ("w_riichi", self.w_riichi),
        ):
            if float(v) < 0.0:
                raise ValueError(f"{name} must be >= 0")


class KNNModel:
    """Simple KNN regressor over round-start states.

    Training rows are *round states* (one per kyoku start), with a 4-vector label
    containing per-seat realized EV.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        k: int = 200,
        weights: KNNWeights | None = None,
    ):
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError("X must be shape (N, 8)")
        if Y.ndim != 2 or Y.shape[1] != 4:
            raise ValueError("Y must be shape (N, 4)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("Empty training set")

        self._X = X.astype(np.float32, copy=False)
        self._Y = Y.astype(np.float32, copy=False)
        self._k = int(k)
        self._w = weights or KNNWeights()

    @property
    def n_train(self) -> int:
        return int(self._X.shape[0])

    @classmethod
    def from_db(
        cls,
        db_path: str,
        *,
        k: int = 200,
        w_wind: float = 1.0,
        w_round: float = 1.0,
        w_honba: float = 1.0,
        w_riichi: float = 1.0,
        max_rows: int | None = None,
        validate_split: float = 0.0,
    ) -> "KNNModel":
        """Load training data from `rounds`.

        If `validate_split` > 0, uses the *first* (1 - split) fraction as training,
        leaving the last split for external evaluation.
        """

        if not (0.0 <= validate_split < 1.0):
            raise ValueError("validate_split must be in [0, 1)")

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT wind, round, honba, riichi,
                       s1_start, s2_start, s3_start, s4_start,
                       s1_final, s2_final, s3_final, s4_final
                FROM rounds
                ORDER BY rowid
                """
            )

            X_rows: list[np.ndarray] = []
            Y_rows: list[np.ndarray] = []

            count = 0
            for row in cur:
                wind = row[0]
                if wind not in ("E", "S"):
                    continue

                round_num = int(row[1])
                honba = int(row[2])
                riichi = int(row[3])

                s_start = row[4:8]
                s_final = row[8:12]

                x = _encode_state(wind, round_num, honba, riichi, s_start)

                final_uma_pts = compute_uma(s_final)
                y = np.array(
                    [
                        (s_final[i] + final_uma_pts[i]) / 1000.0 - 25.0
                        for i in range(4)
                    ],
                    dtype=np.float32,
                )

                X_rows.append(x)
                Y_rows.append(y)

                count += 1
                if max_rows is not None and count >= max_rows:
                    break
        finally:
            conn.close()

        if not X_rows:
            raise RuntimeError("No training rows loaded (is rounds.db present and non-empty?)")

        X = np.stack(X_rows, axis=0)
        Y = np.stack(Y_rows, axis=0)

        # Optionally hold out the tail (by row order) for validation.
        if validate_split > 0.0:
            n = X.shape[0]
            n_train = max(1, int(n * (1.0 - validate_split)))
            X = X[:n_train]
            Y = Y[:n_train]

        return cls(
            X,
            Y,
            k=k,
            weights=KNNWeights(
                w_wind=float(w_wind),
                w_round=float(w_round),
                w_honba=float(w_honba),
                w_riichi=float(w_riichi),
            ),
        )

    def _dist2(self, q: np.ndarray) -> np.ndarray:
        # Weighted Euclidean on (wind, round, honba, riichi) + unweighted on scores.
        d = self._X - q[np.newaxis, :]

        # Apply weights to the first four features.
        d0 = d[:, 0] * np.sqrt(self._w.w_wind)
        d1 = d[:, 1] * np.sqrt(self._w.w_round)
        d2 = d[:, 2] * np.sqrt(self._w.w_honba)
        d3 = d[:, 3] * np.sqrt(self._w.w_riichi)

        # Scores are in thousands and unweighted.
        ds = d[:, 4:8]

        return (d0 * d0) + (d1 * d1) + (d2 * d2) + (d3 * d3) + np.sum(ds * ds, axis=1)

    def predict(
        self,
        wind: str | int,
        round_num: int,
        honba: int,
        riichi: int,
        scores_pts: Sequence[int] | Sequence[float],
    ) -> tuple[float, float, float, float]:
        """Predict per-seat EVs (thousands, relative to 25k)."""

        q = _encode_state(wind, round_num, honba, riichi, scores_pts)
        dist2 = self._dist2(q)

        k = min(max(int(self._k), 1), self.n_train)
        nn_idx = np.argpartition(dist2, kth=k - 1)[:k]

        ev = np.mean(self._Y[nn_idx, :], axis=0)
        ev = _recenter_zero_sum(ev)
        return (float(ev[0]), float(ev[1]), float(ev[2]), float(ev[3]))


def build_knn_predictor(
    *,
    db_path: str = "data/rounds.db",
    k: int = 200,
    w_wind: float = 1.0,
    w_round: float = 1.0,
    w_honba: float = 1.0,
    w_riichi: float = 1.0,
    max_rows: int | None = None,
    validate_split: float = 0.0,
) -> KNNModel:
    """Convenience wrapper."""

    return KNNModel.from_db(
        db_path,
        k=k,
        w_wind=w_wind,
        w_round=w_round,
        w_honba=w_honba,
        w_riichi=w_riichi,
        max_rows=max_rows,
        validate_split=validate_split,
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="KNN EV predictor quick demo")
    p.add_argument("--db", default="data/rounds.db")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--w-wind", type=float, default=1.0)
    p.add_argument("--w-round", type=float, default=1.0)
    p.add_argument("--w-honba", type=float, default=1.0)
    p.add_argument("--w-riichi", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args(argv)

    m = build_knn_predictor(
        db_path=args.db,
        k=args.k,
        w_wind=args.w_wind,
        w_round=args.w_round,
        w_honba=args.w_honba,
        w_riichi=args.w_riichi,
        max_rows=args.max_rows,
    )

    evs = m.predict("E", 1, 0, 0, [25000, 25000, 25000, 25000])
    print(f"Loaded {m.n_train} training rows")
    print("Example E1 25/25/25/25 EVs:", evs)


if __name__ == "__main__":
    main()
