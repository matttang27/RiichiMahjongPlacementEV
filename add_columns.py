import sqlite3
from typing import Optional
import numpy as np

from ev_model import compute_uma, _encode_state_row

ROUNDS_DB_PATH = "rounds.db"

def add_ev_columns(db_path: str = ROUNDS_DB_PATH):
    """
    ALTER TABLE to add per-seat target residuals and placements.

    Columns added:
      s1_y_residual, ..., s4_y_residual  (REAL)
      s1_place, ..., s4_place            (INTEGER, 1–4)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Try adding columns; ignore if they already exist
    def try_add(column_def: str):
        try:
            cur.execute(f"ALTER TABLE rounds ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            # Probably "duplicate column name" – safe to ignore
            pass

    for seat in range(1, 5):
        try_add(f"s{seat}_y_residual REAL")
    for seat in range(1, 5):
        try_add(f"s{seat}_place INTEGER")

    conn.commit()
    conn.close()


def compute_placements(final_scores: list[int]) -> list[int]:
    """
    Given final raw scores [s1, s2, s3, s4], return placements [p1..p4],
    where p_i in {1,2,3,4}, 1 = first, 4 = fourth.

    Tie-breaking: higher score wins; ties broken by seat index (1 < 2 < 3 < 4).
    """
    # (negative score so we sort descending; seat_idx for stable tiebreak)
    order = sorted(
        [( -score, seat_idx) for seat_idx, score in enumerate(final_scores)],
        key=lambda t: (t[0], t[1])
    )
    # order[k] = ( -score, seat_idx ) for placement k+1
    placements = [0] * 4
    for place, (_, seat_idx) in enumerate(order, start=1):
        placements[seat_idx] = place
    return placements


def backfill_ev_targets_and_placements(
    db_path: str = ROUNDS_DB_PATH,
    max_rows: Optional[int] = None
):
    """
    For each row in `rounds`, compute:
      - y_residual for each seat (same definition as in build_training_matrix)
      - final placement for each seat (1–4 based on final scores)

    and write them into the new columns.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure columns exist
    add_ev_columns(db_path)

    # Select with rowid so we can UPDATE by rowid
    cur.execute(
        """
        SELECT rowid, wind, round, honba, riichi,
               s1_start, s2_start, s3_start, s4_start,
               s1_final, s2_final, s3_final, s4_final
        FROM rounds
        """
    )

    rows_processed = 0
    batch_size = 1000

    while True:
        batch = cur.fetchmany(batch_size)
        if not batch:
            break

        for row in batch:
            rowid = row[0]
            wind = row[1]
            round_num = int(row[2])
            honba = int(row[3])
            riichi = int(row[4])

            # Skip West rounds for consistency with training
            if wind not in ("E", "S"):
                continue

            s_start_pts = list(row[5:9])
            s_final_pts = list(row[9:13])

            start_uma_pts = compute_uma(s_start_pts)
            final_uma_pts = compute_uma(s_final_pts)

            # placements from final raw scores (you can change to use EV if you prefer)
            placements = compute_placements(s_final_pts)

            y_residuals: list[float] = []

            for seat in range(4):
                baseline_thousands = (
                    s_start_pts[seat] + start_uma_pts[seat]
                ) / 1000.0

                target_thousands = (
                    s_final_pts[seat] + final_uma_pts[seat]
                ) / 1000.0

                y_residual = target_thousands - baseline_thousands
                y_residuals.append(float(y_residual))

            # UPDATE this row
            cur.execute(
                """
                UPDATE rounds
                SET s1_y_residual = ?, s2_y_residual = ?,
                    s3_y_residual = ?, s4_y_residual = ?,
                    s1_place = ?, s2_place = ?,
                    s3_place = ?, s4_place = ?
                WHERE rowid = ?
                """,
                (
                    y_residuals[0], y_residuals[1],
                    y_residuals[2], y_residuals[3],
                    placements[0], placements[1],
                    placements[2], placements[3],
                    rowid,
                )
            )

            rows_processed += 1
            if max_rows is not None and rows_processed >= max_rows:
                conn.commit()
                conn.close()
                print(f"Backfilled {rows_processed} rows (truncated by max_rows).")
                return

        conn.commit()

    conn.close()
    print(f"Backfilled {rows_processed} rows total.")

if __name__ == "__main__":
    backfill_ev_targets_and_placements()