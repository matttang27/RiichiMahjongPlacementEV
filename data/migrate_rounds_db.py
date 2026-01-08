"""data/migrate_rounds_db.py

Rebuild `rounds` with a new schema without re-unzipping Tenhou logs.

This script reads an existing `rounds` table that already has, at minimum:
    - round_key, log_id, wind, round, honba, riichi
    - s1_start..s4_start, s1_final..s4_final

It writes a new SQLite DB containing a `rounds` table matching the schema
created by `data/unzip.py:init_dest_db`, notably including:
    - s*_start_s and s*_final_s ("score relative to 25k + uma", in points)
    - s*_place (final placement 1..4)

The derived fields are computed solely from the existing start/final scores:
    placement: sort by score desc, tie-break by seat index asc
    uma (Tenhou): 90 / 45 / 0 / -135 (thousands)
    *_s: score_pts - 25000 + uma_thousands(place) * 1000

Typical usage (from repo root):

    python data/migrate_rounds_db.py --src data/rounds.db --dst data/rounds_v2.db

"""

from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Iterable, Sequence


UMA_THOUSANDS = (90, 45, 0, -135)


def compute_placements(scores_pts: Sequence[int]) -> list[int]:
    if len(scores_pts) != 4:
        raise ValueError("scores must have length 4")

    order = sorted(range(4), key=lambda i: (-int(scores_pts[i]), i))
    placements = [0] * 4
    for place, seat in enumerate(order, start=1):
        placements[seat] = place
    return placements


def _create_rounds_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    score_cols = ",\n            ".join(
        [f"s{i}_start INTEGER NOT NULL" for i in range(1, 5)]
        + [f"s{i}_final INTEGER NOT NULL" for i in range(1, 5)]
        + [f"s{i}_start_s INTEGER" for i in range(1, 5)]
        + [f"s{i}_final_s INTEGER" for i in range(1, 5)]
        + [f"s{i}_place INTEGER" for i in range(1, 5)]
    )

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rounds (
            round_key TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,

            wind TEXT NOT NULL,
            round INTEGER NOT NULL,
            honba INTEGER NOT NULL,
            riichi INTEGER NOT NULL,

            {score_cols}
        );
        """
    )

    conn.commit()


def _iter_source_rows(cur: sqlite3.Cursor, batch_size: int) -> Iterable[tuple]:
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            return
        yield from rows


def migrate_rounds_db(*, src_db: str, dst_db: str, batch_size: int = 50_000) -> None:
    if not os.path.exists(src_db):
        raise FileNotFoundError(src_db)

    os.makedirs(os.path.dirname(dst_db) or ".", exist_ok=True)
    if os.path.exists(dst_db):
        raise FileExistsError(f"Destination DB already exists: {dst_db}")

    src_conn = sqlite3.connect(src_db)
    try:
        src_cur = src_conn.cursor()
        src_cur.execute(
            """
            SELECT round_key, log_id, wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final
            FROM rounds
            ORDER BY rowid
            """
        )

        dst_conn = sqlite3.connect(dst_db)
        try:
            _create_rounds_table(dst_conn)
            dst_cur = dst_conn.cursor()

            insert_sql = (
                "INSERT INTO rounds ("
                "round_key, log_id, wind, round, honba, riichi, "
                "s1_start, s2_start, s3_start, s4_start, "
                "s1_final, s2_final, s3_final, s4_final, "
                "s1_start_s, s2_start_s, s3_start_s, s4_start_s, "
                "s1_final_s, s2_final_s, s3_final_s, s4_final_s, "
                "s1_place, s2_place, s3_place, s4_place"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            )

            pending: list[tuple] = []
            total = 0

            for row in _iter_source_rows(src_cur, batch_size=batch_size):
                (
                    round_key,
                    log_id,
                    wind,
                    round_num,
                    honba,
                    riichi,
                    s1_start,
                    s2_start,
                    s3_start,
                    s4_start,
                    s1_final,
                    s2_final,
                    s3_final,
                    s4_final,
                ) = row

                s_start = [int(s1_start), int(s2_start), int(s3_start), int(s4_start)]
                s_final = [int(s1_final), int(s2_final), int(s3_final), int(s4_final)]

                start_places = compute_placements(s_start)
                final_places = compute_placements(s_final)

                def uma_pts(place: int) -> int:
                    return int(UMA_THOUSANDS[place - 1]) * 1000

                s_start_s = [s_start[i] - 25000 + uma_pts(start_places[i]) for i in range(4)]
                s_final_s = [s_final[i] - 25000 + uma_pts(final_places[i]) for i in range(4)]

                out_row = (
                    str(round_key),
                    str(log_id),
                    str(wind),
                    int(round_num),
                    int(honba),
                    int(riichi),
                    *s_start,
                    *s_final,
                    *s_start_s,
                    *s_final_s,
                    *final_places,
                )

                pending.append(out_row)
                total += 1

                if len(pending) >= batch_size:
                    dst_cur.executemany(insert_sql, pending)
                    dst_conn.commit()
                    pending.clear()
                    print(f"Migrated {total:,} rows...")

            if pending:
                dst_cur.executemany(insert_sql, pending)
                dst_conn.commit()
                pending.clear()

            # Basic sanity check
            src_cur2 = src_conn.cursor()
            src_cur2.execute("SELECT COUNT(*) FROM rounds")
            src_count = int(src_cur2.fetchone()[0])

            dst_cur2 = dst_conn.cursor()
            dst_cur2.execute("SELECT COUNT(*) FROM rounds")
            dst_count = int(dst_cur2.fetchone()[0])

            if src_count != dst_count:
                raise RuntimeError(
                    f"Row count mismatch after migration: src={src_count} dst={dst_count}"
                )

            print(f"Done. Migrated {dst_count:,} rows to {dst_db}")
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Rebuild rounds.db with start_s/final_s/place columns computed from existing start/final scores"
    )
    p.add_argument("--src", default="data/rounds.db", help="Source SQLite DB path")
    p.add_argument("--dst", default="data/rounds_v2.db", help="Destination SQLite DB path (must not exist)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per insert batch (affects speed/memory)",
    )

    args = p.parse_args(argv)
    migrate_rounds_db(src_db=args.src, dst_db=args.dst, batch_size=int(args.batch_size))


if __name__ == "__main__":
    main()
