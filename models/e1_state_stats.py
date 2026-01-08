import argparse
import sqlite3
import time
import math

from helper import UMA, _get_round_count  # [90, 45, 0, -135] (index by place-1)


def compute_e1_final_ev_stats(db_path: str) -> None:
    t0 = time.perf_counter()

    # Use last 10% of the database (validation slice)
    total_count = _get_round_count(db_path)
    start_index = int(total_count * 0.1)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_final, s2_final, s3_final, s4_final,
                   s1_place, s2_place, s3_place, s4_place
            FROM rounds
            ORDER BY rowid
            """,
        )

        count = 0
        sum_ev_by_seat = [0.0, 0.0, 0.0, 0.0]
        sumsq_ev_by_seat = [0.0, 0.0, 0.0, 0.0]

        for row in cur:
            wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])
            # Filter to the identical start state: East 1, honba=0, riichi=0
            if not (wind == "E" and rnd == 1 and honba == 0 and riichi == 0):
                continue

            finals = [int(v) for v in row[4:8]]
            places = [int(v) for v in row[8:12]]

            # EV(thousands) = (final_score + final_uma)/1000 - 25
            for i in range(4):
                uma_pts = UMA[places[i] - 1]
                ev_thousands = (finals[i] + uma_pts) / 1000.0 - 25.0
                sum_ev_by_seat[i] += ev_thousands
                sumsq_ev_by_seat[i] += ev_thousands * ev_thousands

            count += 1

        print("=== East 1, honba=0, riichi=0: Final EV statistics (last 10% of DB) ===")
        print(f"Total rows in DB: {total_count}")
        print(f"Validation slice starts at row: {start_index}")
        print(f"Matching rounds in slice: {count}")

        if count == 0:
            print("No rows found in the validation slice.")
            return

        avg_ev_by_seat = [s / float(count) for s in sum_ev_by_seat]
        std_ev_by_seat = [
            math.sqrt((s2 / float(count)) - (avg_ev_by_seat[i] ** 2))
            for i, s2 in enumerate(sumsq_ev_by_seat)
        ]

        print("Sum of final EV by seat (thousands):")
        print(
            f"  seat0: {sum_ev_by_seat[0]:+.4f}  seat1: {sum_ev_by_seat[1]:+.4f}  "
            f"seat2: {sum_ev_by_seat[2]:+.4f}  seat3: {sum_ev_by_seat[3]:+.4f}"
        )
        print("Average final EV by seat (thousands):")
        print(
            f"  seat0: {avg_ev_by_seat[0]:+.4f}  seat1: {avg_ev_by_seat[1]:+.4f}  "
            f"seat2: {avg_ev_by_seat[2]:+.4f}  seat3: {avg_ev_by_seat[3]:+.4f}"
        )
        print("Std dev of final EV by seat (thousands):")
        print(
            f"  seat0: {std_ev_by_seat[0]:.4f}  seat1: {std_ev_by_seat[1]:.4f}  "
            f"seat2: {std_ev_by_seat[2]:.4f}  seat3: {std_ev_by_seat[3]:.4f}"
        )

    finally:
        conn.close()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.2f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute final EV stats for E1 honba=0 riichi=0 (last 10% of DB)")
    p.add_argument("--db", default="data/rounds.db", help="Path to SQLite rounds.db")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compute_e1_final_ev_stats(args.db)


if __name__ == "__main__":
    main()