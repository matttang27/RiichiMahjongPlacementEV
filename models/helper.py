UMA = [90000, 45000, 0, -135000]

import sqlite3

def _get_round_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rounds")
        return int(cur.fetchone()[0])
    finally:
        conn.close()

def compute_uma(final_scores, uma_scheme=UMA):
    """
    final_scores: list/tuple of 4 ints in points (e.g. [45000, 25000, ...])
    returns: list of 4 ints in points (e.g. [90000, 0, 45000, -135000])
    """
    # Higher score is better; break ties by smaller seat index.
    order = sorted(range(4), key=lambda i: (-final_scores[i], i))
    uma = [0, 0, 0, 0]
    for seat, pts in zip(order, uma_scheme):
        uma[seat] = pts
    return uma