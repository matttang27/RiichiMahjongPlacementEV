import sqlite3

DB = "rounds.db"

def bucket(x):
    """Buckets honba/riichi: 0–4 stay, >=5 becomes 5."""
    return x if x < 5 else 5

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Pull raw wind, round, honba, riichi counts
    cur.execute("""
        SELECT wind, round, honba, riichi, COUNT(*)
        FROM rounds
        GROUP BY wind, round, honba, riichi
    """)
    rows = cur.fetchall()

    # Bucketed counts {(wind, round, H_bucket, R_bucket): count}
    counts = {}

    for wind, rnd, honba, riichi, cnt in rows:
        hb = bucket(honba)
        rb = bucket(riichi)
        key = (wind, rnd, hb, rb)
        counts[key] = counts.get(key, 0) + cnt

    # Define ordering for pretty printing
    wind_order = {"E": 0, "S": 1, "W": 2, "N": 3}
    
    def sort_key(item):
        (wind, rnd, hb, rb), count = item
        return (wind_order.get(wind, 99), rnd, hb, rb)

    print("wind round honba_bucket riichi_bucket  count")
    print("===========================================")

    for (wind, rnd, hb, rb), cnt in sorted(counts.items(), key=sort_key):
        hb_label = f"H{hb}" if hb < 5 else "H5+"
        rb_label = f"R{rb}" if rb < 5 else "R5+"
        print(f"{wind} {rnd} {hb_label} {rb_label} : {cnt}")

    conn.close()


if __name__ == "__main__":
    main()
