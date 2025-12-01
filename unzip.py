import gzip
import json
import zipfile
import sqlite3
import os

# Path to the ZIP file and the file inside it
ZIPS_DIR = "zips"
DEST_DB_PATH = "rounds.db"

def parse_mjson_lines(mjson_text: str, log_id: int):
    """
    Given the text of a single mjson game (one JSON object per line),
    yield dicts representing each kyoku with:
      - bakaze, round, honba, kyotaku, oya
      - s_start: [s1, s2, s3, s4]
      - s_final: [f1, f2, f3, f4] (final game scores)
      - log_id
    """
    events = (json.loads(line) for line in mjson_text if line.strip())

    current_scores = None          # [s1, s2, s3, s4]
    rows = []                      # per-kyoku stubs

    for ev in events:
        t = ev.get("type")

        if t in ["tsumo", "dahai", "chi", "pon", "kan", "kakan", "ankan", "dora", "daiminkan", "none", "end_kyoku", "reach", "start_game"]:
            continue # Finding events I don't know about
        elif t == "start_kyoku":
            # Initialize current_scores when a new kyoku starts
            current_scores = ev["scores"][:]
            
            rows.append({
                "log_id": log_id,
                "wind": ev["bakaze"],
                "round": ev["kyoku"],
                "honba": ev["honba"],
                "riichi": ev["kyotaku"],
                "s_start": ev["scores"][:]
            })

        elif t == "reach_accepted":
            current_scores[ev["actor"]] -= 1000

        elif t == "hora":
            for i in range(4):
                current_scores[i] += ev["deltas"][i]

        elif t == "ryukyoku":
            for i in range(4):
                current_scores[i] += ev["deltas"][i]

        elif t == "end_game":
            # Game finished: current_scores is the final game scores
            final_scores = current_scores[:]
            for row in rows:
                row["s_final"] = final_scores[:]
                key = f"{row['log_id']}|{row['wind']}|{row['round']}|{row['honba']}|{row['riichi']}"
                row["round_key"] = key

                yield row #bro yield is so crazy this is genius 

        else:
            print(f"Unknown event type: {t}")

def init_dest_db(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rounds (
            round_key TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,

            wind TEXT NOT NULL,
            round INTEGER NOT NULL,
            honba INTEGER NOT NULL,
            riichi INTEGER NOT NULL,

            s1_start INTEGER NOT NULL,
            s2_start INTEGER NOT NULL,
            s3_start INTEGER NOT NULL,
            s4_start INTEGER NOT NULL,

            s1_final INTEGER NOT NULL,
            s2_final INTEGER NOT NULL,
            s3_final INTEGER NOT NULL,
            s4_final INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    return conn

def get_existing_log_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT log_id FROM rounds;")
    return {row[0] for row in cur.fetchall()}


def process_zip(zip_path: str, dest_conn, existing_log_ids):
    dest_cur = dest_conn.cursor()
    games_in_zip = 0
    kyokus_in_zip = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in z.namelist():
            if not (file_name.endswith(".mjson") or file_name.endswith(".mjson.gz")):
                continue

            log_id = file_name  # should be stable across runs

            # Already processed this game before → skip
            if log_id in existing_log_ids:
                continue

            with z.open(file_name, "r") as zipped_file:
                try:
                    # Try gzip
                    with gzip.GzipFile(fileobj=zipped_file, mode="rb") as f:
                        lines = [line.decode("utf-8") for line in f]
                except gzip.BadGzipFile:
                    # Not gzipped → plain text
                    zipped_file.seek(0)
                    lines = [line.decode("utf-8") for line in zipped_file]

            rows = list(parse_mjson_lines(lines, log_id))

            for r in rows:
                s_start = r["s_start"]
                s_final = r["s_final"]

                dest_cur.execute(
                    """
                    INSERT OR IGNORE INTO rounds (
                        round_key, log_id, wind, round, honba, riichi,
                        s1_start, s2_start, s3_start, s4_start,
                        s1_final, s2_final, s3_final, s4_final
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["round_key"],
                        r["log_id"],
                        r["wind"],
                        r["round"],
                        r["honba"],
                        r["riichi"],
                        s_start[0], s_start[1], s_start[2], s_start[3],
                        s_final[0], s_final[1], s_final[2], s_final[3],
                    ),
                )
                kyokus_in_zip += 1

            if rows:
                existing_log_ids.add(log_id)
                games_in_zip += 1

            if games_in_zip and games_in_zip % 500 == 0:
                dest_conn.commit()
                print(
                    f"[{os.path.basename(zip_path)}] "
                    f"Processed {games_in_zip} games, inserted {kyokus_in_zip} kyokus so far."
                )

    dest_conn.commit()
    print(
        f"[{os.path.basename(zip_path)}] DONE. "
        f"Processed {games_in_zip} new games, inserted {kyokus_in_zip} kyokus from this zip."
    )


def build_rounds_all_years(start_year=2025, end_year=2025):
    conn = init_dest_db(DEST_DB_PATH)
    existing_log_ids = get_existing_log_ids(conn)
    print(f"Loaded {len(existing_log_ids)} existing log_ids from DB.")

    for year in range(start_year, end_year + 1):
        zip_path = os.path.join(ZIPS_DIR, f"{year}.zip")
        if not os.path.exists(zip_path):
            print(f"{zip_path} not found, skipping year {year}.")
            continue

        print(f"\n=== Processing {zip_path} ===")
        process_zip(zip_path, conn, existing_log_ids)

    conn.close()
    print("\nAll done.")


if __name__ == "__main__":
    build_rounds_all_years()