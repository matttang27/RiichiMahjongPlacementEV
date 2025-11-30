import gzip
import json
import zipfile

# Path to the ZIP file and the file inside it
zip_path = "2025.zip"
file_in_zip = "2025/2025010100gm-00a9-0000-029e0f35.mjson"



def print_final_scores(mjson_path: str) -> None:
    # Read all lines once so we can scan & then re-scan from the last start_kyoku
    with open(mjson_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_start_idx = None
    last_start_scores = None

    # 1) Find the last start_kyoku and remember its scores + index
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "start_kyoku":
            last_start_idx = i
            last_start_scores = list(event["scores"])
            print(last_start_scores)

    if last_start_idx is None:
        print("No start_kyoku found in file.")
        return

    # 2) Initialize scores to the scores at the start of the last hand
    scores = list(last_start_scores)

    # 3) Walk events of the last hand and update scores
    for j in range(last_start_idx + 1, len(lines)):
        line = lines[j].strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")

        if etype == "reach_accepted":
            actor = event["actor"]
            scores[actor] -= 1000

        elif etype in ("hora", "ryukyoku"):
            deltas = event["deltas"]
            # Apply deltas to all four players
            for idx in range(4):
                scores[idx] += deltas[idx]

    print("Final scores:", scores)

with zipfile.ZipFile(zip_path, 'r') as zip_file:
    with zip_file.open(file_in_zip) as compressed_file:
        # Since the file is gzipped, we need to decompress it
        with gzip.open(compressed_file, "rt", encoding="utf-8") as f:
            with open("output.txt2", "w", encoding="utf-8") as output_file:
                
                for line in f:
                    event = json.loads(line)
                    output_file.write(json.dumps(event) + "\n")

print_final_scores("output.txt")
print_final_scores("output.txt2")

