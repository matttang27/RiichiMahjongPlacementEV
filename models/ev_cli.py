import argparse
from typing import List, Tuple

try:
    from .xgboost_model import MODEL_PATH, estimate_all_values, load_model
except ImportError:  # Allows `python models/ev_cli.py`.
    from xgboost_model import MODEL_PATH, estimate_all_values, load_model


def _parse_input(line: str) -> Tuple[str, int, int, int, List[int]]:
    """
    Accepts exactly:
      - 8 tokens: wind round honba riichi s1 s2 s3 s4
    Example: "E 1 1 0 25000 25000 25000 25000"
    """
    toks = line.strip().split()
    if len(toks) != 8:
        raise ValueError("Expected 8 tokens: wind round honba riichi s1 s2 s3 s4")

    wind = toks[0]
    if wind not in ("E", "S"):
        raise ValueError("wind must be 'E' or 'S'")

    round_num = int(toks[1])
    honba = min(int(toks[2]), 5)
    riichi = min(int(toks[3]), 5)

    scores = [int(x) for x in toks[4:8]]
    if len(scores) != 4:
        raise ValueError("Need 4 start scores")

    return wind, round_num, honba, riichi, scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive EV predictor")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to XGBoost JSON model")
    args = ap.parse_args()

    model = load_model(args.model)
    print(f"Loaded model: {args.model}")
    print("Enter: wind round honba riichi s1 s2 s3 s4 (scores in raw points).")
    print("Example:  E 1 1 0 25000 25000 25000 25000")
    print("Type 'quit' to exit.")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        try:
            wind, round_num, honba, riichi, scores_pts = _parse_input(line)
        except Exception as e:
            print(f"Parse error: {e}")
            continue

        scores_thousands = [s / 1000.0 for s in scores_pts]
        evs = estimate_all_values(
            model=model,
            wind=wind,
            round_num=round_num,
            honba=honba,
            riichi=riichi,
            scores_thousands=scores_thousands,
        )
        evs = tuple(float(x) for x in evs)
        print(f"Predicted EVs (thousands, relative to 25k): "
              f"seat0={evs[0]:+.3f} seat1={evs[1]:+.3f} seat2={evs[2]:+.3f} seat3={evs[3]:+.3f} | sum={sum(evs):+.3f}")


if __name__ == "__main__":
    main()
