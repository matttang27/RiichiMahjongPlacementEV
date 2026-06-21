"""Interactive CLI for the PyTorch joint-output EV model."""

from __future__ import annotations

import argparse
from typing import List, Tuple

try:
    from .torch_nn_model import FEATURE_VERSION, MODEL_PATH, TARGET_MODE, estimate_all_values, load_model
except ImportError:  # Allows `python models/torch_nn_cli.py`.
    from torch_nn_model import FEATURE_VERSION, MODEL_PATH, TARGET_MODE, estimate_all_values, load_model


def _parse_input(line: str) -> Tuple[str, int, int, int, List[int]]:
    toks = line.strip().split()
    if len(toks) != 8:
        raise ValueError("Expected 8 tokens: wind round honba riichi s1 s2 s3 s4")

    wind = toks[0]
    if wind not in ("E", "S"):
        raise ValueError("wind must be 'E' or 'S'")

    round_num = int(toks[1])
    honba = int(toks[2])
    riichi = int(toks[3])
    scores = [int(x) for x in toks[4:8]]
    return wind, round_num, honba, riichi, scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive PyTorch NN EV predictor")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to PyTorch model")
    ap.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    print(f"Loaded model: {args.model}")
    print(f"Feature version: {FEATURE_VERSION}")
    print(f"Target mode: {TARGET_MODE}")
    print("Enter: wind round honba riichi s1 s2 s3 s4 (scores in raw points).")
    print("Example:  E 1 0 0 25000 25000 25000 25000")
    print("Type 'quit' to exit.")

    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().lower() in ("q", "quit", "exit"):
            break
        if not line.strip():
            continue

        try:
            wind, round_num, honba, riichi, scores_pts = _parse_input(line)
            evs = estimate_all_values(
                model,
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=[s / 1000.0 for s in scores_pts],
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(
            "Predicted EVs (thousands, relative to 25k): "
            f"seat0={evs[0]:+.3f} seat1={evs[1]:+.3f} "
            f"seat2={evs[2]:+.3f} seat3={evs[3]:+.3f} | sum={sum(evs):+.3f}"
        )


if __name__ == "__main__":
    main()
