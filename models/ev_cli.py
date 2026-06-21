import argparse
from typing import List, Tuple

try:
    from .xgboost_model import (
        SUPPORTED_TARGET_MODES,
        default_model_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
    )
except ImportError:  # Allows `python models/ev_cli.py`.
    from xgboost_model import (
        SUPPORTED_TARGET_MODES,
        default_model_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
    )


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
    ap.add_argument(
        "--features",
        default="legacy",
        choices=["legacy", "v1", "v2"],
        help="Feature version expected by the model",
    )
    ap.add_argument(
        "--target-mode",
        default=None,
        choices=list(SUPPORTED_TARGET_MODES),
        help="Prediction target mode. Defaults by feature version.",
    )
    ap.add_argument("--model", default=None, help="Path to XGBoost JSON model")
    args = ap.parse_args()

    model_path = args.model or str(default_model_path_for_features(args.features))
    target_mode = args.target_mode or default_target_mode_for_features(args.features)
    model = load_model(model_path)
    print(f"Loaded model: {model_path}")
    print(f"Feature version: {args.features}")
    print(f"Target mode: {target_mode}")
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
            feature_version=args.features,
            target_mode=target_mode,
        )
        evs = tuple(float(x) for x in evs)
        print(f"Predicted EVs (thousands, relative to 25k): "
              f"seat0={evs[0]:+.3f} seat1={evs[1]:+.3f} seat2={evs[2]:+.3f} seat3={evs[3]:+.3f} | sum={sum(evs):+.3f}")


if __name__ == "__main__":
    main()
