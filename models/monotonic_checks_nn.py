"""Monotonic sanity checks for the joint-output NN EV model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

try:
    from .monotonic_checks import DEFAULT_CASES
    from .nn_model import FEATURE_VERSION, MODEL_PATH, MONOTONIC_PATH, TARGET_MODE, estimate_all_values, load_model
except ImportError:  # Allows `python models/monotonic_checks_nn.py`.
    from monotonic_checks import DEFAULT_CASES
    from nn_model import FEATURE_VERSION, MODEL_PATH, MONOTONIC_PATH, TARGET_MODE, estimate_all_values, load_model


def run_checks(
    *,
    model_path: str | Path,
    tolerance: float,
) -> tuple[list[str], int]:
    model = load_model(model_path)
    lines = [
        "# Monotonic EV sanity checks",
        "",
        f"Model: {Path(model_path).resolve()}",
        f"Feature version: {FEATURE_VERSION}",
        f"Target mode: {TARGET_MODE}",
        f"Tolerance: {tolerance:.3f}",
        "",
    ]

    total_violations = 0
    for case in DEFAULT_CASES:
        lines.append(f"## {case.name}")
        lines.append("delta_pts\tscores\tplayer_ev\tall_evs")

        last_ev: float | None = None
        case_violations = 0
        for step in range(case.steps + 1):
            delta = step * case.step_points
            scores = list(case.scores)
            scores[case.player] += delta
            scores[case.donor] -= delta

            evs = estimate_all_values(
                model,
                wind=case.wind,
                round_num=case.round_num,
                honba=case.honba,
                riichi=case.riichi,
                scores_thousands=[s / 1000.0 for s in scores],
            )
            player_ev = float(evs[case.player])
            if last_ev is not None and player_ev + tolerance < last_ev:
                case_violations += 1
                marker = " VIOLATION"
            else:
                marker = ""
            last_ev = player_ev

            ev_text = ", ".join(f"{ev:+.3f}" for ev in evs)
            lines.append(
                f"{delta}\t{tuple(scores)}\t{player_ev:+.3f}\t{ev_text}{marker}"
            )

        total_violations += case_violations
        lines.append(f"Violations: {case_violations}")
        lines.append("")

    lines.append(f"Total violations: {total_violations}")
    return lines, total_violations


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check NN EV monotonicity on score-transfer cases")
    p.add_argument("--model", default=str(MODEL_PATH), help="Path to NN joblib model")
    p.add_argument("--out", default=str(MONOTONIC_PATH), help="Where to write the text report")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed EV decrease in thousands before counting a violation.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    lines, violations = run_checks(model_path=args.model, tolerance=float(args.tolerance))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote monotonic check report to: {out_path.resolve()}")
    print(f"Total violations: {violations}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
