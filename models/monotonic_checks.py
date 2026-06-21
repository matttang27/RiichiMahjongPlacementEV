"""Sanity checks for score-transfer monotonicity in EV predictions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from .xgboost_model import (
        SUPPORTED_TARGET_MODES,
        default_model_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
    )
except ImportError:  # Allows `python models/monotonic_checks.py`.
    from xgboost_model import (
        SUPPORTED_TARGET_MODES,
        default_model_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "models" / "experiments"


@dataclass(frozen=True)
class TransferCase:
    name: str
    wind: str
    round_num: int
    honba: int
    riichi: int
    scores: tuple[int, int, int, int]
    player: int
    donor: int
    step_points: int
    steps: int


DEFAULT_CASES = [
    TransferCase(
        name="S4 seat2 chases seat3 from 14400-15600",
        wind="S",
        round_num=4,
        honba=0,
        riichi=0,
        scores=(40000, 30000, 14400, 15600),
        player=2,
        donor=3,
        step_points=100,
        steps=12,
    ),
    TransferCase(
        name="S1 seat2 chases seat3 from 14900-15100",
        wind="S",
        round_num=1,
        honba=0,
        riichi=0,
        scores=(35000, 35000, 14900, 15100),
        player=2,
        donor=3,
        step_points=100,
        steps=4,
    ),
]


def _predict(
    model,
    *,
    case: TransferCase,
    scores: Sequence[int],
    feature_version: str,
    target_mode: str,
) -> tuple[float, float, float, float]:
    return estimate_all_values(
        model,
        wind=case.wind,
        round_num=case.round_num,
        honba=case.honba,
        riichi=case.riichi,
        scores_thousands=[s / 1000.0 for s in scores],
        feature_version=feature_version,
        target_mode=target_mode,
    )


def run_checks(
    *,
    model_path: str | Path,
    feature_version: str,
    target_mode: str,
    tolerance: float,
) -> tuple[list[str], int]:
    model = load_model(model_path)
    lines = [
        "# Monotonic EV sanity checks",
        "",
        f"Model: {Path(model_path).resolve()}",
        f"Feature version: {feature_version}",
        f"Target mode: {target_mode}",
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

            evs = _predict(
                model,
                case=case,
                scores=scores,
                feature_version=feature_version,
                target_mode=target_mode,
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
    p = argparse.ArgumentParser(description="Check EV monotonicity on score-transfer cases")
    p.add_argument(
        "--features",
        default="legacy",
        choices=["legacy", "v1", "v2"],
        help="Feature version expected by the model",
    )
    p.add_argument(
        "--target-mode",
        default=None,
        choices=list(SUPPORTED_TARGET_MODES),
        help="Prediction target mode. Defaults by feature version.",
    )
    p.add_argument("--model", default=None, help="Path to XGBoost JSON model")
    p.add_argument(
        "--out",
        default=None,
        help="Where to write the text report. Defaults under models/experiments.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed EV decrease in thousands before counting a violation.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model) if args.model else default_model_path_for_features(args.features)
    target_mode = args.target_mode or default_target_mode_for_features(args.features)
    out_path = (
        Path(args.out)
        if args.out
        else DEFAULT_EXPERIMENTS_DIR / "xgboost" / args.features / "monotonic_checks.txt"
    )

    lines, violations = run_checks(
        model_path=model_path,
        feature_version=args.features,
        target_mode=target_mode,
        tolerance=float(args.tolerance),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote monotonic check report to: {out_path.resolve()}")
    print(f"Total violations: {violations}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
