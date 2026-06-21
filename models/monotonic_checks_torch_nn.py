"""Monotonic sanity checks for the PyTorch joint-output EV model."""

from __future__ import annotations

import argparse
import random
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from .features import is_supported_wind
    from .monotonic_checks import DEFAULT_CASES, TransferCase
    from .torch_nn_model import FEATURE_VERSION, MODEL_PATH, MONOTONIC_PATH, TARGET_MODE, estimate_all_values, load_model
except ImportError:  # Allows `python models/monotonic_checks_torch_nn.py`.
    from features import is_supported_wind
    from monotonic_checks import DEFAULT_CASES, TransferCase
    from torch_nn_model import FEATURE_VERSION, MODEL_PATH, MONOTONIC_PATH, TARGET_MODE, estimate_all_values, load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "rounds.db"


@dataclass(frozen=True)
class RandomCaseSpec:
    count: int
    steps: int
    step_points: int
    seed: int


def _load_random_cases(db_path: str | Path, spec: RandomCaseSpec) -> list[TransferCase]:
    if spec.count <= 0:
        return []

    rng = random.Random(spec.seed)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start
            FROM rounds
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (spec.count * 5,),
        )
        cases: list[TransferCase] = []
        idx = 0
        for row in cur.fetchall():
            wind = row[0]
            if not is_supported_wind(wind):
                continue
            scores = [int(v) for v in row[4:8]]
            recipient = rng.randrange(4)
            donors = [
                i
                for i in range(4)
                if i != recipient and scores[i] > spec.steps * spec.step_points
            ]
            if not donors:
                continue
            donor = rng.choice(donors)
            cases.append(
                TransferCase(
                    name=f"random {idx}: seat{recipient} gains from seat{donor}",
                    wind="E" if int(wind) == 0 else "S",
                    round_num=int(row[1]),
                    honba=int(row[2]),
                    riichi=int(row[3]),
                    scores=tuple(scores),
                    player=recipient,
                    donor=donor,
                    step_points=spec.step_points,
                    steps=spec.steps,
                )
            )
            idx += 1
            if len(cases) >= spec.count:
                break
    finally:
        conn.close()
    return cases


def run_checks(
    *,
    model_path: str | Path,
    tolerance: float,
    db_path: str | Path,
    random_case_spec: RandomCaseSpec,
    device: str | None,
) -> tuple[list[str], int]:
    model = load_model(model_path, device=device)
    cases = list(DEFAULT_CASES) + _load_random_cases(db_path, random_case_spec)
    lines = [
        "# Monotonic EV sanity checks",
        "",
        f"Model: {Path(model_path).resolve()}",
        f"Feature version: {FEATURE_VERSION}",
        f"Target mode: {TARGET_MODE}",
        f"Tolerance: {tolerance:.3f}",
        f"Random cases: {random_case_spec.count}",
        "",
    ]

    total_violations = 0
    for case in cases:
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
    p = argparse.ArgumentParser(description="Check PyTorch NN EV monotonicity")
    p.add_argument("--model", default=str(MODEL_PATH), help="Path to PyTorch model")
    p.add_argument("--out", default=str(MONOTONIC_PATH), help="Where to write the text report")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite rounds.db")
    p.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed EV decrease in thousands before counting a violation.",
    )
    p.add_argument("--random-cases", type=int, default=0, help="Additional random score-transfer states")
    p.add_argument("--random-steps", type=int, default=6, help="Steps per random case")
    p.add_argument("--random-step-points", type=int, default=100, help="Point transfer per random step")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    lines, violations = run_checks(
        model_path=args.model,
        tolerance=float(args.tolerance),
        db_path=args.db,
        random_case_spec=RandomCaseSpec(
            count=int(args.random_cases),
            steps=int(args.random_steps),
            step_points=int(args.random_step_points),
            seed=int(args.seed),
        ),
        device=args.device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote monotonic check report to: {out_path.resolve()}")
    print(f"Total violations: {violations}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
