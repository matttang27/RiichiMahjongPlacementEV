"""evaluate_model.py

Evaluate EV predictors against held-out rounds from `data/rounds.db`.

Uses the default JSON/XGBoost model (via `ev_model.estimate_all_values`).

Predictions are 4 EVs (one per seat), in *thousands*, relative to 25k:

        EV = (final_score + final_uma)/1000 - 25

Notes on metrics:
    - Calibration by EV buckets is the primary correctness check for EV-as-mean.
    - RMSE is the proper scalar metric for estimating the conditional mean.
    - MAE is reported as a diagnostic only (it targets the median).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .features import is_supported_wind, normalize_wind_label
    from .helper import UMA, _get_round_count
    from .xgboost_model import (
        default_evaluation_path_for_features,
        default_model_path_for_features,
        default_summary_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
        SUPPORTED_TARGET_MODES,
    )
except ImportError:  # Allows `python models/evaluate_model.py`.
    from features import is_supported_wind, normalize_wind_label
    from helper import UMA, _get_round_count
    from xgboost_model import (
        default_evaluation_path_for_features,
        default_model_path_for_features,
        default_summary_path_for_features,
        default_target_mode_for_features,
        estimate_all_values,
        load_model,
        SUPPORTED_TARGET_MODES,
    )


# Last 10% of database is for validation by default
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATE_SPLIT = 0.1
DEFAULT_DB_PATH = str(REPO_ROOT / "data" / "rounds.db")
DEFAULT_CALIBRATION_BUCKETS = 20
DEFAULT_ORACLE_MIN_ROUNDS = 20
DEFAULT_ORACLE_SCORE_BUCKET = 1000

DEFAULT_GROUP_CALIB_MIN_SAMPLES = 200  # measured in seat-samples (4 per round)

model = None
current_feature_version = "legacy"
current_target_mode = "residual_uma"


def predict(wind, round_num, honba, riichi, scores_pts) -> tuple[float, float, float, float]:
    if model is None:
        raise RuntimeError("Model has not been loaded. Call evaluate_model_ev first.")

    scores_div = [s / 1000.0 for s in scores_pts]

    evs = estimate_all_values(
        model=model,
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_div,
        feature_version=current_feature_version,
        target_mode=current_target_mode,
    )
    return evs


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) * (a - b)))


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _format_group_key(key: tuple[str, int, int, int]) -> str:
    wind, rnd, honba_b, riichi_b = key
    # Example: "S1 honba=1 riichi=1" (South 1, 1 honba, 1 riichi)
    return f"{wind}{rnd} honba={honba_b} riichi={riichi_b}"


# Sorts the list of actual EVs and model EVs by the model ev.
# Buckets similar model EV together to see whether the actual EV mean matches.
def _print_calibration_by_EV_buckets(
    actual: np.ndarray, model: np.ndarray, buckets: int
) -> tuple[list[str], float]:
    if buckets <= 0 or actual.size == 0:
        return [], float("nan")

    buckets = max(min(buckets, int(actual.size)), 1)

    order = np.argsort(model, kind="mergesort")
    actual_sorted = actual[order]
    model_sorted = model[order]

    # Split into approximately-equal-count buckets (quantile buckets).
    splits = np.array_split(np.arange(model_sorted.size), buckets)

    lines: list[str] = []
    lines.append("\n--- Calibration by predicted-EV buckets (primary check) ---")
    lines.append("bucket\tcount\tmean_pred\tmean_actual\tdiff")
    total_diff = 0.0
    for idx, s in enumerate(splits, start=1):
        if s.size == 0:
            continue
        mp = float(np.mean(model_sorted[s]))
        ma = float(np.mean(actual_sorted[s]))
        diff = ma - mp
        total_diff += abs(diff)
        lines.append(f"{idx:02d}\t{s.size}\t{mp:+.3f}\t\t{ma:+.3f}\t\t{diff:+.3f}")
    avg_abs_diff = total_diff / float(len(splits))
    lines.append(
        f"Avg abs diff over all buckets: {avg_abs_diff:+.3f} - note this changes with #buckets"
    )
    return lines, avg_abs_diff


def _metrics_block(
    *,
    actual: np.ndarray,
    model: np.ndarray,
    rounds: int,
    skipped_rounds: int,
    total_sum_model: float,
    sum_model_squares: float,
    max_abs_sum_model: float,
) -> list[str]:
    if rounds == 0:
        return [
            "\n=== EV ACCURACY RESULTS ===",
            "Rounds evaluated      : 0",
            f"Rounds skipped (W+)   : {skipped_rounds}",
            "No rounds to evaluate (check DB / validate split).",
        ]

    model_mse = _mse(actual, model)
    model_rmse = float(np.sqrt(model_mse))
    model_mae = float(np.mean(np.abs(actual - model)))
    model_corr = _safe_corrcoef(actual, model)

    avg_sum_model = total_sum_model / max(rounds, 1)
    sum_model_rmse = float(np.sqrt(sum_model_squares / max(rounds, 1)))

    lines: list[str] = []
    lines.append("\n=== EV ACCURACY RESULTS ===")
    lines.append(f"Rounds evaluated      : {rounds}")
    lines.append(f"Rounds skipped (W+)   : {skipped_rounds}")
    lines.append(f"Avg sum(model EVs)    : {avg_sum_model:.4f} (should be ~0)")
    lines.append(f"RMSE(sum model EVs)   : {sum_model_rmse:.4f} (should be small)")
    lines.append(f"Max |sum model EVs|   : {max_abs_sum_model:.4f}")

    lines.append("\n--- MSE / RMSE (thousands of points² / thousands) ---")
    lines.append(f"Model MSE             : {model_mse:.4f}")
    lines.append(f"Model RMSE            : {model_rmse:.4f}")
    lines.append(f"Model MAE (diagnostic): {model_mae:.4f}")

    lines.append("\n--- Correlation with true EV ---")
    lines.append(f"Model corr            : {model_corr:.4f}")
    return lines


@dataclass
class _GroupAccum:
    actual_evs: list[float] = field(default_factory=list)
    model_evs: list[float] = field(default_factory=list)
    rounds: int = 0
    total_sum_model: float = 0.0
    sum_model_squares: float = 0.0
    max_abs_sum_model: float = 0.0


@dataclass
class _OracleGroupAccum:
    actual_by_round: list[tuple[float, float, float, float]] = field(default_factory=list)
    model_by_round: list[tuple[float, float, float, float]] = field(default_factory=list)


def _bucket_score(score_pts: int | float, bucket_points: int) -> int:
    if bucket_points <= 0:
        return int(score_pts)
    return int(round(float(score_pts) / float(bucket_points)) * bucket_points)


def _oracle_metrics(
    groups: dict[tuple, _OracleGroupAccum],
    *,
    min_rounds: int,
) -> dict:
    min_rounds = max(2, int(min_rounds))
    oracle_errors: list[float] = []
    model_errors: list[float] = []
    covered_rounds = 0
    eligible_groups = 0

    for group in groups.values():
        n_rounds = len(group.actual_by_round)
        if n_rounds < min_rounds:
            continue

        actual_arr = np.asarray(group.actual_by_round, dtype=np.float64)
        model_arr = np.asarray(group.model_by_round, dtype=np.float64)
        sum_by_seat = np.sum(actual_arr, axis=0)

        eligible_groups += 1
        covered_rounds += n_rounds

        # Leave-one-out group mean avoids the direct self-leak that would make
        # singleton or tiny groups look artificially perfect.
        oracle_pred = (sum_by_seat.reshape(1, 4) - actual_arr) / float(n_rounds - 1)
        oracle_errors.extend((actual_arr - oracle_pred).reshape(-1).tolist())
        model_errors.extend((actual_arr - model_arr).reshape(-1).tolist())

    if not oracle_errors:
        return {
            "eligible_groups": 0,
            "covered_rounds": 0,
            "seat_samples": 0,
            "oracle_mse": float("nan"),
            "oracle_rmse": float("nan"),
            "oracle_mae": float("nan"),
            "model_mse_on_subset": float("nan"),
            "model_rmse_on_subset": float("nan"),
            "model_mae_on_subset": float("nan"),
        }

    oracle_err = np.asarray(oracle_errors, dtype=np.float64)
    model_err = np.asarray(model_errors, dtype=np.float64)
    return {
        "eligible_groups": eligible_groups,
        "covered_rounds": covered_rounds,
        "seat_samples": int(oracle_err.size),
        "oracle_mse": float(np.mean(oracle_err * oracle_err)),
        "oracle_rmse": float(np.sqrt(np.mean(oracle_err * oracle_err))),
        "oracle_mae": float(np.mean(np.abs(oracle_err))),
        "model_mse_on_subset": float(np.mean(model_err * model_err)),
        "model_rmse_on_subset": float(np.sqrt(np.mean(model_err * model_err))),
        "model_mae_on_subset": float(np.mean(np.abs(model_err))),
    }


def _oracle_report_lines(
    *,
    exact: dict,
    coarse: dict,
    count_rounds: int,
    min_rounds: int,
    score_bucket_points: int,
) -> list[str]:
    def block(name: str, metrics: dict) -> list[str]:
        covered = int(metrics["covered_rounds"])
        coverage = 0.0 if count_rounds == 0 else 100.0 * covered / float(count_rounds)
        lines = [
            f"\n{name}",
            f"Eligible groups       : {int(metrics['eligible_groups'])}",
            f"Covered rounds        : {covered} ({coverage:.2f}% of evaluated rounds)",
            f"Seat-samples          : {int(metrics['seat_samples'])}",
        ]
        if covered == 0:
            lines.append("No groups met the minimum round count.")
            return lines

        lines.extend(
            [
                f"Oracle RMSE           : {metrics['oracle_rmse']:.4f}",
                f"Oracle MAE            : {metrics['oracle_mae']:.4f}",
                f"Model RMSE same subset: {metrics['model_rmse_on_subset']:.4f}",
                f"Model MAE same subset : {metrics['model_mae_on_subset']:.4f}",
            ]
        )
        return lines

    lines = [
        "\n\n=== EMPIRICAL ORACLE / NOISE FLOOR CHECK ===",
        "Oracle uses leave-one-out group-average actual EV from the validation slice.",
        "This is not the true perfect RMSE; it is a repeated/similar-state noise-floor diagnostic.",
        f"Minimum group size: {min_rounds} rounds",
        f"Coarse score bucket: nearest {score_bucket_points} points",
    ]
    lines.extend(block("Exact-state oracle", exact))
    lines.extend(block("Coarse-state oracle", coarse))
    return lines


def _next_available_report_path(out_path: str) -> Path:
    """
    Never overwrite an existing report file.
    If out_path exists, write out_path with an incrementing suffix: _1, _2, ...
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        return p

    stem = p.stem
    suffix = p.suffix or ".txt"
    i = 1
    while True:
        candidate = p.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def evaluate_model_ev(
    *,
    db_path: str,
    model_path: str,
    feature_version: str,
    target_mode: str,
    validate_split: float,
    max_rounds: int | None,
    calibration_buckets: int,
    out_path: str,
    summary_out_path: str | None,
    oracle_min_rounds: int = DEFAULT_ORACLE_MIN_ROUNDS,
    oracle_score_bucket: int = DEFAULT_ORACLE_SCORE_BUCKET,
    model_loader=None,
    predictor=None,
) -> dict:
    t0 = time.perf_counter()

    global model, current_feature_version, current_target_mode
    current_feature_version = feature_version
    current_target_mode = target_mode
    loader = model_loader or load_model
    model = loader(model_path)

    def call_predict(wind, round_num, honba, riichi, scores_pts):
        if predictor is not None:
            return predictor(model, wind, round_num, honba, riichi, scores_pts)
        return predict(wind, round_num, honba, riichi, scores_pts)

    if not (0.0 < validate_split < 1.0):
        raise ValueError("validate_split must be in (0, 1)")

    total_count = _get_round_count(db_path)
    start_index = int(total_count * (1 - validate_split))

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final,
                     s1_place, s2_place, s3_place, s4_place
            FROM rounds
            ORDER BY rowid
            LIMIT -1 OFFSET ?
            """,
            (start_index,),
        )

        actual_evs: list[float] = []
        model_evs: list[float] = []

        count_rounds = 0
        skipped_rounds = 0
        total_sum_model = 0.0
        abs_sum_model_max = 0.0
        sum_model_squares = 0.0

        # Grouped stats: (wind, round_num, honba_bucket, riichi_bucket) -> accumulator
        group_acc: dict[tuple[str, int, int, int], _GroupAccum] = defaultdict(_GroupAccum)
        exact_oracle_groups: dict[tuple, _OracleGroupAccum] = defaultdict(_OracleGroupAccum)
        coarse_oracle_groups: dict[tuple, _OracleGroupAccum] = defaultdict(_OracleGroupAccum)

        # Special identical-state accumulator: E1, honba=0, riichi=0, all scores=25000
        e1_identical_rounds = 0
        e1_sum_actual_by_seat = np.zeros(4, dtype=np.float64)

        for row in cur:
            wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])

            # Match repo convention: evaluate only East/South games.
            if not is_supported_wind(wind):
                skipped_rounds += 1
                continue

            honba_b = min(honba, 5)
            riichi_b = min(riichi, 5)
            wind_label = normalize_wind_label(wind)
            gkey = (wind_label, int(rnd), honba_b, riichi_b)

            s_start_pts = list(row[4:8])
            s_final_pts = row[8:12]
            s_final_places = row[12:16]
            final_uma = [UMA[place - 1] for place in s_final_places]

            pred = call_predict(wind, rnd, honba_b, riichi_b, s_start_pts)
            if len(pred) != 4:
                raise ValueError(f"Model must return 4 EVs. Got {len(pred)}")

            # Per-seat datapoints (overall + group)
            per_seat_actual = [
                (s_final_pts[i] + final_uma[i]) / 1000.0 - 25.0
                for i in range(4)
            ]
            per_seat_model = [float(x) for x in pred]
            actual_tuple = tuple(float(x) for x in per_seat_actual)
            model_tuple = tuple(float(x) for x in per_seat_model)

            actual_evs.extend(per_seat_actual)
            model_evs.extend(per_seat_model)

            ga = group_acc[gkey]
            ga.actual_evs.extend(per_seat_actual)
            ga.model_evs.extend(per_seat_model)

            exact_key = (
                wind_label,
                int(rnd),
                int(honba),
                int(riichi),
                tuple(int(s) for s in s_start_pts),
            )
            coarse_key = (
                wind_label,
                int(rnd),
                int(honba_b),
                int(riichi_b),
                tuple(_bucket_score(s, int(oracle_score_bucket)) for s in s_start_pts),
            )
            exact_oracle_groups[exact_key].actual_by_round.append(actual_tuple)
            exact_oracle_groups[exact_key].model_by_round.append(model_tuple)
            coarse_oracle_groups[coarse_key].actual_by_round.append(actual_tuple)
            coarse_oracle_groups[coarse_key].model_by_round.append(model_tuple)

            # Special identical-state check: only E1/0/0 with all scores == 25000
            if (
                wind_label == "E"
                and int(rnd) == 1
                and int(honba_b) == 0
                and int(riichi_b) == 0
                and s_start_pts == [25000, 25000, 25000, 25000]
            ):
                e1_identical_rounds += 1
                e1_sum_actual_by_seat += np.array(per_seat_actual, dtype=np.float64)

            # Per-round zero-sum checks (overall + group)
            round_sum = float(sum(pred))
            total_sum_model += round_sum
            abs_sum_model_max = max(abs_sum_model_max, abs(round_sum))
            sum_model_squares += round_sum * round_sum

            ga.rounds += 1
            ga.total_sum_model += round_sum
            ga.max_abs_sum_model = max(ga.max_abs_sum_model, abs(round_sum))
            ga.sum_model_squares += round_sum * round_sum

            count_rounds += 1
            if max_rounds is not None and count_rounds >= max_rounds:
                break
            if (count_rounds % 1000) == 0:
                print(f"Processed {count_rounds} rounds...")
    finally:
        conn.close()

    actual = np.array(actual_evs, dtype=np.float32)
    model_arr = np.array(model_evs, dtype=np.float32)

    # Build report (single string -> console + file)
    report_lines: list[str] = []
    report_lines.append(f"Using model: {model_path}")
    report_lines.append(f"Feature version: {feature_version}")
    report_lines.append(f"Target mode: {target_mode}")
    report_lines.append(f"DB: {db_path}")
    report_lines.append(f"Validate split (last N%): {validate_split}")
    report_lines.append(f"Max rounds cap: {max_rounds}")
    report_lines.append(f"Calibration buckets: {calibration_buckets}")

    report_lines.extend(
        _metrics_block(
            actual=actual,
            model=model_arr,
            rounds=count_rounds,
            skipped_rounds=skipped_rounds,
            total_sum_model=total_sum_model,
            sum_model_squares=sum_model_squares,
            max_abs_sum_model=abs_sum_model_max,
        )
    )

    # Overall calibration (kept)
    calibration_avg_abs_diff = float("nan")
    if count_rounds != 0:
        calibration_lines, calibration_avg_abs_diff = _print_calibration_by_EV_buckets(
            actual=actual, model=model_arr, buckets=int(calibration_buckets)
        )
        report_lines.extend(calibration_lines)

    exact_oracle = _oracle_metrics(
        exact_oracle_groups,
        min_rounds=int(oracle_min_rounds),
    )
    coarse_oracle = _oracle_metrics(
        coarse_oracle_groups,
        min_rounds=int(oracle_min_rounds),
    )
    report_lines.extend(
        _oracle_report_lines(
            exact=exact_oracle,
            coarse=coarse_oracle,
            count_rounds=count_rounds,
            min_rounds=int(oracle_min_rounds),
            score_bucket_points=int(oracle_score_bucket),
        )
    )

    # Special identical-state section (E1 honba=0 riichi=0, all 25000)
    report_lines.append("\n\n=== SPECIAL CHECK: IDENTICAL START STATE (E1 honba=0 riichi=0, all 25000) ===")
    if e1_identical_rounds == 0:
        report_lines.append("No matching rounds found in validation slice.")
    else:
        avg_actual = (e1_sum_actual_by_seat / float(e1_identical_rounds)).astype(float)
        canonical_pred = tuple(
            float(x) for x in call_predict("E", 1, 0, 0, [25000, 25000, 25000, 25000])
        )

        report_lines.append(f"Rounds in this exact state: {e1_identical_rounds}")
        report_lines.append(
            "Avg ACTUAL EV by seat: "
            + ", ".join(f"{float(avg_actual[i]):+.3f}" for i in range(4))
        )
        report_lines.append(
            "MODEL prediction for this state (single call): "
            + ", ".join(f"{float(canonical_pred[i]):+.3f}" for i in range(4))
        )

    # Per-group report (metrics only; no per-group calibration)
    report_lines.append("\n\n=== PER-ROUND-STATE BREAKDOWN (METRICS ONLY) ===")
    report_lines.append(
        "Groups are keyed by (wind, round_num, honba_bucket<=5, riichi_bucket<=5)."
    )
    report_lines.append(
        "Note: sample counts below are seat-samples; divide by 4 to approximate rounds."
    )

    for gkey in sorted(group_acc.keys()):
        ga = group_acc[gkey]
        g_actual = np.array(ga.actual_evs, dtype=np.float32)
        g_model = np.array(ga.model_evs, dtype=np.float32)

        report_lines.append("\n" + ("-" * 72))
        report_lines.append(f"Group: {_format_group_key(gkey)}")
        report_lines.append(f"Rounds evaluated in group: {ga.rounds}")
        report_lines.append(f"Seat-samples in group    : {int(g_actual.size)}")

        report_lines.extend(
            _metrics_block(
                actual=g_actual,
                model=g_model,
                rounds=ga.rounds,
                skipped_rounds=0,
                total_sum_model=ga.total_sum_model,
                sum_model_squares=ga.sum_model_squares,
                max_abs_sum_model=ga.max_abs_sum_model,
            )
        )

    elapsed_s = time.perf_counter() - t0

    report_text = "\n".join(
        [
            "=== TIMING ===",
            f"Total analysis time: {elapsed_s:.2f}s",
            "",
            *report_lines,
            "",
        ]
    )

    out_file = _next_available_report_path(out_path)
    out_file.write_text(report_text, encoding="utf-8")

    print(f"Wrote evaluation report to: {out_file.resolve()} (took {elapsed_s:.2f}s)")

    if count_rounds == 0:
        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_path": str(Path(model_path).resolve()),
            "feature_version": feature_version,
            "target_mode": target_mode,
            "db_path": str(Path(db_path).resolve()),
            "validate_split": validate_split,
            "max_rounds": max_rounds,
            "calibration_buckets": calibration_buckets,
            "oracle_min_rounds": oracle_min_rounds,
            "oracle_score_bucket": oracle_score_bucket,
            "oracle_exact": exact_oracle,
            "oracle_coarse": coarse_oracle,
            "model_mse": float("nan"),
            "model_rmse": float("nan"),
            "model_mae": float("nan"),
            "model_corr": float("nan"),
            "avg_sum_model": float("nan"),
            "sum_model_rmse": float("nan"),
            "max_abs_sum_model": float("nan"),
            "rounds": 0,
            "skipped_rounds": skipped_rounds,
            "out_path": str(out_file),
        }
        if summary_out_path is not None:
            _write_summary(result, summary_out_path)
        return result

    model_mse = _mse(actual, model_arr)
    model_rmse = float(np.sqrt(model_mse))
    model_mae = float(np.mean(np.abs(actual - model_arr)))
    model_corr = _safe_corrcoef(actual, model_arr)

    avg_sum_model = total_sum_model / max(count_rounds, 1)
    sum_model_rmse = float(np.sqrt(sum_model_squares / max(count_rounds, 1)))

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(Path(model_path).resolve()),
        "feature_version": feature_version,
        "target_mode": target_mode,
        "db_path": str(Path(db_path).resolve()),
        "validate_split": validate_split,
        "max_rounds": max_rounds,
        "calibration_buckets": calibration_buckets,
        "oracle_min_rounds": oracle_min_rounds,
        "oracle_score_bucket": oracle_score_bucket,
        "model_mse": model_mse,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "model_corr": model_corr,
        "calibration_avg_abs_diff": calibration_avg_abs_diff,
        "oracle_exact": exact_oracle,
        "oracle_coarse": coarse_oracle,
        "avg_sum_model": avg_sum_model,
        "sum_model_rmse": sum_model_rmse,
        "max_abs_sum_model": abs_sum_model_max,
        "rounds": count_rounds,
        "skipped_rounds": skipped_rounds,
        "out_path": str(out_file),
    }
    if summary_out_path is not None:
        _write_summary(result, summary_out_path)
    return result


def _write_summary(result: dict, summary_out_path: str) -> None:
    p = Path(summary_out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote evaluation summary to: {p.resolve()}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EV predictor on held-out rounds")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to SQLite rounds.db")
    p.add_argument(
        "--model",
        default=None,
        help="Path to XGBoost JSON model. Defaults by feature version.",
    )
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
    p.add_argument(
        "--validate-split",
        type=float,
        default=DEFAULT_VALIDATE_SPLIT,
        help="Fraction of rows used for validation (uses last N%%)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional cap on number of validation rounds",
    )
    p.add_argument(
        "--calibration-buckets",
        type=int,
        default=DEFAULT_CALIBRATION_BUCKETS,
        help="Number of quantile buckets for calibration table (0 to disable)",
    )
    p.add_argument(
        "--oracle-min-rounds",
        type=int,
        default=DEFAULT_ORACLE_MIN_ROUNDS,
        help="Minimum rounds in an exact/coarse state group for oracle diagnostics",
    )
    p.add_argument(
        "--oracle-score-bucket",
        type=int,
        default=DEFAULT_ORACLE_SCORE_BUCKET,
        help="Score bucket size in points for coarse oracle diagnostics",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Where to write the full text report. Defaults by feature version.",
    )
    p.add_argument(
        "--summary-out",
        default=None,
        help="Where to write compact JSON metrics. Defaults by feature version.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = _parse_args(argv)
    model_path = Path(args.model) if args.model is not None else default_model_path_for_features(args.features)
    out_path = Path(args.out) if args.out is not None else default_evaluation_path_for_features(args.features)
    target_mode = args.target_mode or default_target_mode_for_features(args.features)
    summary_out_path = (
        Path(args.summary_out)
        if args.summary_out is not None
        else default_summary_path_for_features(args.features)
    )

    print(f"Using JSON model: {model_path}")
    print(f"Using feature version: {args.features}")
    print(f"Using target mode: {target_mode}")
    return evaluate_model_ev(
        db_path=str(args.db),
        model_path=str(model_path),
        feature_version=args.features,
        target_mode=target_mode,
        validate_split=float(args.validate_split),
        max_rounds=args.max_rounds,
        calibration_buckets=int(args.calibration_buckets),
        out_path=str(out_path),
        summary_out_path=str(summary_out_path),
        oracle_min_rounds=int(args.oracle_min_rounds),
        oracle_score_bucket=int(args.oracle_score_bucket),
    )


if __name__ == "__main__":
    main()
