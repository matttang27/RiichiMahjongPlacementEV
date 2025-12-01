# evaluate_ev.py

import sqlite3
import numpy as np
from sklearn.metrics import mean_squared_error
from ev_model import (
    load_model,
    compute_uma,
    estimate_all_values,
)

ROUNDS_DB_PATH = "rounds.db"
MODEL_PATH = "ev_model2.json"


def evaluate_model_ev(max_rows=None):
    conn = sqlite3.connect(ROUNDS_DB_PATH)
    cur = conn.cursor()

    # For now: S1, honba=0, riichi=0, like your original script
    cur.execute("""
        SELECT wind, round, honba, riichi,
               s1_start, s2_start, s3_start, s4_start,
               s1_final, s2_final, s3_final, s4_final
        FROM rounds
        WHERE wind = "S" AND round = 1 AND honba = 0 AND riichi = 0
    """)

    model = load_model(MODEL_PATH)

    actual_evs = []    # true EV, thousands
    baseline_evs = []  # baseline EV, thousands
    model_evs = []     # model EV, thousands

    count = 0
    total_sum_model = 0.0
    total_sum_baseline = 0.0

    while count < 10000:
        if max_rows is not None and count >= max_rows:
            break

        row = cur.fetchone()
        if row is None:
            break

        wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])
        s_start = list(row[4:8])   # start scores (points)
        s_final = list(row[8:12])  # final scores (points)

        # Uma at start and end
        start_uma = compute_uma(s_start)
        final_uma = compute_uma(s_final)

        # Convert start scores to thousands for the model
        s_start_k = [s / 1000.0 for s in s_start]

        # Model EVs in thousands (relative to 25k) for all 4 seats
        # estimate_all_values should already be returning EVs in thousands
        # that sum to ~0 (or exactly 0 if you added recentering).
        model_ev_tuple = estimate_all_values(
            model=model,
            wind=wind,
            round_num=rnd,
            honba=honba,
            riichi=riichi,
            scores_thousands=s_start_k,
        )

        # True EVs and baseline EVs in thousands (relative to 25k)
        # EV = (score + uma)/1000 - 25
        actual_ev_round = []
        baseline_ev_round = []
        model_ev_round = list(model_ev_tuple)

        for seat in range(4):
            actual = (s_final[seat] + final_uma[seat]) / 1000.0 - 25.0
            baseline = (s_start[seat] + start_uma[seat]) / 1000.0 - 25.0

            actual_ev_round.append(actual)
            baseline_ev_round.append(baseline)

        # Accumulate for metrics
        actual_evs.extend(actual_ev_round)
        baseline_evs.extend(baseline_ev_round)
        model_evs.extend(model_ev_round)

        # Sanity: sums per round
        total_sum_model += sum(model_ev_round)
        total_sum_baseline += sum(baseline_ev_round)

        count += 1
        if (count % 1000) == 0:
            print(f"Processed {count} rounds...")

    conn.close()

    actual_evs = np.array(actual_evs, dtype=np.float32)
    baseline_evs = np.array(baseline_evs, dtype=np.float32)
    model_evs = np.array(model_evs, dtype=np.float32)

    # MSE + RMSE (the meaningful EV metrics)
    baseline_mse = mean_squared_error(actual_evs, baseline_evs)
    model_mse = mean_squared_error(actual_evs, model_evs)

    baseline_rmse = np.sqrt(baseline_mse)
    model_rmse = np.sqrt(model_mse)

    # Correlation (how well the model tracks true EVs)
    baseline_corr = np.corrcoef(actual_evs, baseline_evs)[0, 1]
    model_corr = np.corrcoef(actual_evs, model_evs)[0, 1]

    avg_sum_model = total_sum_model / max(count, 1)
    avg_sum_baseline = total_sum_baseline / max(count, 1)

    print("\n=== EV ACCURACY RESULTS (S1, h0, r0) ===")
    print(f"Rounds evaluated      : {count}")
    print(f"Avg sum(model EVs)    : {avg_sum_model:.4f} (should be ~0)")
    print(f"Avg sum(baseline EVs) : {avg_sum_baseline:.4f} (should be ~0)")

    print("\n--- MSE / RMSE (thousands of points² / thousands) ---")
    print(f"Baseline MSE          : {baseline_mse:.4f}")
    print(f"Model MSE             : {model_mse:.4f}")
    print(f"Baseline RMSE         : {baseline_rmse:.4f}")
    print(f"Model RMSE            : {model_rmse:.4f}")

    print("\n--- Correlation with true EV ---")
    print(f"Baseline corr         : {baseline_corr:.4f}")
    print(f"Model corr            : {model_corr:.4f}")

    return {
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "baseline_corr": baseline_corr,
        "model_corr": model_corr,
        "avg_sum_model": avg_sum_model,
        "avg_sum_baseline": avg_sum_baseline,
        "rounds": count,
    }


if __name__ == "__main__":
    evaluate_model_ev()
