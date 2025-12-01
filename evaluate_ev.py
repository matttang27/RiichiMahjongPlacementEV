# evaluate_ev.py

import sqlite3
import numpy as np
from sklearn.metrics import mean_absolute_error
from ev_model import (
    load_model,
    compute_uma,
    _encode_state_row,
    estimate_all_values
)

ROUNDS_DB_PATH = "rounds.db"
MODEL_PATH = "ev_model.json"


def evaluate_model_ev(max_rows=None):
    conn = sqlite3.connect(ROUNDS_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT wind, round, honba, riichi,
               s1_start, s2_start, s3_start, s4_start,
               s1_final, s2_final, s3_final, s4_final
        FROM rounds
        WHERE wind = "S" AND round = 1 AND honba = 0 AND riichi = 0
    """)

    model = load_model(MODEL_PATH)

    baseline_errors = []
    model_errors = []

    count = 0

    total_sum_model = 0

    while count < 5000:
        row = cur.fetchone()
        if row is None:
            break

        wind, rnd, honba, riichi = row[0], int(row[1]), int(row[2]), int(row[3])
        s_start = list(row[4:8])   # start scores (points)
        s_final = list(row[8:12])  # final scores (points)

        # Compute baseline uma
        start_uma = compute_uma(s_start)
        final_uma = compute_uma(s_final)

        # Convert to thousands for model
        s_start_k = [s / 1000.0 for s in s_start]
        
        predict = estimate_all_values(
            model=model,
            wind=wind,
            round_num=rnd,
            honba=honba,
            riichi=riichi,
            scores_thousands=s_start_k,
        )

        for seat in range(4):
            baseline_ev = s_start[seat] + start_uma[seat]
            actual_ev = s_final[seat] + final_uma[seat]
            model_ev = predict[seat] * 1000.0  # back to points

            baseline_errors.append(abs(actual_ev - baseline_ev))
            model_errors.append(abs(actual_ev - model_ev))
        
        if (sum(predict)) < 0:
            print(s_start)
        total_sum_model += sum(predict)

        count += 1
        if (count % 1000) == 0:
            print(f"Processed {count} rounds...")
        if max_rows is not None and count >= max_rows:
            break

    conn.close()

    baseline_mae = np.mean(baseline_errors)
    model_mae = np.mean(model_errors)

    print("\n=== EV ACCURACY RESULTS ===")
    print(f"Average sum of model EVs per round: {total_sum_model / count:.3f}")
    print(f"Baseline MAE: {baseline_mae:.3f}")
    print(f"Model MAE   : {model_mae:.3f}")
    print(f"Improvement : {baseline_mae - model_mae:.3f}")
    print(f"% Improvement: {(baseline_mae - model_mae) / baseline_mae * 100:.2f}%")

    return baseline_mae, model_mae


if __name__ == "__main__":
    evaluate_model_ev()
