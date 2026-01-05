# EV Modeling Notes (Long-Form)

This is the longer conceptual writeup that’s intentionally *not* in `.github/copilot-instructions.md` (which is meant to stay short and workflow-focused).

## Problem definition

We estimate **expected value (EV)** in Riichi Mahjong.

- **Input** $X$: game state at the start of a round
  - wind / round (E1–S4, etc.)
  - honba
  - riichi sticks (kyotaku)
  - current scores of 4 players
- **Output** $Y$: per-player final value
  - Tenhou uma: `90 / 45 / 0 / -135` (ties broken by lower seat index)
  - In this repo’s units (thousands):

$$EV = (final\_score + final\_uma)/1000 - 25$$

Goal: learn $f(X) = \mathbb{E}[Y\mid X]$, i.e. the **conditional mean**, not a single outcome.

Outcomes are high-variance and heavy-tailed; individual game “errors” can be large and still be consistent with a good EV estimator.

## RMSE vs MAE (why RMSE is the metric here)

- **RMSE** (squared error) is the proper scoring rule for the **mean**.
  - Minimizing expected squared error predicts $\mathbb{E}[Y\mid X]$.
  - This makes RMSE appropriate for both training and evaluation when the target is EV.
- **MAE** is the proper scoring rule for the **median**.
  - With heavy tails, MAE prefers conservative/shrunk predictions.
  - MAE can rank a biased EV estimator above a correct one.

Conclusion: RMSE is the correct scalar metric for EV-as-mean; MAE is diagnostic only.

## Calibration (the most important correctness test)

A model is calibrated if:

> When it predicts EV = $v$, the average realized EV of all states with that prediction is $v$.

Practical check (bucket calibration):
- Take held-out data
- Bucket by predicted EV
- For each bucket: compare mean predicted EV vs mean realized EV
- A calibrated model will lie near the diagonal

A model can have high per-game error and still be well calibrated; calibration targets the *mean correctness* property directly.

## Structural constraints

Riichi end results are effectively **zero-sum** under fixed total points.
- The repo enforces this by projecting predicted values so the implied `(score+uma)/1000` sums to `100.0` (100k points) across the 4 seats.

## Evaluation hierarchy (recommended)

1. Calibration by EV buckets (primary)
2. RMSE (proper scalar metric for mean estimation)
3. Structural sanity checks: zero-sum violations, behavior by round (E vs S), behavior in sparse endgame states
