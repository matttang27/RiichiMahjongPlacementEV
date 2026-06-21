"""PyTorch joint-output EV model.

This is the first custom neural-net experiment:

- one full game-state row as input
- four EV outputs jointly
- zero-sum enforced inside the model forward pass
- direct final-EV targets
- optional score-transfer monotonicity penalty during training
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .features import final_evs_thousands, is_supported_wind, normalize_wind_id
    from .helper import _get_round_count
    from .nn_model import encode_joint_state, get_feature_names
except ImportError:  # Allows `python models/torch_nn_model.py`.
    from features import final_evs_thousands, is_supported_wind, normalize_wind_id
    from helper import _get_round_count
    from nn_model import encode_joint_state, get_feature_names


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUNDS_DB_PATH = REPO_ROOT / "data" / "rounds.db"
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "torch_nn" / "joint_v1"
MODEL_PATH = EXPERIMENT_DIR / "model.pt"
EVALUATION_PATH = EXPERIMENT_DIR / "evaluation.txt"
SUMMARY_PATH = EXPERIMENT_DIR / "summary.json"
MONOTONIC_PATH = EXPERIMENT_DIR / "monotonic_checks.txt"
TRAINING_LOG_PATH = EXPERIMENT_DIR / "training_log.json"

FEATURE_VERSION = "torch_joint_v1"
TARGET_MODE = "joint_direct_ev_zero_sum"


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 256
    layers: int = 4
    dropout: float = 0.05
    batch_size: int = 4096
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.1
    monotonic_weight: float = 0.05
    monotonic_delta_thousands: float = 0.1
    random_seed: int = 42


class FeatureScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureScaler":
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X.astype(np.float32) - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureScaler":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
        )


class TargetScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        # Use one shared scale for all seats so zero-sum in scaled space remains
        # zero-sum after converting back to EV units.
        global_scale = float(np.std(y.astype(np.float64)))
        if global_scale < 1e-6:
            global_scale = 1.0
        mean = np.zeros((4,), dtype=np.float32)
        scale = np.full((4,), global_scale, dtype=np.float32)
        return cls(mean=mean, scale=scale)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y.astype(np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform_tensor(self, y_scaled: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=y_scaled.dtype, device=y_scaled.device)
        scale = torch.as_tensor(self.scale, dtype=y_scaled.dtype, device=y_scaled.device)
        return y_scaled * scale + mean

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TargetScaler":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
        )


class JointEVNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(prev_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.backbone(x))
        return y - y.mean(dim=1, keepdim=True)


class TorchJointEVModel:
    def __init__(
        self,
        network: JointEVNet,
        feature_scaler: FeatureScaler,
        target_scaler: TargetScaler,
        config: TrainConfig,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.config = config
        self.device = device

    def predict(self, X: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        X_scaled = self.feature_scaler.transform(X)
        preds: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, X_scaled.shape[0], batch_size):
                batch = torch.from_numpy(X_scaled[start : start + batch_size]).to(self.device)
                pred_scaled = self.network(batch)
                pred = self.target_scaler.inverse_transform_tensor(pred_scaled)
                pred = pred - pred.mean(dim=1, keepdim=True)
                preds.append(pred.cpu().numpy().astype(np.float32))
        return np.vstack(preds)


def _training_scan_limit(
    *,
    db_path: str | Path,
    train_fraction: float,
    max_rows: int | None,
) -> int:
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")

    total = _get_round_count(str(db_path))
    scan_limit = max(1, int(total * train_fraction))
    if max_rows is not None:
        scan_limit = min(scan_limit, int(max_rows))
    return scan_limit


def build_training_matrix(
    db_path: str | Path = ROUNDS_DB_PATH,
    *,
    train_fraction: float = 0.9,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scan_limit = _training_scan_limit(
        db_path=db_path,
        train_fraction=train_fraction,
        max_rows=max_rows,
    )

    feature_count = len(get_feature_names())
    X = np.empty((scan_limit, feature_count), dtype=np.float32)
    y = np.empty((scan_limit, 4), dtype=np.float32)
    meta = np.empty((scan_limit, 8), dtype=np.float32)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT wind, round, honba, riichi,
                   s1_start, s2_start, s3_start, s4_start,
                   s1_final, s2_final, s3_final, s4_final
            FROM rounds
            LIMIT ?
            """,
            (scan_limit,),
        )

        out_idx = 0
        while True:
            row = cur.fetchone()
            if row is None:
                break

            wind = row[0]
            if not is_supported_wind(wind):
                continue

            round_num = int(row[1])
            honba = int(row[2])
            riichi = int(row[3])
            start_scores_pts = list(row[4:8])
            final_scores_pts = list(row[8:12])
            scores_thousands = [s / 1000.0 for s in start_scores_pts]
            X[out_idx, :] = encode_joint_state(
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=scores_thousands,
            )
            y[out_idx, :] = np.asarray(final_evs_thousands(final_scores_pts), dtype=np.float32)
            meta[out_idx, :] = np.asarray(
                [
                    float(normalize_wind_id(wind)),
                    float(round_num),
                    float(honba),
                    float(riichi),
                    *scores_thousands,
                ],
                dtype=np.float32,
            )
            out_idx += 1
            if out_idx >= scan_limit:
                break
    finally:
        conn.close()

    if out_idx == 0:
        raise RuntimeError("No supported East/South rounds loaded from rounds.db.")

    X = X[:out_idx, :]
    y = y[:out_idx, :]
    meta = meta[:out_idx, :]
    print(
        f"Built PyTorch NN training matrix: X.shape={X.shape}, y.shape={y.shape}, "
        f"feature_version={FEATURE_VERSION}, target_mode={TARGET_MODE}"
    )
    return X, y, meta


def _make_monotonic_batch(
    meta_batch: torch.Tensor,
    *,
    delta_thousands: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta_np = meta_batch.detach().cpu().numpy()
    before_rows = []
    after_rows = []
    recipients = []

    for row in meta_np:
        wind, round_num, honba, riichi = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        scores = row[4:8].astype(np.float64)
        recipient = random.randrange(4)
        donors = [i for i in range(4) if i != recipient and scores[i] > delta_thousands]
        if not donors:
            continue
        donor = random.choice(donors)

        after = scores.copy()
        after[recipient] += delta_thousands
        after[donor] -= delta_thousands

        before_rows.append(
            encode_joint_state(
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=scores,
            )
        )
        after_rows.append(
            encode_joint_state(
                wind=wind,
                round_num=round_num,
                honba=honba,
                riichi=riichi,
                scores_thousands=after,
            )
        )
        recipients.append(recipient)

    if not before_rows:
        empty = np.empty((0, len(get_feature_names())), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.int64)

    return (
        np.asarray(before_rows, dtype=np.float32),
        np.asarray(after_rows, dtype=np.float32),
        np.asarray(recipients, dtype=np.int64),
    )


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
    *,
    config: TrainConfig,
    device: str | None = None,
) -> tuple[TorchJointEVModel, list[dict]]:
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    n = X.shape[0]
    rng = np.random.default_rng(config.random_seed)
    order = rng.permutation(n)
    val_count = max(1, int(n * config.validation_fraction))
    val_idx = order[:val_count]
    train_idx = order[val_count:]

    feature_scaler = FeatureScaler.fit(X[train_idx])
    target_scaler = TargetScaler.fit(y[train_idx])

    X_train = feature_scaler.transform(X[train_idx])
    y_train = target_scaler.transform(y[train_idx])
    meta_train = meta[train_idx].astype(np.float32)
    X_val = feature_scaler.transform(X[val_idx])
    y_val = target_scaler.transform(y[val_idx])

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(meta_train),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    X_val_t = torch.from_numpy(X_val).to(torch_device)
    y_val_t = torch.from_numpy(y_val).to(torch_device)

    network = JointEVNet(
        input_dim=X.shape[1],
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    mse = nn.MSELoss()
    history: list[dict] = []

    print(f"Training on {torch_device} with {len(train_idx)} train rows and {len(val_idx)} validation rows")
    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        network.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mono = 0.0
        total_rows = 0

        for xb, yb, metab in train_loader:
            xb = xb.to(torch_device)
            yb = yb.to(torch_device)
            metab = metab.to(torch_device)

            optimizer.zero_grad(set_to_none=True)
            pred = network(xb)
            mse_loss = mse(pred, yb)
            mono_loss = torch.zeros((), dtype=pred.dtype, device=torch_device)

            if config.monotonic_weight > 0.0:
                before_np, after_np, recipients_np = _make_monotonic_batch(
                    metab,
                    delta_thousands=config.monotonic_delta_thousands,
                    device=torch_device,
                )
                if before_np.shape[0] > 0:
                    before = torch.from_numpy(feature_scaler.transform(before_np)).to(torch_device)
                    after = torch.from_numpy(feature_scaler.transform(after_np)).to(torch_device)
                    recipients = torch.from_numpy(recipients_np).to(torch_device)
                    before_pred = target_scaler.inverse_transform_tensor(network(before))
                    after_pred = target_scaler.inverse_transform_tensor(network(after))
                    before_pred = before_pred - before_pred.mean(dim=1, keepdim=True)
                    after_pred = after_pred - after_pred.mean(dim=1, keepdim=True)
                    batch_idx = torch.arange(recipients.shape[0], device=torch_device)
                    recipient_before = before_pred[batch_idx, recipients]
                    recipient_after = after_pred[batch_idx, recipients]
                    mono_loss = torch.relu(recipient_before - recipient_after).mean()

            loss = mse_loss + config.monotonic_weight * mono_loss
            loss.backward()
            optimizer.step()

            rows = xb.shape[0]
            total_loss += float(loss.detach().cpu()) * rows
            total_mse += float(mse_loss.detach().cpu()) * rows
            total_mono += float(mono_loss.detach().cpu()) * rows
            total_rows += rows

        network.eval()
        with torch.no_grad():
            val_pred = network(X_val_t)
            val_mse = float(mse(val_pred, y_val_t).detach().cpu())

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_rows, 1),
            "train_scaled_mse": total_mse / max(total_rows, 1),
            "train_monotonic_penalty_ev": total_mono / max(total_rows, 1),
            "val_scaled_mse": val_mse,
            "elapsed_s": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"mse={row['train_scaled_mse']:.6f} "
            f"mono={row['train_monotonic_penalty_ev']:.6f} "
            f"val_mse={row['val_scaled_mse']:.6f} "
            f"elapsed={elapsed:.1f}s"
        )

    model = TorchJointEVModel(
        network=network,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        config=config,
        device=device,
    )
    return model, history


def save_model(model: TorchJointEVModel, path: str | Path = MODEL_PATH, history: list[dict] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_version": FEATURE_VERSION,
        "target_mode": TARGET_MODE,
        "feature_names": get_feature_names(),
        "config": asdict(model.config),
        "feature_scaler": model.feature_scaler.to_dict(),
        "target_scaler": model.target_scaler.to_dict(),
        "state_dict": model.network.cpu().state_dict(),
        "history": history or [],
    }
    torch.save(payload, p)
    model.network.to(model.device)
    print(f"Saved PyTorch NN model to {p}")


def load_model(path: str | Path = MODEL_PATH, device: str | None = None) -> TorchJointEVModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**payload["config"])
    network = JointEVNet(
        input_dim=len(payload["feature_names"]),
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return TorchJointEVModel(
        network=network,
        feature_scaler=FeatureScaler.from_dict(payload["feature_scaler"]),
        target_scaler=TargetScaler.from_dict(payload["target_scaler"]),
        config=config,
        device=device,
    )


def estimate_all_values(
    model: TorchJointEVModel,
    wind,
    round_num: int,
    honba: int,
    riichi: int,
    scores_thousands: Sequence[int] | Sequence[float],
) -> tuple[float, float, float, float]:
    X = encode_joint_state(
        wind=wind,
        round_num=round_num,
        honba=honba,
        riichi=riichi,
        scores_thousands=scores_thousands,
    ).reshape(1, -1)
    pred = model.predict(X)[0]
    return tuple(float(x) for x in pred)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PyTorch joint-output EV model")
    p.add_argument("--db", default=str(ROUNDS_DB_PATH), help="Path to rounds.db")
    p.add_argument("--model", default=str(MODEL_PATH), help="Where to save the PyTorch model")
    p.add_argument("--training-log", default=str(TRAINING_LOG_PATH), help="Where to save training history JSON")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of DB rows to scan for training; default excludes the last 10%.",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--monotonic-weight", type=float, default=0.05)
    p.add_argument("--monotonic-delta-thousands", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = TrainConfig(
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        monotonic_weight=float(args.monotonic_weight),
        monotonic_delta_thousands=float(args.monotonic_delta_thousands),
        random_seed=int(args.seed),
    )
    X, y, meta = build_training_matrix(
        args.db,
        train_fraction=float(args.train_fraction),
        max_rows=args.max_rows,
    )
    model, history = train_model(X, y, meta, config=config, device=args.device)
    save_model(model, args.model, history=history)

    log_path = Path(args.training_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training log to {log_path}")

    ex_vals = estimate_all_values(
        model,
        wind="S",
        round_num=4,
        honba=0,
        riichi=0,
        scores_thousands=[40.0, 30.0, 14.4, 15.6],
    )
    print("Example S4 EVs [40,30,14.4,15.6]:", ex_vals)


if __name__ == "__main__":
    main()
