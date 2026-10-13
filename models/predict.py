"""Checkpoint-backed inference using the same feature and sampler contracts as training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import DATASET_PATH, MAX_NEIGHBORS_PER_ENDPOINT, MODEL_PATH
from data.generate_data import FEATURE_NAMES
from graph.temporal_sampler import NODE_FEATURE_NAMES, TemporalNeighborSampler
from models.gatv2_model import GATv2EdgeClassifier, model_from_checkpoint


def load_checkpoint(model_path: str | Path = MODEL_PATH) -> tuple[GATv2EdgeClassifier, dict[str, object]]:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run: python -m models.train")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    return model_from_checkpoint(checkpoint), checkpoint


def load_transactions(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Synthetic dataset not found at {dataset_path}. Run: python -m models.train")
    return pd.read_csv(dataset_path)


def build_custom_target(
    transactions: pd.DataFrame,
    sender_id: str,
    receiver_id: str,
    amount: float,
) -> pd.DataFrame:
    """Append an unlabeled, latest-time transfer with features computed from its prefix."""
    if sender_id == receiver_id:
        raise ValueError("Sender and receiver must be different accounts")
    if amount <= 0:
        raise ValueError("Transaction amount must be greater than zero")
    if sender_id not in set(transactions.sender_id) | set(transactions.receiver_id):
        raise ValueError("Sender account is not in this demonstration dataset")
    if receiver_id not in set(transactions.sender_id) | set(transactions.receiver_id):
        raise ValueError("Receiver account is not in this demonstration dataset")

    history = transactions.sort_values("timestamp", kind="stable")
    cutoff = int(history.timestamp.max()) + 1
    prior = history.loc[history.timestamp < cutoff]
    sender_history = prior.loc[prior.sender_id == sender_id]
    receiver_history = prior.loc[prior.receiver_id == receiver_id]
    pair_history = prior.loc[(prior.sender_id == sender_id) & (prior.receiver_id == receiver_id)]

    def last_balance(account: str) -> float:
        sent = prior.loc[prior.sender_id == account, ["timestamp", "sender_balance_after"]].rename(
            columns={"sender_balance_after": "balance"}
        )
        received = prior.loc[prior.receiver_id == account, ["timestamp", "receiver_balance_after"]].rename(
            columns={"receiver_balance_after": "balance"}
        )
        events = pd.concat([sent, received]).sort_values("timestamp", kind="stable")
        return float(events.balance.iloc[-1]) if not events.empty else 20_000.0

    sender_balance = last_balance(sender_id)
    receiver_balance = last_balance(receiver_id)
    sender_amounts = sender_history.amount.to_numpy(dtype=np.float64)
    all_amounts = prior.amount.to_numpy(dtype=np.float64)
    sender_mean = float(sender_amounts.mean()) if len(sender_amounts) else 300.0
    sender_std = float(sender_amounts.std()) if len(sender_amounts) > 1 else 300.0
    global_mean = float(all_amounts.mean()) if len(all_amounts) else 300.0
    global_std = float(all_amounts.std()) if len(all_amounts) > 1 else 300.0
    timestamp = pd.Timestamp.utcfromtimestamp(cutoff)
    row: dict[str, object] = {
        "transaction_id": "custom_unlabeled",
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "timestamp": cutoff,
        "amount": float(amount),
        "sender_balance_before": sender_balance,
        "sender_balance_after": sender_balance - amount,
        "receiver_balance_before": receiver_balance,
        "receiver_balance_after": receiver_balance + amount,
        "fraud_label": 0,
        "amount_log1p": float(np.log1p(amount)),
        "amount_over_sender_balance": float(amount / max(sender_balance, 1.0)),
        "amount_over_receiver_balance": float(amount / max(receiver_balance, 1.0)),
        "sender_balance_log1p": float(np.log1p(sender_balance)),
        "receiver_balance_log1p": float(np.log1p(receiver_balance)),
        "hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * timestamp.weekday() / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * timestamp.weekday() / 7)),
        "sender_tx_count_1h": float((sender_history.timestamp > cutoff - 3600).sum()),
        "sender_tx_count_24h": float((sender_history.timestamp > cutoff - 86400).sum()),
        "receiver_tx_count_24h": float((receiver_history.timestamp > cutoff - 86400).sum()),
        "receiver_seen_before": float(not receiver_history.empty),
        "sender_receiver_tx_count_before": float(len(pair_history)),
        "amount_z_sender_history": float((amount - sender_mean) / max(sender_std, 100.0)),
        "amount_z_global_history": float((amount - global_mean) / max(global_std, 100.0)),
    }
    return pd.concat([history, pd.DataFrame([row])], ignore_index=True)


def prepare_model_sample(
    transactions: pd.DataFrame,
    target_index: int,
    checkpoint: dict[str, object],
    neighbors_per_endpoint: int = MAX_NEIGHBORS_PER_ENDPOINT,
    sampler: TemporalNeighborSampler | None = None,
):
    if sampler is None:
        sampler = TemporalNeighborSampler(transactions, neighbors_per_endpoint)
    sample = sampler.sample(target_index)
    means = np.asarray(checkpoint["edge_scaler_mean"], dtype=np.float32)
    scales = np.asarray(checkpoint["edge_scaler_scale"], dtype=np.float32)
    graph = Data(
        x=sample.x,
        edge_index=sample.edge_index,
        source_index=torch.tensor([sample.source_index], dtype=torch.long),
        destination_index=torch.tensor([sample.destination_index], dtype=torch.long),
        transaction_features=torch.tensor(
            ((sample.transaction_features.numpy() - means) / scales).reshape(1, -1), dtype=torch.float32
        ),
    )
    return sample, graph


def predict_sample(
    model: GATv2EdgeClassifier,
    graph: Data,
) -> float:
    model.eval()
    with torch.no_grad():
        logit = model(
            graph.x,
            graph.edge_index,
            graph.source_index,
            graph.destination_index,
            graph.transaction_features,
        )
    return float(torch.sigmoid(logit).item())