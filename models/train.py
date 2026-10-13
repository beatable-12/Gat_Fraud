"""Train and save the compact chronological GATv2 demonstration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch

from config import (
    BATCH_SIZE,
    DATASET_PATH,
    EPOCHS,
    LEARNING_RATE,
    MAX_NEIGHBORS_PER_ENDPOINT,
    METADATA_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    SEED,
    WEIGHT_DECAY,
)
from data.generate_data import FEATURE_NAMES, generate_transactions
from data.preprocess import chronological_split
from evaluation.baseline import class_prior_probabilities
from evaluation.metrics import calculate_metrics
from graph.temporal_sampler import NODE_FEATURE_NAMES, TemporalNeighborSampler, TemporalSample
from models.gatv2_model import GATv2EdgeClassifier


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_sample(sample: TemporalSample, edge_scaler) -> Data:
    graph = Data(x=sample.x, edge_index=sample.edge_index)
    graph.source_index = torch.tensor([sample.source_index], dtype=torch.long)
    graph.destination_index = torch.tensor([sample.destination_index], dtype=torch.long)
    graph.transaction_features = torch.tensor(
        edge_scaler.transform(sample.transaction_features.numpy().reshape(1, -1)), dtype=torch.float32
    )
    graph.y = sample.label.reshape(1)
    graph.target_index = torch.tensor([sample.target_index], dtype=torch.long)
    return graph


def _samples_for(sampler: TemporalNeighborSampler, indices: list[int], edge_scaler) -> list[Data]:
    return [_prepare_sample(sampler.sample(index), edge_scaler) for index in indices]


def _predict(model: GATv2EdgeClassifier, graphs: list[Data], batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[float] = []
    labels: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size])
            logits = model(
                batch.x,
                batch.edge_index,
                batch.source_index,
                batch.destination_index,
                batch.transaction_features,
            )
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch.y.cpu().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def train_model(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
    n_transactions: int = 2_400,
    epochs: int = EPOCHS,
) -> dict[str, object]:
    seed_everything()
    dataset_path, model_path, metadata_path = map(Path, (dataset_path, model_path, metadata_path))
    if dataset_path.exists():
        transactions = pd.read_csv(dataset_path)
    else:
        transactions = generate_transactions(n_transactions=n_transactions)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(dataset_path, index=False)
    train_frame, validation_frame, test_frame = chronological_split(transactions)

    # Feature transforms see training targets only; graph history itself remains strictly temporal.
    edge_scaler = StandardScaler().fit(train_frame[FEATURE_NAMES].to_numpy(dtype=np.float32))
    train_sampler = TemporalNeighborSampler(train_frame, MAX_NEIGHBORS_PER_ENDPOINT)
    train_graphs = _samples_for(train_sampler, list(range(len(train_frame))), edge_scaler)
    validation_sampler = TemporalNeighborSampler(pd.concat([train_frame, validation_frame], ignore_index=True), MAX_NEIGHBORS_PER_ENDPOINT)
    validation_offset = len(train_frame)
    validation_graphs = _samples_for(
        validation_sampler,
        list(range(validation_offset, validation_offset + len(validation_frame))), edge_scaler,
    )
    complete_history = pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)
    test_sampler = TemporalNeighborSampler(complete_history, MAX_NEIGHBORS_PER_ENDPOINT)
    test_offset = len(train_frame) + len(validation_frame)
    test_graphs = _samples_for(
        test_sampler,
        list(range(test_offset, test_offset + len(test_frame))), edge_scaler,
    )

    model = GATv2EdgeClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_labels = np.asarray([int(graph.y.item()) for graph in train_graphs], dtype=np.int64)
    positive_count = max(int(train_labels.sum()), 1)
    negative_count = max(len(train_labels) - int(train_labels.sum()), 1)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative_count / positive_count))
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_graphs))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_graphs[int(index)] for index in order[start : start + BATCH_SIZE]]
            batch = Batch.from_data_list(selected)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch.x,
                batch.edge_index,
                batch.source_index,
                batch.destination_index,
                batch.transaction_features,
            )
            loss = loss_function(logits, batch.y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected)

        validation_labels, validation_probabilities = _predict(model, validation_graphs)
        validation_metrics = calculate_metrics(validation_labels, validation_probabilities)
        history.append({
            "epoch": float(epoch + 1),
            "train_loss": total_loss / len(train_graphs),
            "validation_average_precision": float(validation_metrics["average_precision"]),
        })
        print(
            f"epoch {epoch + 1:02d}/{epochs} loss={history[-1]['train_loss']:.4f} "
            f"validation AP={history[-1]['validation_average_precision']:.4f}"
        )

    test_labels, test_probabilities = _predict(model, test_graphs)
    test_metrics = calculate_metrics(test_labels, test_probabilities)
    prior_probabilities = class_prior_probabilities(train_labels, len(test_labels))
    prior_metrics = calculate_metrics(test_labels, prior_probabilities)
    predictions_path = Path(PREDICTIONS_PATH)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "transaction_id": test_frame.transaction_id.to_numpy(),
        "label": test_labels,
        "fraud_probability": test_probabilities,
        "class_prior_probability": prior_probabilities,
        "split": "chronological_test",
    }).to_csv(predictions_path, index=False)
    checkpoint = {
        "state_dict": model.state_dict(),
        "node_features": len(NODE_FEATURE_NAMES),
        "transaction_features": len(FEATURE_NAMES),
        "hidden_channels": model.gat1.out_channels,
        "heads": model.gat1.heads,
        "dropout": model.dropout_rate,
        "node_feature_names": NODE_FEATURE_NAMES,
        "transaction_feature_names": FEATURE_NAMES,
        "edge_scaler_mean": edge_scaler.mean_.tolist(),
        "edge_scaler_scale": edge_scaler.scale_.tolist(),
        "seed": SEED,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    metadata = {
        "dataset": {
            "transactions": int(len(transactions)),
            "accounts": int(len(set(transactions.sender_id) | set(transactions.receiver_id))),
            "features": 16,
            "fraud_cases": int(transactions.fraud_label.sum()),
            "fraud_rate": float(transactions.fraud_label.mean()),
            "seed": SEED,
        },
        "splits": {
            name: {"transactions": int(len(split)), "fraud_cases": int(split.fraud_label.sum()),
                  "start_timestamp": int(split.timestamp.min()), "end_timestamp": int(split.timestamp.max())}
            for name, split in (("train", train_frame), ("validation", validation_frame), ("test", test_frame))
        },
        "model": {"architecture": "GATv2Conv(2 layers, 2 heads) + edge MLP", "epochs": epochs,
                  "class_weight": float(negative_count / positive_count)},
        "training_history": history,
        "test_metrics": test_metrics,
        "class_prior_test_metrics": prior_metrics,
        "test_predictions_file": str(predictions_path.name),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"class-prior test metrics: {json.dumps(prior_metrics, indent=2)}")
    print(f"Saved model to {model_path}")
    return metadata


if __name__ == "__main__":
    train_model()