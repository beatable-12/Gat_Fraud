"""Two-endpoint historical neighborhood sampler for target transactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.generate_data import FEATURE_NAMES

NODE_FEATURE_NAMES = [
    "historical_out_degree",
    "historical_in_degree",
    "historical_mean_out_amount",
    "historical_mean_in_amount",
    "historical_unique_receivers",
    "historical_unique_senders",
    "historical_sender_balance",
    "historical_receiver_balance",
]


@dataclass
class TemporalSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    source_index: int
    destination_index: int
    transaction_features: torch.Tensor
    label: torch.Tensor
    node_ids: list[str]
    historical_edge_indices: list[int]
    target_index: int
    target_timestamp: int


class TemporalNeighborSampler:
    """Sample only historical interactions incident to either target endpoint."""

    def __init__(self, transactions: pd.DataFrame, neighbors_per_endpoint: int = 12) -> None:
        if neighbors_per_endpoint < 1:
            raise ValueError("neighbors_per_endpoint must be positive")
        self.transactions = transactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
        self.neighbors_per_endpoint = neighbors_per_endpoint
        self._timestamps = self.transactions["timestamp"].to_numpy(dtype=np.int64, copy=True)
        self._sender_ids = self.transactions["sender_id"].astype(str).to_numpy()
        self._receiver_ids = self.transactions["receiver_id"].astype(str).to_numpy()
        self._amounts = self.transactions["amount"].to_numpy(dtype=np.float64, copy=True)
        self._sender_balances = self.transactions["sender_balance_before"].to_numpy(dtype=np.float64, copy=True)
        self._receiver_balances = self.transactions["receiver_balance_before"].to_numpy(dtype=np.float64, copy=True)
        self._transaction_features = self.transactions.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32, copy=True)
        self._labels = pd.to_numeric(self.transactions["fraud_label"], errors="coerce").to_numpy(dtype=np.float32)
        self.account_ids = sorted(set(self._sender_ids) | set(self._receiver_ids))
        self.account_to_index = {account: index for index, account in enumerate(self.account_ids)}
        account_edge_lists: dict[str, list[int]] = {account: [] for account in self.account_ids}
        for edge_index, (source, destination) in enumerate(zip(self._sender_ids, self._receiver_ids, strict=True)):
            account_edge_lists[source].append(edge_index)
            account_edge_lists[destination].append(edge_index)
        self._account_edges = {
            account: np.asarray(edge_indices, dtype=np.int64)
            for account, edge_indices in account_edge_lists.items()
        }
        self._account_edge_timestamps = {
            account: self._timestamps[edge_indices]
            for account, edge_indices in self._account_edges.items()
        }

    def sample(self, target_index: int) -> TemporalSample:
        if target_index < 0 or target_index >= len(self.transactions):
            raise IndexError("target_index is outside the transaction sequence")
        source = self._sender_ids[target_index]
        destination = self._receiver_ids[target_index]
        target_timestamp = int(self._timestamps[target_index])

        endpoint_edges: set[int] = set()
        for account in (source, destination):
            account_edges = self._account_edges[account]
            prior_count = int(np.searchsorted(self._account_edge_timestamps[account], target_timestamp, side="left"))
            first_neighbor = max(0, prior_count - self.neighbors_per_endpoint)
            endpoint_edges.update(account_edges[first_neighbor:prior_count].tolist())

        historical_indices = sorted(endpoint_edges)
        node_ids = [source, destination]
        for edge_index in historical_indices:
            edge = self.transactions.iloc[edge_index]
            for account in (str(edge.sender_id), str(edge.receiver_id)):
                if account not in node_ids:
                    node_ids.append(account)
        node_to_local = {account: index for index, account in enumerate(node_ids)}

        node_values: dict[str, list[float]] = {
            account: [0.0] * len(NODE_FEATURE_NAMES) for account in node_ids
        }
        outgoing_amounts: dict[str, list[float]] = {account: [] for account in node_ids}
        incoming_amounts: dict[str, list[float]] = {account: [] for account in node_ids}
        outgoing_neighbors: dict[str, set[str]] = {account: set() for account in node_ids}
        incoming_neighbors: dict[str, set[str]] = {account: set() for account in node_ids}
        latest_sender_balance: dict[str, float] = {}
        latest_receiver_balance: dict[str, float] = {}
        edge_pairs: list[tuple[int, int]] = []

        for edge_index in historical_indices:
            edge_source = self._sender_ids[edge_index]
            edge_destination = self._receiver_ids[edge_index]
            amount = float(self._amounts[edge_index])
            outgoing_amounts[edge_source].append(amount)
            incoming_amounts[edge_destination].append(amount)
            outgoing_neighbors[edge_source].add(edge_destination)
            incoming_neighbors[edge_destination].add(edge_source)
            latest_sender_balance[edge_source] = float(self._sender_balances[edge_index])
            latest_receiver_balance[edge_destination] = float(self._receiver_balances[edge_index])
            edge_pairs.append((node_to_local[edge_source], node_to_local[edge_destination]))

        node_rows: list[list[float]] = []
        for account in node_ids:
            out_amounts = outgoing_amounts[account]
            in_amounts = incoming_amounts[account]
            node_rows.append(
                [
                    float(np.log1p(len(out_amounts))),
                    float(np.log1p(len(in_amounts))),
                    float(np.log1p(np.mean(out_amounts))) if out_amounts else 0.0,
                    float(np.log1p(np.mean(in_amounts))) if in_amounts else 0.0,
                    float(np.log1p(len(outgoing_neighbors[account]))),
                    float(np.log1p(len(incoming_neighbors[account]))),
                    float(np.log1p(latest_sender_balance.get(account, 0.0))),
                    float(np.log1p(latest_receiver_balance.get(account, 0.0))),
                ]
            )

        edge_index_tensor = (
            torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            if edge_pairs
            else torch.empty((2, 0), dtype=torch.long)
        )
        return TemporalSample(
            x=torch.tensor(node_rows, dtype=torch.float32),
            edge_index=edge_index_tensor,
            source_index=node_to_local[source],
            destination_index=node_to_local[destination],
            transaction_features=torch.from_numpy(self._transaction_features[target_index].copy()),
            label=torch.tensor(float(self._labels[target_index]), dtype=torch.float32),
            node_ids=node_ids,
            historical_edge_indices=historical_indices,
            target_index=target_index,
            target_timestamp=target_timestamp,
        )