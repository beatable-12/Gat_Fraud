"""GATv2 edge classifier combining historical endpoint context and edge features."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from config import ATTENTION_HEADS, HIDDEN_CHANNELS
from graph.temporal_sampler import NODE_FEATURE_NAMES
from data.generate_data import FEATURE_NAMES


class GATv2EdgeClassifier(nn.Module):
    def __init__(
        self,
        node_features: int = len(NODE_FEATURE_NAMES),
        transaction_features: int = len(FEATURE_NAMES),
        hidden_channels: int = HIDDEN_CHANNELS,
        heads: int = ATTENTION_HEADS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.transaction_features = transaction_features
        self.dropout_rate = dropout
        self.gat1 = GATv2Conv(node_features, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + transaction_features, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        source_index: torch.Tensor | int,
        destination_index: torch.Tensor | int,
        transaction_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.gat1(x, edge_index).relu()
        x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat2(x, edge_index).relu()
        source_index = torch.as_tensor(source_index, dtype=torch.long, device=x.device).reshape(-1)
        destination_index = torch.as_tensor(destination_index, dtype=torch.long, device=x.device).reshape(-1)
        transaction_features = transaction_features.to(device=x.device, dtype=x.dtype)
        if transaction_features.ndim == 1:
            transaction_features = transaction_features.unsqueeze(0)
        if source_index.numel() != transaction_features.shape[0] or destination_index.numel() != transaction_features.shape[0]:
            raise ValueError("Endpoint index counts must match the transaction feature batch")
        edge_representation = torch.cat(
            (x[source_index], x[destination_index], transaction_features), dim=1
        )
        return self.edge_mlp(edge_representation).squeeze(-1)


def model_from_checkpoint(checkpoint: dict[str, object]) -> GATv2EdgeClassifier:
    model = GATv2EdgeClassifier(
        node_features=int(checkpoint["node_features"]),
        transaction_features=int(checkpoint["transaction_features"]),
        hidden_channels=int(checkpoint["hidden_channels"]),
        heads=int(checkpoint["heads"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model