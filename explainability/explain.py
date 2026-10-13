"""GNNExplainer neighborhood masks and Integrated Gradients edge attributions."""

from __future__ import annotations

import torch
from torch_geometric.explain import Explainer, GNNExplainer

from data.generate_data import FEATURE_NAMES
from graph.temporal_sampler import NODE_FEATURE_NAMES
from models.gatv2_model import GATv2EdgeClassifier


def explain_prediction(
    model: GATv2EdgeClassifier,
    sample,
    graph,
    explainer_epochs: int = 35,
    integrated_gradient_steps: int = 32,
) -> dict[str, object]:
    """Return model-generated input-feature scores and historical graph masks."""
    model.eval()
    x = graph.x.detach()
    edge_index = graph.edge_index
    source_index = graph.source_index
    destination_index = graph.destination_index
    edge_features = graph.transaction_features.detach()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explainer_epochs, lr=0.01),
        explanation_type="model",
        model_config={"mode": "regression", "task_level": "graph", "return_type": "raw"},
        node_mask_type="attributes",
        edge_mask_type="object",
    )
    explanation = explainer(
        x,
        edge_index,
        source_index=source_index,
        destination_index=destination_index,
        transaction_features=edge_features,
    )

    baseline = torch.zeros_like(edge_features)
    accumulated_gradient = torch.zeros_like(edge_features)
    for step in range(1, integrated_gradient_steps + 1):
        alpha = step / integrated_gradient_steps
        interpolated = (baseline + alpha * (edge_features - baseline)).detach().requires_grad_(True)
        logit = model(x, edge_index, source_index, destination_index, interpolated)
        gradient = torch.autograd.grad(logit.sum(), interpolated)[0]
        accumulated_gradient += gradient
    integrated_gradients = ((edge_features - baseline) * accumulated_gradient / integrated_gradient_steps)
    feature_scores = integrated_gradients.abs().reshape(-1).cpu().tolist()
    ranked_features = sorted(
        ({"feature": name, "score": float(score)} for name, score in zip(FEATURE_NAMES, feature_scores, strict=True)),
        key=lambda item: item["score"],
        reverse=True,
    )

    node_mask = explanation.node_mask.detach().abs()
    node_importance = node_mask.mean(dim=1).cpu().tolist()
    edge_mask = explanation.edge_mask.detach().abs().cpu().tolist()
    ranked_nodes = sorted(
        ({"account": account, "score": float(score)} for account, score in zip(sample.node_ids, node_importance, strict=True)),
        key=lambda item: item["score"],
        reverse=True,
    )
    ranked_edges = sorted(
        (
            {
                "transaction_index": int(transaction_index),
                "sender": sample.node_ids[int(edge_index[0, local_index])],
                "receiver": sample.node_ids[int(edge_index[1, local_index])],
                "score": float(edge_mask[local_index]),
            }
            for local_index, transaction_index in enumerate(sample.historical_edge_indices)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "features": ranked_features,
        "nodes": ranked_nodes,
        "historical_edges": ranked_edges,
        "node_feature_names": NODE_FEATURE_NAMES,
        "method": "PyG GNNExplainer neighborhood masks + Integrated Gradients for target edge features",
    }