import torch
import numpy as np

from data.generate_data import generate_transactions
from explainability.explain import explain_prediction
from models.predict import build_custom_target, predict_sample, prepare_model_sample
from models.gatv2_model import GATv2EdgeClassifier


def test_gatv2_forward_shape_and_probabilities():
    model = GATv2EdgeClassifier()
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 2, 3], [1, 1, 4]], dtype=torch.long)
    logits = model(x, edge_index, torch.tensor([0, 3]), torch.tensor([1, 4]), torch.randn(2, 16))

    assert logits.shape == (2,)
    assert torch.all((torch.sigmoid(logits) >= 0) & (torch.sigmoid(logits) <= 1))


def test_custom_target_prediction_uses_checkpoint_feature_scaling():
    transactions = generate_transactions(n_transactions=60, n_accounts=14)
    sender = transactions.iloc[0].sender_id
    receiver = next(account for account in transactions.receiver_id.unique() if account != sender)
    custom = build_custom_target(transactions, sender, receiver, 900.0)
    checkpoint = {
        "edge_scaler_mean": np.zeros(16).tolist(),
        "edge_scaler_scale": np.ones(16).tolist(),
    }
    sample, graph = prepare_model_sample(custom, len(custom) - 1, checkpoint)
    probability = predict_sample(GATv2EdgeClassifier(), graph)

    assert sample.transaction_features.shape == (16,)
    assert graph.transaction_features.shape == (1, 16)
    assert 0.0 <= probability <= 1.0


def test_gnnexplainer_and_integrated_gradients_return_model_attributions():
    transactions = generate_transactions(n_transactions=50, n_accounts=12)
    checkpoint = {"edge_scaler_mean": np.zeros(16).tolist(), "edge_scaler_scale": np.ones(16).tolist()}
    sample, graph = prepare_model_sample(transactions, 40, checkpoint)
    explanation = explain_prediction(
        GATv2EdgeClassifier(), sample, graph, explainer_epochs=4, integrated_gradient_steps=4
    )

    assert len(explanation["features"]) == 16
    assert len(explanation["historical_edges"]) == len(sample.historical_edge_indices)
    assert "GNNExplainer" in explanation["method"]