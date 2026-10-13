import pandas as pd
import numpy as np
from io import StringIO

import numpy as np
import pytest

import app
from app import analyze_uploaded_transactions, infer_mapping, validate_and_map_uploaded_csv
from data.generate_data import FEATURE_NAMES
from graph.temporal_sampler import NODE_FEATURE_NAMES, TemporalNeighborSampler
from models.gatv2_model import GATv2EdgeClassifier
from models.predict import predict_sample, prepare_model_sample


def canonical_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["txn-a", "txn-b"],
            "sender_id": ["acct-a", "acct-b"],
            "receiver_id": ["acct-b", "acct-c"],
            "timestamp": [1_735_689_655, 1_735_690_018],
            "amount": ["19.5", "50.0"],
            "sender_balance_before": [20000.0, 20000.0],
            "sender_balance_after": [19980.5, 19950.0],
            "receiver_balance_before": [20000.0, 20000.0],
            "receiver_balance_after": [20019.5, 20050.0],
            "fraud_label": [0, 1],
            "merchant_category": ["retail", "utilities"],
        }
    )


def test_canonical_columns_auto_map_and_preserve_uploaded_data():
    frame = canonical_transactions()
    original = frame.copy(deep=True)
    mapping = infer_mapping(frame)

    assert mapping["sender_id"] == "sender_id"
    assert mapping["receiver_id"] == "receiver_id"
    assert mapping["timestamp"] == "timestamp"
    assert mapping["amount"] == "amount"
    assert mapping["fraud_label"] == "fraud_label"
    assert mapping["transaction_type"] is None

    prepared, issues = validate_and_map_uploaded_csv(frame, mapping)

    assert prepared["transaction_id"].tolist() == ["txn-a", "txn-b"]
    assert prepared["sender_id"].tolist() == ["acct-a", "acct-b"]
    assert prepared["receiver_id"].tolist() == ["acct-b", "acct-c"]
    assert prepared["fraud_label"].tolist() == [0, 1]
    assert prepared["merchant_category"].tolist() == ["retail", "utilities"]
    assert prepared["transaction_type"].tolist() == ["unknown", "unknown"]
    assert list(prepared.loc[:, FEATURE_NAMES].columns) == FEATURE_NAMES
    assert len(prepared) == len(frame)
    assert frame.equals(original)
    assert any("ground-truth fraud labels" in issue for issue in issues)


def test_paysim_columns_auto_map_and_normalize():
    frame = pd.DataFrame(
        {
            "nameOrig": ["acct-a", "acct-b"],
            "nameDest": ["acct-b", "acct-c"],
            "step": [1, 2],
            "amount": [19.5, 50.0],
            "type": ["TRANSFER", "PAYMENT"],
            "isFraud": [0, 1],
        }
    )
    mapping = infer_mapping(frame)

    assert mapping == {
        "sender_id": "nameOrig",
        "receiver_id": "nameDest",
        "timestamp": "step",
        "amount": "amount",
        "transaction_type": "type",
        "fraud_label": "isFraud",
    }

    prepared, _ = validate_and_map_uploaded_csv(frame, mapping)

    assert prepared["sender_id"].tolist() == ["acct-a", "acct-b"]
    assert prepared["receiver_id"].tolist() == ["acct-b", "acct-c"]
    assert prepared["timestamp"].tolist() == [1, 2]
    assert prepared["transaction_type"].tolist() == ["TRANSFER", "PAYMENT"]
    assert prepared["fraud_label"].tolist() == [0, 1]


def test_balance_columns_are_not_inferred_as_account_or_type_fields():
    frame = pd.DataFrame(
        {
            "sender_balance_before": [100.0],
            "receiver_balance_before": [100.0],
            "transaction_id": ["txn-a"],
        }
    )

    mapping = infer_mapping(frame)

    assert mapping["sender_id"] is None
    assert mapping["receiver_id"] is None
    assert mapping["transaction_type"] is None


def test_missing_or_unmapped_receiver_returns_readable_validation_error():
    frame = canonical_transactions()
    mapping = infer_mapping(frame)
    mapping["receiver_id"] = None

    with pytest.raises(ValueError, match="Receiver Id"):
        validate_and_map_uploaded_csv(frame, mapping)


def test_missing_required_source_column_returns_readable_validation_error():
    frame = canonical_transactions().drop(columns=["sender_id"])

    with pytest.raises(ValueError, match="Sender Id"):
        validate_and_map_uploaded_csv(frame, infer_mapping(frame))


def test_selected_source_must_exist_in_uploaded_frame():
    frame = canonical_transactions()
    mapping = infer_mapping(frame)
    mapping["receiver_id"] = "old_receiver_column"

    with pytest.raises(ValueError, match="not present in the uploaded CSV"):
        validate_and_map_uploaded_csv(frame, mapping)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("amount", ["10.0", "not-an-amount"], "Amount column has 1 missing or invalid value"),
        ("timestamp", [1_735_689_655, "not-a-time"], "Timestamp column contains 1 missing or non-numeric value"),
    ],
)
def test_invalid_required_values_fail_without_dropping_rows(column, values, message):
    frame = canonical_transactions()
    frame[column] = values

    with pytest.raises(ValueError, match=message):
        validate_and_map_uploaded_csv(frame, infer_mapping(frame))


def test_missing_sender_ids_are_not_stringified_or_fabricated():
    frame = canonical_transactions()
    frame.loc[0, "sender_id"] = None

    with pytest.raises(ValueError, match="sender id column has 1 missing or blank value"):
        validate_and_map_uploaded_csv(frame, infer_mapping(frame))


def test_uploaded_analysis_reuses_sampler_and_returns_one_valid_gat_score_per_row(monkeypatch):
    frame = canonical_transactions()
    prepared, _ = validate_and_map_uploaded_csv(frame, infer_mapping(frame))
    model = GATv2EdgeClassifier()
    checkpoint = {
        "transaction_feature_names": FEATURE_NAMES,
        "transaction_features": len(FEATURE_NAMES),
        "node_feature_names": NODE_FEATURE_NAMES,
        "node_features": len(NODE_FEATURE_NAMES),
        "edge_scaler_mean": [0.0] * len(FEATURE_NAMES),
        "edge_scaler_scale": [1.0] * len(FEATURE_NAMES),
    }
    original_sampler = TemporalNeighborSampler
    sampler_constructions = 0

    class CountingSampler(original_sampler):
        def __init__(self, *args, **kwargs):
            nonlocal sampler_constructions
            sampler_constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(app, "TemporalNeighborSampler", CountingSampler)
    progress_updates = []
    predictions = analyze_uploaded_transactions(
        prepared,
        model,
        checkpoint,
        threshold=0.5,
        batch_size=2,
        progress_callback=lambda message, fraction: progress_updates.append((message, fraction)),
    )

    assert sampler_constructions == 1
    assert len(predictions) == len(prepared)
    assert predictions["transaction_id"].tolist() == prepared["transaction_id"].tolist()
    assert predictions["ground_truth"].tolist() == prepared["fraud_label"].tolist()
    assert np.isfinite(predictions["fraud_probability"]).all()
    assert predictions["fraud_probability"].between(0, 1).all()
    assert predictions["predicted_label"].tolist() == (predictions["fraud_probability"] >= 0.5).astype(int).tolist()
    assert "fraud_label" not in predictions.columns
    assert progress_updates[-1] == ("Predictions ready", 1.0)
    exported = pd.read_csv(StringIO(predictions.to_csv(index=False)))
    assert len(exported) == len(prepared)
    assert exported["transaction_id"].tolist() == predictions["transaction_id"].tolist()
    assert exported["ground_truth"].tolist() == predictions["ground_truth"].tolist()

    model.eval()
    baseline_sampler = original_sampler(prepared, neighbors_per_endpoint=12)
    baseline_scores = [
        predict_sample(model, prepare_model_sample(prepared, index, checkpoint, sampler=baseline_sampler)[1])
        for index in range(len(prepared))
    ]
    assert predictions["fraud_probability"].to_numpy() == pytest.approx(baseline_scores, abs=1e-5)


def test_uploaded_analysis_reports_checkpoint_feature_order_mismatch():
    frame = canonical_transactions()
    prepared, _ = validate_and_map_uploaded_csv(frame, infer_mapping(frame))
    checkpoint = {
        "transaction_feature_names": list(reversed(FEATURE_NAMES)),
        "transaction_features": len(FEATURE_NAMES),
        "node_feature_names": NODE_FEATURE_NAMES,
        "node_features": len(NODE_FEATURE_NAMES),
        "edge_scaler_mean": [0.0] * len(FEATURE_NAMES),
        "edge_scaler_scale": [1.0] * len(FEATURE_NAMES),
    }

    with pytest.raises(RuntimeError, match="input validation failed: Checkpoint transaction feature names/order"):
        analyze_uploaded_transactions(prepared, GATv2EdgeClassifier(), checkpoint)