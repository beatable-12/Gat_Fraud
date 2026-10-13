import pandas as pd

from data.generate_data import FEATURE_NAMES, generate_transactions
from graph.temporal_sampler import TemporalNeighborSampler


def test_sampler_never_includes_same_time_or_future_edges():
    transactions = generate_transactions(n_transactions=80, n_accounts=16)
    target = transactions.iloc[40].copy()
    same_time = transactions.iloc[0].copy()
    same_time["sender_id"] = target.sender_id
    same_time["timestamp"] = target.timestamp
    transactions = pd.concat([transactions.iloc[:40], pd.DataFrame([same_time]), transactions.iloc[40:]], ignore_index=True)
    target_index = 41

    sample = TemporalNeighborSampler(transactions, neighbors_per_endpoint=100).sample(target_index)

    timestamps = transactions.iloc[sample.historical_edge_indices]["timestamp"]
    assert all(timestamps < sample.target_timestamp)
    assert len(sample.transaction_features) == 16
    assert sample.transaction_features.shape[0] == len(FEATURE_NAMES)


def test_sample_has_both_target_endpoints_and_valid_edges():
    transactions = generate_transactions(n_transactions=80, n_accounts=16)
    target = transactions.iloc[50]

    sample = TemporalNeighborSampler(transactions).sample(50)

    assert sample.node_ids[sample.source_index] == target.sender_id
    assert sample.node_ids[sample.destination_index] == target.receiver_id
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_index.numel() == 0 or int(sample.edge_index.max()) < len(sample.node_ids)