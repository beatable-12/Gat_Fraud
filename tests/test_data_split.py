from data.generate_data import FEATURE_NAMES, generate_transactions
from data.preprocess import chronological_split, feature_matrix


def test_split_is_chronological_and_disjoint():
    frame = generate_transactions(n_transactions=100, n_accounts=20)
    train, validation, test = chronological_split(frame)

    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_feature_contract_is_exactly_sixteen_columns():
    frame = generate_transactions(n_transactions=40, n_accounts=10)

    assert len(FEATURE_NAMES) == 16
    assert feature_matrix(frame).shape == (40, 16)
    assert (frame["sender_tx_count_24h"] >= frame["sender_tx_count_1h"]).all()