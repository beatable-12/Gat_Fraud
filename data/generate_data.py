"""Generate reproducible PaySim-like UPI transactions with prefix-only features."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import ACCOUNT_COUNT, DEFAULT_TRANSACTIONS, SEED

FEATURE_NAMES = [
    "amount_log1p",
    "amount_over_sender_balance",
    "amount_over_receiver_balance",
    "sender_balance_log1p",
    "receiver_balance_log1p",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "sender_tx_count_1h",
    "sender_tx_count_24h",
    "receiver_tx_count_24h",
    "receiver_seen_before",
    "sender_receiver_tx_count_before",
    "amount_z_sender_history",
    "amount_z_global_history",
]


def generate_transactions(
    n_transactions: int = DEFAULT_TRANSACTIONS,
    n_accounts: int = ACCOUNT_COUNT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create ordered transfers; every engineered feature uses prior events only."""
    if n_transactions < 20:
        raise ValueError("n_transactions must be at least 20")
    if n_accounts < 4:
        raise ValueError("n_accounts must be at least 4")

    rng = np.random.default_rng(seed)
    accounts = [f"acct_{index:04d}" for index in range(n_accounts)]
    balances = defaultdict(lambda: 20_000.0)
    sender_events: dict[str, deque[int]] = defaultdict(deque)
    receiver_events: dict[str, deque[int]] = defaultdict(deque)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    receiver_seen: set[str] = set()
    sender_amounts: dict[str, list[float]] = defaultdict(list)
    global_amounts: list[float] = []
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    elapsed_seconds = 0
    records: list[dict[str, object]] = []

    for index in range(n_transactions):
        elapsed_seconds += int(rng.integers(20, 420))
        sender = str(rng.choice(accounts))
        if rng.random() < 0.08:
            receiver = str(rng.choice(accounts[: max(2, n_accounts // 12)]))
        else:
            receiver = str(rng.choice(accounts))
        while receiver == sender:
            receiver = str(rng.choice(accounts))

        sender_balance_before = balances[sender]
        receiver_balance_before = balances[receiver]
        if rng.random() < 0.035:
            amount = float(rng.uniform(1_200.0, 7_000.0))
        else:
            amount = float(rng.lognormal(mean=4.65, sigma=0.85))
        amount = min(amount, max(sender_balance_before * 0.85, 1.0))
        timestamp = int(base_time.timestamp()) + elapsed_seconds

        sender_history = sender_amounts[sender]
        global_history = global_amounts
        sender_mean = float(np.mean(sender_history)) if sender_history else 300.0
        sender_std = float(np.std(sender_history)) if len(sender_history) > 1 else 300.0
        global_mean = float(np.mean(global_history)) if global_history else 300.0
        global_std = float(np.std(global_history)) if len(global_history) > 1 else 300.0
        hour = (base_time + timedelta(seconds=elapsed_seconds)).hour
        weekday = (base_time + timedelta(seconds=elapsed_seconds)).weekday()

        while sender_events[sender] and sender_events[sender][0] <= timestamp - 86400:
            sender_events[sender].popleft()
        while receiver_events[receiver] and receiver_events[receiver][0] <= timestamp - 86400:
            receiver_events[receiver].popleft()

        features = [
            np.log1p(amount),
            amount / max(sender_balance_before, 1.0),
            amount / max(receiver_balance_before, 1.0),
            np.log1p(sender_balance_before),
            np.log1p(receiver_balance_before),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            float(sum(event_time > timestamp - 3600 for event_time in sender_events[sender])),
            float(len(sender_events[sender])),
            float(len(receiver_events[receiver])),
            float(receiver in receiver_seen),
            float(pair_counts[(sender, receiver)]),
            float((amount - sender_mean) / max(sender_std, 100.0)),
            float((amount - global_mean) / max(global_std, 100.0)),
        ]

        large_or_unusual = features[0] > np.log1p(1_500) or features[1] > 0.35
        rapid_new_receiver = features[9] >= 2 and features[12] == 0
        repeated_suspicious_pair = features[13] >= 2 and features[0] > np.log1p(600)
        risk_pattern = large_or_unusual or rapid_new_receiver or repeated_suspicious_pair
        fraud = int(risk_pattern and rng.random() < 0.58)
        if not risk_pattern and rng.random() < 0.0015:
            fraud = 1

        record: dict[str, object] = {
            "transaction_id": f"txn_{index:06d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "timestamp": timestamp,
            "amount": amount,
            "sender_balance_before": sender_balance_before,
            "sender_balance_after": sender_balance_before - amount,
            "receiver_balance_before": receiver_balance_before,
            "receiver_balance_after": receiver_balance_before + amount,
            "fraud_label": fraud,
        }
        record.update(dict(zip(FEATURE_NAMES, map(float, features), strict=True)))
        records.append(record)

        balances[sender] = sender_balance_before - amount
        balances[receiver] = receiver_balance_before + amount
        sender_events[sender].append(timestamp)
        receiver_events[receiver].append(timestamp)
        sender_amounts[sender].append(amount)
        global_amounts.append(amount)
        pair_counts[(sender, receiver)] += 1
        receiver_seen.add(receiver)

    frame = pd.DataFrame.from_records(records)
    return frame


def save_dataset(path: str | None = None, **kwargs: object) -> pd.DataFrame:
    """Generate and save the dataset, creating only the small output directory."""
    from config import DATASET_PATH

    output = DATASET_PATH if path is None else path
    frame = generate_transactions(**kwargs)
    frame.to_csv(output, index=False)
    return frame


if __name__ == "__main__":
    frame = save_dataset()
    print(f"Saved {len(frame):,} transactions to {__import__('config').DATASET_PATH}")
    print(f"Fraud cases: {int(frame['fraud_label'].sum()):,} ({frame['fraud_label'].mean():.2%})")