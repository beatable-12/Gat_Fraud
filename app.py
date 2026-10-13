"""Streamlit UI for the trained, small-scale GAT-Fraud demo and CSV upload analysis workflow."""

from __future__ import annotations

import json
import hashlib
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_curve

from config import BATCH_SIZE, DATASET_PATH, METADATA_PATH, MODEL_PATH, PREDICTIONS_PATH
from data.generate_data import FEATURE_NAMES
from evaluation.metrics import calculate_metrics
from explainability.explain import explain_prediction
from models.predict import (
    build_custom_target,
    load_checkpoint,
    load_transactions,
    predict_sample,
    prepare_model_sample,
)
from graph.temporal_sampler import NODE_FEATURE_NAMES, TemporalNeighborSampler

logger = logging.getLogger(__name__)

PAYSIM_COLUMN_MAP = {
    "sender_id": ["sender_id", "nameorig", "sender", "from_account", "source", "payer", "account_from", "source_account", "originator_id"],
    "receiver_id": ["receiver_id", "namedest", "receiver", "to_account", "destination", "payee", "account_to", "destination_account", "beneficiary_id"],
    "timestamp": ["timestamp", "step", "date", "time", "transaction_time", "created_at"],
    "amount": ["amount", "value", "transaction_amount", "transfer_amount"],
    "transaction_type": ["transaction_type", "type", "payment_type", "channel"],
    "fraud_label": ["fraud_label", "isfraud", "fraud", "label", "is_fraud"],
}

st.set_page_config(page_title="GAT-Fraud | Intelligent Transaction Risk Analysis", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --bg: #071827;
      --bg-2: #0d1f31;
      --panel: #11263d;
      --panel-2: #183452;
      --panel-3: #0d1d2d;
      --line: rgba(127, 164, 215, 0.25);
      --text: #edf6ff;
      --muted: #9bb4d1;
      --primary: #4d8df7;
      --primary-2: #86b7ff;
      --success: #4fd1a5;
      --danger: #ff6d74;
      --warning: #f4b25d;
      --shadow: rgba(9, 15, 24, 0.45);
    }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--text); }
    .stApp { background: linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%); }
    [data-testid="stSidebar"] { background: rgba(11, 22, 35, 0.98); border-right: 1px solid var(--line); }
    header[data-testid="stHeader"] { background-color: var(--bg) !important; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] { background-color: transparent !important; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] button { color: var(--text) !important; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] button:hover { background-color: rgba(134, 183, 255, 0.12) !important; }
    header[data-testid="stHeader"] [data-testid="stToolbar"] button svg { color: var(--text) !important; fill: currentColor; }
    .stApp [data-testid="stMetricValue"], h1, h2, h3, h4 { color: var(--text) !important; }
    .stApp .stMarkdown p, .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #c5d5e4 !important; }
    div[data-testid="stDataFrame"] { border: 1px solid #7990a6; border-radius: 6px; }
    .eyebrow { font: 600 11px 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.12em; color: var(--primary-2); }
    .stat { background: linear-gradient(180deg, var(--panel) 0%, rgba(18, 37, 56, 0.95) 100%); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; min-height: 92px; box-shadow: 0 8px 18px var(--shadow); }
    .stat-label { color: var(--muted); font-size: 12px; letter-spacing: 0.03em; }
    .stat-value { color: var(--text); font-size: 23px; font-weight: 800; margin-top: 5px; }
    .note { padding: 12px 14px; background: rgba(77, 141, 247, 0.10); border-left: 3px solid var(--primary); border-radius: 8px; color: var(--text); font-size: 13px; }
    .paper-note { padding: 12px 14px; background: rgba(244, 178, 93, 0.12); border-left: 3px solid var(--warning); border-radius: 8px; color: var(--text); font-size: 13px; }
    .mono { font-family: 'DM Mono', monospace; }
    div[data-testid="stFileUploaderDropzone"] { border: 1px dashed rgba(134, 183, 255, 0.4); background: rgba(17, 38, 61, 0.9); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    logger.info("Loading GATv2 checkpoint from %s", MODEL_PATH)
    try:
        model, checkpoint = load_checkpoint(MODEL_PATH)
    except Exception:
        logger.exception("GATv2 checkpoint loading failed")
        raise
    logger.info(
        "Loaded GATv2 checkpoint: node_features=%s transaction_features=%s device=%s",
        checkpoint.get("node_features"),
        checkpoint.get("transaction_features"),
        next(model.parameters()).device,
    )
    return model, checkpoint


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_transactions(DATASET_PATH)


@st.cache_data
def get_metadata() -> dict[str, object]:
    with Path(METADATA_PATH).open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@st.cache_data
def get_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


def metric_tile(label: str, value: str) -> None:
    st.markdown(
        f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame.reset_index(drop=True), hide_index=True, width="stretch")


def normalize_field_name(column_name: object) -> str:
    return str(column_name).strip().lower().replace(" ", "_").replace("-", "_")


def infer_mapping(frame: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_field_name(column): column for column in frame.columns}
    mapping: dict[str, str | None] = {}
    for canonical, aliases in PAYSIM_COLUMN_MAP.items():
        for alias in aliases:
            normalized_alias = normalize_field_name(alias)
            if normalized_alias in normalized:
                mapping[canonical] = normalized[normalized_alias]
                break
        else:
            mapping[canonical] = None
    return mapping


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().any():
        if series.isna().any():
            invalid_count = int(series.isna().sum())
            raise ValueError(f"Timestamp column contains {invalid_count} missing or non-numeric value(s). Correct them before analysis; no rows were dropped.")
        if not np.isfinite(series.to_numpy(dtype=np.float64)).all():
            raise ValueError("Timestamp column contains non-finite values. Correct them before analysis; no rows were dropped.")
        try:
            return series.astype(np.int64)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("Timestamp values are outside the supported numeric range.") from error
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise ValueError(f"Timestamp column contains {invalid_count} invalid date/time value(s). Correct them before analysis; no rows were dropped.")
    return parsed.map(lambda value: int(value.timestamp()))


def _compute_uploaded_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    balances: dict[str, float] = defaultdict(float)
    sender_events: dict[str, deque[int]] = defaultdict(deque)
    receiver_events: dict[str, deque[int]] = defaultdict(deque)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    receiver_seen: set[str] = set()
    sender_amounts: dict[str, list[float]] = defaultdict(list)
    global_amounts: list[float] = []

    for _, row in frame.sort_values("timestamp", kind="stable").reset_index(drop=True).iterrows():
        sender = str(row["sender_id"])
        receiver = str(row["receiver_id"])
        timestamp = int(row["timestamp"])
        amount = float(row["amount"])
        sender_balance_before = balances.get(sender, 20_000.0)
        receiver_balance_before = balances.get(receiver, 20_000.0)

        sender_history = sender_amounts.get(sender, [])
        sender_mean = float(np.mean(sender_history)) if sender_history else 300.0
        sender_std = float(np.std(sender_history)) if len(sender_history) > 1 else 300.0
        global_mean = float(np.mean(global_amounts)) if global_amounts else 300.0
        global_std = float(np.std(global_amounts)) if len(global_amounts) > 1 else 300.0
        datetime_value = pd.to_datetime(timestamp, unit="s", utc=True)
        hour = datetime_value.hour
        weekday = datetime_value.weekday()

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

        record = row.to_dict()
        record.update({
            "transaction_id": str(row.get("transaction_id", f"txn_{len(records) + 1:06d}")),
            "sender_id": sender,
            "receiver_id": receiver,
            "timestamp": timestamp,
            "amount": amount,
            "sender_balance_before": sender_balance_before,
            "sender_balance_after": sender_balance_before - amount,
            "receiver_balance_before": receiver_balance_before,
            "receiver_balance_after": receiver_balance_before + amount,
            "fraud_label": row.get("fraud_label", np.nan),
            "transaction_type": row.get("transaction_type", "unknown"),
        })
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

    return pd.DataFrame.from_records(records)


def validate_and_map_uploaded_csv(frame: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, list[str]]:
    logger.info("Validating uploaded transaction CSV: rows=%d columns=%d", len(frame), len(frame.columns))
    issues: list[str] = []
    required = ["sender_id", "receiver_id", "timestamp", "amount"]
    unmapped_values = (None, "", "-- not mapped --")
    missing = [field for field in required if mapping.get(field) in unmapped_values]
    if missing:
        labels = ", ".join(field.replace("_", " ").title() for field in missing)
        raise ValueError(f"Map the required CSV column(s) before analysis: {labels}.")
    if frame.empty:
        raise ValueError("The uploaded CSV contains no transaction rows.")

    invalid_sources = [
        (field, source)
        for field, source in mapping.items()
        if source not in unmapped_values and source not in frame.columns
    ]
    if invalid_sources:
        field, source = invalid_sources[0]
        raise ValueError(
            f"The selected {field.replace('_', ' ')} column '{source}' is not present in the uploaded CSV. "
            "Choose a column from the current file."
        )

    if mapping["sender_id"] == mapping["receiver_id"]:
        raise ValueError("Sender ID and Receiver ID must use different CSV columns.")

    working = frame.copy(deep=True)
    for field in PAYSIM_COLUMN_MAP:
        source = mapping.get(field)
        if source not in unmapped_values:
            working[field] = frame[source].copy()
        elif field == "transaction_type":
            working[field] = "unknown"
        elif field == "fraud_label":
            working[field] = np.nan

    if "transaction_id" not in working.columns:
        working.insert(0, "transaction_id", [f"txn_{index + 1:06d}" for index in range(len(working))])

    for field in ("sender_id", "receiver_id"):
        values = working[field].astype("string").str.strip()
        invalid_count = int((values.isna() | values.eq("")).sum())
        if invalid_count:
            raise ValueError(f"The selected {field.replace('_', ' ')} column has {invalid_count} missing or blank value(s). Correct them before analysis; no rows were dropped.")
        working[field] = values.astype(str)
    if (working["sender_id"] == working["receiver_id"]).any():
        issues.append("Some records have the same sender and receiver account; those rows were kept but may not be useful for graph learning.")
    working["timestamp"] = parse_timestamp_series(working["timestamp"])
    amounts = pd.to_numeric(working["amount"], errors="coerce")
    invalid_amounts = amounts.isna() | ~np.isfinite(amounts.to_numpy(dtype=np.float64))
    if invalid_amounts.any():
        raise ValueError(f"The selected Amount column has {int(invalid_amounts.sum())} missing or invalid value(s). Correct them before analysis; no rows were dropped.")
    if (amounts < 0).any():
        raise ValueError("The selected Amount column contains negative values, which are not supported by this transaction model.")
    working["amount"] = amounts.astype(float)

    raw_labels = working["fraud_label"]
    normalized_labels = raw_labels.astype("string").str.strip().str.lower()
    label_values = normalized_labels.map({
        "0": 0, "1": 1, "true": 1, "false": 0, "fraud": 1,
        "legit": 0, "legitimate": 0, "nonfraud": 0, "not_fraud": 0,
    })
    label_values = label_values.fillna(pd.to_numeric(raw_labels, errors="coerce"))
    invalid_labels = raw_labels.notna() & ~label_values.isin([0, 1])
    if invalid_labels.any():
        issues.append(f"{int(invalid_labels.sum())} fraud label value(s) were unrecognized and will not be used for evaluation.")
    working["fraud_label"] = label_values.where(label_values.isin([0, 1]), np.nan)
    if working["fraud_label"].notna().any():
        issues.append(f"Detected {int(working['fraud_label'].notna().sum())} ground-truth fraud labels for evaluation.")

    working["transaction_type"] = working["transaction_type"].fillna("unknown").astype(str)

    logger.info("Preparing prefix-only transaction features for %d uploaded rows", len(working))
    prepared = _compute_uploaded_transaction_features(working)
    logger.info("Prepared %d uploaded transactions with %d model features", len(prepared), len(FEATURE_NAMES))
    return prepared, issues


def analyze_uploaded_transactions(
    frame: pd.DataFrame,
    model,
    checkpoint: dict[str, object],
    threshold: float = 0.5,
    progress_callback: Callable[[str, float], None] | None = None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    stage = "input validation"
    try:
        required_columns = {"transaction_id", "sender_id", "receiver_id", "timestamp", "amount", "fraud_label", *FEATURE_NAMES}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Prepared transactions are missing required model columns: {', '.join(missing_columns)}.")
        if frame.empty:
            raise ValueError("There are no prepared transactions to score.")
        if batch_size < 1:
            raise ValueError("Inference batch size must be positive.")

        expected_features = tuple(checkpoint.get("transaction_feature_names", ()))
        if expected_features != tuple(FEATURE_NAMES):
            raise ValueError("Checkpoint transaction feature names/order do not match this application.")
        if int(checkpoint.get("transaction_features", -1)) != len(FEATURE_NAMES):
            raise ValueError("Checkpoint transaction feature dimension does not match this application.")
        if tuple(checkpoint.get("node_feature_names", ())) != tuple(NODE_FEATURE_NAMES):
            raise ValueError("Checkpoint node feature names/order do not match this application.")
        if int(checkpoint.get("node_features", -1)) != len(NODE_FEATURE_NAMES):
            raise ValueError("Checkpoint node feature dimension does not match this application.")
        if getattr(model, "transaction_features", None) != len(FEATURE_NAMES):
            raise ValueError("Loaded GATv2 model does not accept the expected transaction feature dimension.")

        scaler_mean = np.asarray(checkpoint.get("edge_scaler_mean"), dtype=np.float32)
        scaler_scale = np.asarray(checkpoint.get("edge_scaler_scale"), dtype=np.float32)
        if scaler_mean.shape != (len(FEATURE_NAMES),) or scaler_scale.shape != (len(FEATURE_NAMES),):
            raise ValueError("Checkpoint feature scaler dimensions do not match the 16 transaction features.")
        if not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_scale).all() or (scaler_scale <= 0).any():
            raise ValueError("Checkpoint feature scaler contains invalid values.")

        prepared = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        transaction_count = len(prepared)
        logger.info(
            "Validated uploaded analysis input: rows=%d transaction_features=%d node_features=%d batch_size=%d",
            transaction_count,
            len(FEATURE_NAMES),
            len(NODE_FEATURE_NAMES),
            batch_size,
        )

        stage = "temporal graph construction and feature preparation"
        logger.info("Building one temporal sampler for %d uploaded transactions", transaction_count)
        sampler = TemporalNeighborSampler(prepared, neighbors_per_endpoint=12)
        batch_count = (transaction_count + batch_size - 1) // batch_size
        probabilities = np.empty(transaction_count, dtype=np.float64)

        stage = "GATv2 inference"
        model.eval()
        device = next(model.parameters()).device
        logger.info("Starting batched GATv2 inference: batches=%d device=%s", batch_count, device)
        logger.info("Starting graph and prefix-feature preparation in batches: batches=%d", batch_count)
        for batch_number, start in enumerate(range(0, transaction_count, batch_size), start=1):
            stop = min(start + batch_size, transaction_count)
            stage = "temporal graph construction and feature preparation"
            graph_samples = []
            for target_index in range(start, stop):
                _, graph = prepare_model_sample(prepared, target_index, checkpoint, sampler=sampler)
                if graph.x.ndim != 2 or graph.x.shape[1] != len(NODE_FEATURE_NAMES):
                    raise ValueError(f"Graph batch {batch_number} has an incompatible node feature shape {tuple(graph.x.shape)}.")
                if graph.transaction_features.shape != (1, len(FEATURE_NAMES)):
                    raise ValueError(f"Graph batch {batch_number} has an incompatible transaction feature shape {tuple(graph.transaction_features.shape)}.")
                if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.transaction_features).all():
                    raise ValueError(f"Graph batch {batch_number} contains non-finite model features.")
                graph_samples.append(graph)

            if progress_callback is not None:
                progress_callback(
                    f"Prepared temporal graphs: {stop:,}/{transaction_count:,}",
                    0.45 * stop / transaction_count,
                )

            stage = "GATv2 inference"
            batch = Batch.from_data_list(graph_samples).to(device)
            with torch.inference_mode():
                logits = model(
                    batch.x,
                    batch.edge_index,
                    batch.source_index,
                    batch.destination_index,
                    batch.transaction_features,
                )
                if logits.ndim != 1 or logits.numel() != stop - start:
                    raise ValueError(
                        f"GATv2 returned {logits.numel()} logits for {stop - start} transactions in batch {batch_number}."
                    )
                batch_probabilities = torch.sigmoid(logits).to(device="cpu", dtype=torch.float64).numpy()
            if not np.isfinite(batch_probabilities).all() or ((batch_probabilities < 0) | (batch_probabilities > 1)).any():
                raise ValueError(f"GATv2 produced an invalid probability in batch {batch_number}.")
            probabilities[start:stop] = batch_probabilities
            if progress_callback is not None:
                progress_callback(
                    f"GATv2 scored batch {batch_number:,}/{batch_count:,}",
                    0.45 + 0.53 * stop / transaction_count,
                )

        stage = "prediction conversion"
        logger.info("Converting %d validated logits into transaction risk scores", len(probabilities))
        if len(probabilities) != transaction_count:
            raise ValueError(f"Produced {len(probabilities)} scores for {transaction_count} input transactions.")
        result = prepared[
            ["transaction_id", "sender_id", "receiver_id", "timestamp", "amount", "transaction_type"]
        ].copy()
        result["fraud_probability"] = probabilities
        result["predicted_label"] = (probabilities >= threshold).astype(np.int64)
        result["ground_truth"] = prepared["fraud_label"].to_numpy(copy=True)
        logger.info(
            "Converted predictions: scored=%d suspicious=%d legitimate=%d",
            len(result),
            int(result["predicted_label"].sum()),
            int((result["predicted_label"] == 0).sum()),
        )
        if progress_callback is not None:
            progress_callback("Predictions ready", 1.0)
        return result.reset_index(drop=True)
    except Exception as error:
        logger.exception("Uploaded fraud analysis failed during %s", stage)
        raise RuntimeError(f"{stage} failed: {error}") from error


def render_overview(metadata: dict[str, object]) -> None:
    dataset = metadata["dataset"]
    st.markdown('<div class="eyebrow">Operational fraud intelligence</div>', unsafe_allow_html=True)
    st.title("GAT-Fraud | Intelligent Transaction Risk Analysis")
    st.write("A local transaction-risk dashboard built around a real temporal GATv2 edge classifier for account-to-account payment monitoring.")
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Transactions", "Accounts", "Fraud cases", "Model state"],
                                   [f"{dataset['transactions']:,}", f"{dataset['accounts']:,}",
                                    f"{dataset['fraud_cases']:,}", "Trained · CPU"], strict=True):
        with column:
            metric_tile(label, value)
    st.subheader("Workflow")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box,style="rounded,filled",fontname="Manrope",fillcolor="#11263d",fontcolor="#edf6ff",color="#4d8df7",margin="0.18,0.12"];
      edge [color="#86b7ff",penwidth=1.5];
      input [label="Upload or load\ntransaction CSV"]; history [label="Temporal account\ncontext sampling"]; gat [label="2-layer GATv2\nencoder"]; combine [label="Endpoint embeddings\n+ 16 edge features"]; score [label="Fraud probability\n+ risk flags"];
      input -> history -> gat -> combine -> score;
    }
    """)
    st.subheader("Model architecture")
    st.write("The live inference path uses the same underlying historical graph model: sender and receiver embeddings are sampled from earlier transactions, combined with the target transaction's 16 engineered features, and scored by a binary classification head.")
    st.markdown('<div class="note">The model uses strict temporal safeguards so only prior transactions can influence the current prediction. No future transaction or label information is used in the feature set.</div>', unsafe_allow_html=True)
    st.caption(f"Model checkpoint: {Path(MODEL_PATH).name} · Dataset file: {Path(DATASET_PATH).name}")
def render_detection(model, checkpoint, transactions: pd.DataFrame) -> None:
    st.title("Transaction detection")
    st.caption("Evaluate a prior transaction or a synthetic next-transfer candidate with the trained GATv2 model.")
    mode = st.radio("Inference input", ["Sample transaction", "Enter transaction details"], horizontal=True)
    work = transactions
    target_index = 0
    if mode == "Sample transaction":
        samples = transactions.copy()
        samples["display"] = samples.apply(
            lambda row: f"{row.transaction_id} · {row.sender_id} → {row.receiver_id} · ₹{row.amount:,.0f}", axis=1
        )
        selected = st.selectbox("Target transaction", samples.display.tolist(), index=min(len(samples) - 1, 2100))
        target_index = int(samples.index[samples.display == selected][0])
    else:
        account_ids = sorted(set(transactions.sender_id) | set(transactions.receiver_id))
        sender = st.selectbox("Sender account", account_ids, index=0)
        receiver = st.selectbox("Receiver account", [account for account in account_ids if account != sender], index=0)
        amount = st.number_input("Amount (₹)", min_value=1.0, max_value=100_000.0, value=1_250.0, step=100.0)
        work = build_custom_target(transactions, sender, receiver, amount)
        target_index = len(work) - 1
        st.caption("The custom transfer is scored from the historical prefix only; no future data is included.")

    threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
    sample, graph = prepare_model_sample(work, target_index, checkpoint)
    probability = predict_sample(model, graph)
    predicted_class = int(probability >= threshold)
    target = work.iloc[target_index]
    st.session_state["active_transactions"] = work
    st.session_state["active_target"] = target
    st.session_state["active_sample"] = sample
    st.session_state["active_graph"] = graph
    st.session_state["active_probability"] = probability
    st.session_state["active_target_index"] = target_index

    left, right, detail = st.columns([1, 1, 2])
    with left:
        metric_tile("Fraud probability", f"{probability:.1%}")
    with right:
        metric_tile("Predicted class", "SUSPICIOUS" if predicted_class else "LEGITIMATE")
    with detail:
        st.markdown(f"**{target.sender_id} → {target.receiver_id}** · ₹{target.amount:,.2f}")
        st.caption(f"Threshold {threshold:.0%} · Historical cutoff: {pd.to_datetime(sample.target_timestamp, unit='s', utc=True)}")
    st.caption("This score is generated from the trained local GATv2 model and should be interpreted as a risk signal, not a production payment decision.")


def render_graph() -> None:
    st.title("Transaction graph")
    if "active_sample" not in st.session_state:
        st.info("Run a transaction risk check first to initialize the graph view.")
        return
    sample = st.session_state["active_sample"]
    target = st.session_state["active_target"]
    cutoff = pd.to_datetime(sample.target_timestamp, unit="s", utc=True)
    st.markdown(f"**Temporal cutoff:** `{cutoff}` · only edges earlier than this instant are visible in the graph context.")
    st.plotly_chart(graph_figure(sample), width="stretch")
    st.caption(f"{len(sample.node_ids)} sampled accounts · {len(sample.historical_edge_indices)} historical edges · highlighted transfer is the target payment.")
    edge_table = st.session_state["active_transactions"].iloc[sample.historical_edge_indices][
        ["transaction_id", "sender_id", "receiver_id", "timestamp", "amount"]
    ].copy()
    edge_table["timestamp"] = pd.to_datetime(edge_table.timestamp, unit="s", utc=True)
    show_table(edge_table.sort_values("timestamp", ascending=False))


def render_explanation(model) -> None:
    st.title("Explainability")
    if "active_sample" not in st.session_state:
        st.info("Run a transaction risk check first to generate a model explanation.")
        return
    sample = st.session_state["active_sample"]
    graph = st.session_state["active_graph"]
    target = st.session_state["active_target"]
    st.write(f"Target `{target.transaction_id}` · fraud probability **{st.session_state['active_probability']:.1%}**")
    if st.button("Generate model explanation", type="primary"):
        with st.spinner("Optimizing GNNExplainer masks and computing feature attributions…"):
            st.session_state["active_explanation"] = explain_prediction(model, sample, graph)
            st.session_state["explanation_for"] = str(target.transaction_id)
    explanation = st.session_state.get("active_explanation")
    if explanation is None or st.session_state.get("explanation_for") != str(target.transaction_id):
        st.info("Generate an explanation to inspect the model's graph and feature attributions for this target.")
        return
    st.markdown(f'<div class="note">Model-generated explanation: {explanation["method"]}. Feature scores come from Integrated Gradients; node and edge scores come from GNNExplainer.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Target transaction features")
        features = pd.DataFrame(explanation["features"][:10])
        figure = go.Figure(go.Bar(x=features.score, y=features.feature, orientation="h", marker_color="#4d8df7"))
        figure.update_layout(height=380, yaxis={"autorange": "reversed", "tickfont": {"color": "#edf6ff"}}, margin={"l": 8, "r": 8, "t": 8, "b": 8}, paper_bgcolor="#0d1f31", plot_bgcolor="#0d1f31", font={"color": "#edf6ff"}, hoverlabel={"bgcolor": "#183452", "font": {"color": "#edf6ff"}})
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Historical graph relationships")
        show_table(pd.DataFrame(explanation["historical_edges"][:10]))
        st.caption("These are local graph-importance scores from the trained model, not causal assertions about fraud behavior.")
    st.subheader("Neighborhood mask")
    st.plotly_chart(graph_figure(sample, explanation["historical_edges"]), width="stretch")
    st.subheader("Important sampled accounts")
    show_table(pd.DataFrame(explanation["nodes"][:10]))

def render_evaluation(metadata: dict[str, object], transactions: pd.DataFrame) -> None:
    st.title("Model evaluation")
    st.caption("Prototype measurements on the saved chronological test split from the trained GATv2 checkpoint.")
    predictions = get_predictions()
    threshold = st.slider("Evaluation decision threshold", 0.01, 0.99, 0.50, 0.01, key="evaluation_threshold")
    labels = predictions.label.to_numpy(dtype=np.int64)
    model_metrics = calculate_metrics(labels, predictions.fraud_probability.to_numpy(), threshold)
    prior_metrics = calculate_metrics(labels, predictions.class_prior_probability.to_numpy(), threshold)
    st.markdown(f'<div class="note">Evaluated dataset: synthetic transaction graph · {model_metrics["support"]:,} rows · {model_metrics["fraud_cases"]:,} fraud cases.</div>', unsafe_allow_html=True)
    columns = st.columns(5)
    metric_names = [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc"), ("Average Precision", "average_precision")]
    for column, (label, key) in zip(columns, metric_names, strict=True):
        value = model_metrics[key]
        with column:
            metric_tile(label, f"{value:.4f}" if np.isfinite(value) else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        matrix = np.asarray(model_metrics["confusion_matrix"])
        figure = go.Figure(go.Heatmap(z=matrix, x=["Legitimate", "Suspicious"], y=["Legitimate", "Suspicious"], text=matrix, texttemplate="%{text}", textfont={"color": "#ffffff"}, colorscale=[[0, "#152d46"], [1, "#255891"]], showscale=False))
        figure.update_layout(
            height=330,
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            paper_bgcolor="#0d1f31",
            plot_bgcolor="#0d1f31",
            font={"family": "Manrope, sans-serif", "color": "#edf6ff", "size": 12},
            xaxis={"tickfont": {"color": "#edf6ff"}},
            yaxis={"tickfont": {"color": "#edf6ff"}},
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Precision-recall curve")
        precision, recall, _ = precision_recall_curve(labels, predictions.fraud_probability.to_numpy())
        figure = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line={"color": "#4d8df7", "width": 3}, name="GATv2"))
        figure.add_hline(y=float(labels.mean()), line_dash="dot", line_color="#f4b25d", annotation_text="Test fraud prevalence")
        figure.update_layout(
            xaxis={"title": {"text": "Recall", "font": {"color": "#edf6ff"}}, "range": [0, 1], "tickfont": {"color": "#edf6ff"}, "gridcolor": "#183452", "zerolinecolor": "#2d4d6f"},
            yaxis={"title": {"text": "Precision", "font": {"color": "#edf6ff"}}, "range": [0, 1], "tickfont": {"color": "#edf6ff"}, "gridcolor": "#183452", "zerolinecolor": "#2d4d6f"},
            height=330,
            margin={"l": 8, "r": 8, "t": 20, "b": 8},
            paper_bgcolor="#0d1f31",
            plot_bgcolor="#0d1f31",
            font={"family": "Manrope, sans-serif", "color": "#edf6ff", "size": 12},
        )
        st.plotly_chart(figure, width="stretch")

    st.subheader("Class-prior baseline comparison")
    metric_keys = [key for _, key in metric_names]
    comparison = pd.DataFrame([
        {"Model": "GATv2 edge classifier", **{key: model_metrics[key] for key in metric_keys}},
        {"Model": "Class prior (train fraud rate)", **{key: prior_metrics[key] for key in metric_keys}},
    ])
    show_table(comparison)
    st.caption(f"The class-prior baseline uses the training split only ({metadata['splits']['train']['fraud_cases']:,}/{metadata['splits']['train']['transactions']:,} fraud cases).")


def render_upload(model, checkpoint):
    st.title("CSV upload")
    st.write("Upload a transaction CSV, map the key columns, and run the trained GATv2 fraud-risk analysis on the file.")
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"], help="Accepted formats include PaySim-style rows and custom transaction histories.")
    if uploaded_file is None:
        st.info("No file selected yet. Upload a CSV to inspect its schema and map the required fields.")
        return

    upload_identity = (
        uploaded_file.name,
        uploaded_file.size,
        getattr(uploaded_file, "file_id", None),
    )
    if st.session_state.get("uploaded_file_identity") != upload_identity:
        for key in (
            "uploaded_prepared",
            "uploaded_prepared_signature",
            "uploaded_predictions",
            "uploaded_prediction_signature",
            "uploaded_issues",
            "uploaded_mapping_signature",
            "uploaded_prediction_search",
        ):
            st.session_state.pop(key, None)
        st.session_state["uploaded_file_identity"] = upload_identity

    try:
        frame = pd.read_csv(uploaded_file)
    except Exception:
        logger.exception("Could not parse the uploaded transaction CSV")
        st.error("Could not read this CSV. Check that it is a valid comma-separated file with a supported text encoding.")
        return

    st.subheader("Dataset summary")
    st.write(f"Rows: {len(frame):,} | Columns: {len(frame.columns):,} | Preview: {uploaded_file.name}")
    show_table(frame.head(10))

    inferred = infer_mapping(frame)
    st.subheader("Column mapping")
    st.caption("Required fields: sender/account ID, receiver/account ID, timestamp, and amount. Optional fields include transaction type and fraud label.")

    schema_signature = hashlib.blake2s(
        "\x1f".join(map(str, frame.columns)).encode("utf-8"), digest_size=6
    ).hexdigest()
    selected_mapping: dict[str, str | None] = {}
    columns = st.columns(2)
    for index, key in enumerate(["sender_id", "receiver_id", "timestamp", "amount", "transaction_type", "fraud_label"]):
        with columns[index % 2]:
            existing = inferred.get(key)
            options = ["-- not mapped --"] + list(frame.columns)
            default_index = options.index(existing) if existing in options else 0
            selected = st.selectbox(
                key.replace("_", " ").title(),
                options,
                index=default_index,
                key=f"map_{schema_signature}_{key}",
            )
            selected_mapping[key] = None if selected == "-- not mapped --" else selected

    mapping_signature = tuple((field, selected_mapping[field]) for field in PAYSIM_COLUMN_MAP)
    prepared_signature = (upload_identity, mapping_signature)
    if st.session_state.get("uploaded_mapping_signature") != mapping_signature:
        for key in ("uploaded_prepared", "uploaded_prepared_signature", "uploaded_predictions", "uploaded_prediction_signature", "uploaded_prediction_search"):
            st.session_state.pop(key, None)
        st.session_state["uploaded_mapping_signature"] = mapping_signature

    if st.button("Validate schema and prepare data", type="primary"):
        try:
            prepared, issues = validate_and_map_uploaded_csv(frame, selected_mapping)
            st.session_state["uploaded_prepared"] = prepared
            st.session_state["uploaded_prepared_signature"] = prepared_signature
            st.session_state["uploaded_issues"] = issues
            st.success("The uploaded file was validated successfully. The data is ready to score.")
            show_table(prepared.head(10))
            for issue in issues:
                st.warning(issue)
        except ValueError as error:
            st.warning(str(error))
            return
        except Exception:
            logger.exception("Unexpected error while validating uploaded transaction columns")
            st.error("The uploaded CSV could not be prepared. Review the column selections and values; technical details were logged.")
            return

    if "uploaded_prepared" not in st.session_state or st.session_state.get("uploaded_prepared_signature") != prepared_signature:
        return

    threshold = st.slider("Risk threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01, key="upload_threshold")
    prediction_signature = (prepared_signature, float(threshold))
    if st.session_state.get("uploaded_prediction_signature") != prediction_signature:
        st.session_state.pop("uploaded_predictions", None)
        st.session_state.pop("uploaded_prediction_signature", None)

    if st.button("Run Fraud Analysis"):
        st.session_state.pop("uploaded_predictions", None)
        st.session_state.pop("uploaded_prediction_signature", None)
        progress_bar = st.progress(0.0, text="Starting transaction analysis…")

        def report_progress(message: str, fraction: float) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)

        try:
            predictions = analyze_uploaded_transactions(
                st.session_state["uploaded_prepared"],
                model,
                checkpoint,
                threshold,
                progress_callback=report_progress,
            )
            st.session_state["uploaded_predictions"] = predictions
            st.session_state["uploaded_prediction_signature"] = prediction_signature
            logger.info("Stored %d uploaded transaction predictions in Streamlit session state", len(predictions))
            st.success(f"Scored {len(predictions):,} transactions and generated a prediction table.")
        except Exception as error:
            logger.exception("Uploaded transaction fraud analysis failed")
            st.error(f"Fraud analysis failed: {error}. No partial predictions were saved; technical details were logged.")
            return
        finally:
            progress_bar.empty()

    predictions = st.session_state.get("uploaded_predictions")
    if predictions is None:
        return

    logger.info("Rendering upload results: rows=%d threshold=%.4f", len(predictions), threshold)

    summary_columns = st.columns(5)
    flagged = int(predictions["predicted_label"].sum())
    legitimate = int((1 - predictions["predicted_label"]).sum())
    with summary_columns[0]:
        metric_tile("Transactions analyzed", f"{len(predictions):,}")
    with summary_columns[1]:
        metric_tile("Flagged as suspicious", f"{flagged:,}")
    with summary_columns[2]:
        metric_tile("Legitimate", f"{legitimate:,}")
    with summary_columns[3]:
        metric_tile("Flagged %", f"{(flagged / max(len(predictions), 1)):.2%}")
    with summary_columns[4]:
        metric_tile("Max risk score", f"{predictions['fraud_probability'].max():.2%}")

    st.subheader("Prediction table")
    display = predictions[["transaction_id", "sender_id", "receiver_id", "timestamp", "amount", "fraud_probability", "predicted_label"]].copy()
    display["timestamp"] = pd.to_datetime(display["timestamp"], unit="s", utc=True)
    display["risk_band"] = np.where(display["predicted_label"] == 1, "Suspicious", "Legitimate")
    search_query = st.text_input("Search transactions", key="uploaded_prediction_search")
    table_display = display
    if search_query.strip():
        query = search_query.strip()
        matches = display.astype("string").apply(
            lambda column: column.str.contains(query, case=False, regex=False, na=False)
        ).any(axis=1)
        table_display = display.loc[matches]
    st.caption(f"Showing {len(table_display):,} of {len(predictions):,} scored transactions.")
    try:
        show_table(table_display)
    except Exception:
        logger.exception("Could not render the uploaded prediction table")
        st.error("Predictions completed, but the results table could not be rendered. The full CSV download is still available.")

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", csv_bytes, file_name="gat_fraud_predictions.csv", mime="text/csv")

    label_values = pd.to_numeric(predictions["ground_truth"], errors="coerce")
    invalid_labels = label_values.notna() & ~label_values.isin([0, 1])
    valid_labels = label_values.isin([0, 1])
    if invalid_labels.any():
        logger.warning("Skipping %d non-binary ground-truth label values during uploaded evaluation", int(invalid_labels.sum()))
        st.warning(f"Skipped {int(invalid_labels.sum())} non-binary ground-truth label value(s) during evaluation.")
    if valid_labels.any():
        try:
            labels = label_values.loc[valid_labels].to_numpy(dtype=np.int64)
            score = predictions.loc[valid_labels, "fraud_probability"].to_numpy(dtype=np.float64)
            logger.info("Evaluating uploaded predictions against %d valid binary labels", len(labels))
            metrics = calculate_metrics(labels, score, threshold)
            logger.info("Uploaded evaluation completed: %s", {key: metrics[key] for key in ("precision", "recall", "f1", "roc_auc", "average_precision")})
            st.subheader("Evaluation on uploaded labels")
            metric_cards = st.columns(5)
            metric_names = [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1-score", "f1"),
                ("ROC-AUC", "roc_auc"),
                ("Average precision", "average_precision"),
            ]
            for column, (label, key) in zip(metric_cards, metric_names, strict=True):
                with column:
                    metric_tile(label, f"{metrics[key]:.4f}" if np.isfinite(metrics[key]) else "N/A")

            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            figure = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=["Legitimate", "Suspicious"],
                    y=["Legitimate", "Suspicious"],
                    text=matrix,
                    texttemplate="%{text}",
                    textfont={"color": "#ffffff"},
                    colorscale=[[0, "#152d46"], [1, "#255891"]],
                    showscale=False,
                )
            )
            figure.update_layout(
                title="Confusion matrix",
                height=320,
                margin={"l": 8, "r": 8, "t": 36, "b": 8},
                paper_bgcolor="#0d1f31",
                plot_bgcolor="#0d1f31",
                font={"color": "#edf6ff"},
                xaxis={"title": "Predicted label", "tickfont": {"color": "#edf6ff"}},
                yaxis={"title": "Ground-truth label", "tickfont": {"color": "#edf6ff"}},
            )
            st.plotly_chart(figure, width="stretch")
        except Exception as error:
            logger.exception("Uploaded evaluation failed")
            st.error(f"Predictions were generated, but evaluation failed: {error}")
    elif label_values.notna().any():
        st.warning("Ground-truth labels were present, but none were valid binary values (0 or 1); evaluation was skipped.")


def graph_figure(sample, highlight_edges: list[dict[str, object]] | None = None) -> go.Figure:
    graph = nx.DiGraph()
    graph.add_nodes_from(sample.node_ids)
    history = st.session_state.get("active_transactions")
    if history is not None:
        for edge_index in sample.historical_edge_indices:
            row = history.iloc[edge_index]
            graph.add_edge(str(row.sender_id), str(row.receiver_id))
    target_row = st.session_state.get("active_target")
    if target_row is not None:
        graph.add_edge(str(target_row.sender_id), str(target_row.receiver_id))
    positions = nx.spring_layout(graph, seed=17, k=0.9)
    figure = go.Figure()
    important = {(int(edge["transaction_index"]), float(edge["score"])) for edge in (highlight_edges or [])}
    score_by_index = {edge_index: score for edge_index, score in important}
    if history is not None:
        for edge_index in sample.historical_edge_indices:
            row = history.iloc[edge_index]
            source = str(row.sender_id)
            destination = str(row.receiver_id)
            source_x, source_y = positions[source]
            destination_x, destination_y = positions[destination]
            score = score_by_index.get(edge_index, 0.0)
            figure.add_annotation(
                x=destination_x, y=destination_y, ax=source_x, ay=source_y,
                xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                arrowhead=2, arrowsize=0.75, arrowwidth=1.0 + 3.0 * score,
                arrowcolor="#ff6d74" if score > 0.08 else "#7aa6d8",
                opacity=0.9,
            )
    if target_row is not None:
        source_x, source_y = positions[str(target_row.sender_id)]
        destination_x, destination_y = positions[str(target_row.receiver_id)]
        figure.add_annotation(
            x=destination_x, y=destination_y, ax=source_x, ay=source_y,
            xref="x", yref="y", axref="x", ayref="y", showarrow=True,
            arrowhead=3, arrowsize=1.2, arrowwidth=3.0, arrowcolor="#ff6d74",
        )
    source_id = str(target_row.sender_id) if target_row is not None else ""
    destination_id = str(target_row.receiver_id) if target_row is not None else ""
    figure.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in sample.node_ids],
            y=[positions[node][1] for node in sample.node_ids],
            mode="markers+text",
            text=[node if node in (source_id, destination_id) else "" for node in sample.node_ids],
            textposition="top center",
            marker={
                "size": [20 if node in (source_id, destination_id) else 11 for node in sample.node_ids],
                "color": ["#ff6d74" if node == source_id else "#4fd1a5" if node == destination_id else "#9bb4d1" for node in sample.node_ids],
                "line": {"color": "#071827", "width": 1.5},
            },
            hovertext=sample.node_ids,
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="#0d1f31",
        plot_bgcolor="#0d1f31",
        font={"color": "#edf6ff"},
        hoverlabel={"bgcolor": "#183452", "font": {"color": "#edf6ff"}},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def main() -> None:
    required = [MODEL_PATH, DATASET_PATH, METADATA_PATH, PREDICTIONS_PATH]
    missing = [path for path in required if not Path(path).is_file()]
    with st.sidebar:
        st.markdown('<div class="eyebrow">GAT-Fraud</div>', unsafe_allow_html=True)
        page = st.radio(
            "Workspace",
            ["Overview", "CSV Upload", "Transaction Detection", "Transaction Graph", "Explainability", "Model Evaluation"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Local transaction-risk intelligence")

    if missing:
        st.error("The trained model or generated experiment files are missing. Train the model first with `python -m models.train`.")
        st.code("python -m models.train\nstreamlit run app.py")
        return

    try:
        model, checkpoint = get_model()
        transactions = get_data()
        metadata = get_metadata()
    except Exception as error:
        st.error(f"Could not load the trained demonstration artifacts: {error}")
        return

    if page == "Overview":
        render_overview(metadata)
    elif page == "CSV Upload":
        render_upload(model, checkpoint)
    elif page == "Transaction Detection":
        render_detection(model, checkpoint, transactions)
    elif page == "Transaction Graph":
        render_graph()
    elif page == "Explainability":
        render_explanation(model)
    elif page == "Model Evaluation":
        render_evaluation(metadata, transactions)


if __name__ == "__main__":
    main()