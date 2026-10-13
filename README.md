# GAT-Fraud: Functional Research Demonstration

A small, reproducible synthetic UPI-like transfer experiment using a temporal graph and a real PyTorch Geometric GATv2 edge classifier. The prototype is designed for local CPU runs; it does not reproduce the paper's millions of rows and its metrics are not paper results.

## Windows setup

Use Python 3.10 or newer (the checked environment is Python 3.13). In PowerShell, from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch Geometric's standard `GATv2Conv` works with the listed CPU PyTorch package; no CUDA extensions, database, credentials, API, or external service is needed.

## Train, test, and launch

```powershell
python -m models.train
python -m pytest -q
streamlit run app.py
```

The dashboard normally opens at `http://localhost:8501` (or the configured Streamlit port). Training is deterministic with seed 42, generates 2,400 transactions over 180 accounts when no CSV exists, performs 8 CPU epochs, evaluates chronologically, and saves:

## CSV upload workflow

Use the `CSV Upload` page in the Streamlit app to:

- upload a PaySim-style transaction CSV or custom transfer dataset,
- validate required columns (`sender_id`, `receiver_id`, `timestamp`, `amount`),
- map optional fields such as `transaction_type` and `fraud_label`,
- preview the parsed file,
- run the trained GATv2 risk analysis, and
- download the prediction CSV.

The uploader supports PaySim conventions such as `nameOrig`, `nameDest`, `step`, `amount`, `type`, and `isFraud` and keeps the model's output separate from any research-only reference metrics.

- `data/generated/transactions.csv`: synthetic dataset
- `artifacts/gatv2_model.pt`: trained model and feature-scaling parameters
- `artifacts/model_metadata.json`: actual dataset counts, splits, training history, and test metrics
- `artifacts/test_predictions.csv`: held-out test labels and model/class-prior scores

To regenerate all experiment artifacts, remove those generated files and rerun `python -m models.train`. To train a different scale or epoch count from Python, call `models.train.train_model(n_transactions=..., epochs=...)`.

## Architecture and temporal protocol

Each transaction is a directed edge from sender to receiver. Target inference samples up to 12 previous incident edges per endpoint, with the strict condition `historical_timestamp < target_timestamp`. The two-layer, multi-head `GATv2Conv` encoder builds historical account embeddings. The source embedding, destination embedding, and the target transaction's 16 standardized features are concatenated and passed to an MLP for fraud probability.

The chronological partition is 70% train, 15% validation, 15% test. Input standardization is fit on train rows only. Positive-class weighting is derived from train labels only. Validation sees the train prefix; test sampling uses the full stored history but strict timestamps ensure only the prefix before each target is available. Labels are not model inputs. Synthetic features are constructed online in timestamp order from past transactions only.

## Transaction feature dictionary

Exactly 16 columns are passed as target-edge input, in this order:

| # | Feature | Meaning |
|---|---|---|
| 1 | `amount_log1p` | Log-transformed transfer amount |
| 2 | `amount_over_sender_balance` | Amount divided by sender's pre-transfer balance |
| 3 | `amount_over_receiver_balance` | Amount divided by receiver's pre-transfer balance |
| 4 | `sender_balance_log1p` | Log sender balance before transfer |
| 5 | `receiver_balance_log1p` | Log receiver balance before transfer |
| 6 | `hour_sin` | Sine encoding of UTC hour |
| 7 | `hour_cos` | Cosine encoding of UTC hour |
| 8 | `weekday_sin` | Sine encoding of UTC weekday |
| 9 | `weekday_cos` | Cosine encoding of UTC weekday |
| 10 | `sender_tx_count_1h` | Sender's prior transfers in the preceding hour |
| 11 | `sender_tx_count_24h` | Sender's prior transfers in the preceding 24 hours |
| 12 | `receiver_tx_count_24h` | Receiver's prior incoming transfers in the preceding 24 hours |
| 13 | `receiver_seen_before` | Whether the receiver previously received a transfer |
| 14 | `sender_receiver_tx_count_before` | Prior transfers on this directed account pair |
| 15 | `amount_z_sender_history` | Amount deviation from sender's prior amount history |
| 16 | `amount_z_global_history` | Amount deviation from the global prior amount history |

The GAT node state uses 8 separate historical summary values (in/out degree, mean in/out amount, unique in/out counterparties, and prior balance snapshots); these are not additional target-transaction features.

## Explainability

The Explainability page runs PyG `GNNExplainer` over the sampled historical graph to produce node and edge masks. Integrated Gradients produces attributions for the target's 16 transaction features. Explanation optimization is interactive and can take a few seconds on CPU. Scores explain this model invocation and are not causal findings.

## Metrics and research references

Model Evaluation computes precision, recall, F1, ROC-AUC, average precision, confusion matrix, and the precision-recall curve from the saved chronological test predictions. Its class-prior baseline predicts the training fraud prevalence. The threshold slider recalculates threshold-dependent metrics.

The dashboard's Research Reference page separately labels the manuscript values supplied in the project brief as **Reported in IEEE draft**. Those figures are not loaded into the synthetic dataset, model, or prototype metrics. The workspace includes the referenced IEEE PDF; this functional demonstration preserves the requested reference figures as specified in the brief and does not claim to reproduce its full-scale experiments.

## Limitations

- Synthetic patterns are intentionally learnable and do not establish real-world UPI fraud performance.
- This is a small static batch experiment, not a latency or streaming benchmark.
- The test split is small and may contain few frauds; metrics can vary substantially if seed or sample size changes.
- Custom dashboard inference requires accounts already present in the synthetic graph and scores the next transfer after the stored history.
- The class-prior baseline is constant-score and therefore has ROC-AUC 0.5; at the default 0.5 decision threshold it predicts no fraud.
- No model calibration, production controls, human-review workflow, or financial safety guarantees are provided.