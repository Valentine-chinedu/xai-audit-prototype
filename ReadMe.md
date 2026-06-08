# XAI Reliability Audit Prototype

A Streamlit prototype for auditing the reliability of local explainability methods on tabular classification models. The app compares LIME and SHAP explanations using repeated-run stability, additive reconstruction fidelity, latency, and a combined explanation confidence score.

## Features

- Run audits on a synthetic demo dataset or an uploaded CSV.
- Train one of three supported classifiers:
  - `RandomForest`
  - `HistGradientBoosting`
  - `MLP`
- Compare local explanations from LIME and SHAP.
- Measure explanation stability with mean pairwise Jaccard overlap of Top-K features.
- Estimate fidelity with R2 reconstruction over held-out test instances.
- Track explanation latency and SHAP/LIME latency ratio.
- Flag low LIME vs SHAP Top-K overlap as explanation disagreement.
- Export audit logs, explanation tables, and diagnostic latency plots.

## Project Structure

```text
.
├── app.py                         # Streamlit app
├── audit/
│   ├── explainers.py              # LIME and SHAP wrapper functions
│   ├── logging_utils.py           # JSONL audit logging helpers
│   ├── metrics.py                 # Jaccard and fidelity metrics
│   ├── models.py                  # Model factory
│   └── scoring.py                 # Explanation confidence score
├── test_aopc.py                   # AOPC calculation smoke test
├── reproduce_lime.py              # LIME feature parsing reproduction script
├── reproduce_fidelity.py          # Fidelity masking strategy reproduction script
├── verify_xgboost_replacement.py  # HistGradientBoosting/SHAP compatibility check
└── requirements.txt               # Python dependencies
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

The app opens in your browser. From the sidebar, choose a model, optionally upload a CSV, select the target column, adjust audit settings, and click `Run Reliability Audit`.

## Data

If no CSV is uploaded, the app uses a synthetic binary classification dataset generated with scikit-learn. For uploaded CSV files, the app:

- Drops common index-like columns.
- Infers a likely target column, which can be changed in the sidebar.
- Keeps numeric features.
- One-hot encodes low-cardinality categorical features.
- Drops high-cardinality non-numeric features.
- Performs simple missing-value handling.

## Audit Outputs

Each completed audit run is appended to:

```text
outputs/runs.jsonl
```

The app also provides download buttons for:

- Full audit log as JSONL.
- Full audit log as CSV.
- LIME explanation CSV.
- SHAP explanation CSV when available.
- High-resolution latency plot PNG.

## Notes

- Top-K is fixed at `5` in `app.py` for evaluation consistency.
- Tree models use TreeSHAP where supported.
- MLP uses LIME by default; KernelSHAP can be enabled from the sidebar, but it is slower.
- The explanation confidence score combines stability, fidelity, and latency into a value between `0` and `1`.

## Validation Scripts

Run the available smoke checks directly:

```bash
python test_aopc.py
python reproduce_lime.py
python reproduce_fidelity.py
python verify_xgboost_replacement.py
```
