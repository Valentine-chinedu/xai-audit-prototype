import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from audit.models import build_model
from audit.explainers import (
    lime_explain_instance,
    shap_explain_instance_tree,
    shap_explain_instance_kernel,
    topk_from_lime,
)
from audit.metrics import mean_pairwise_jaccard, fidelity_r2
from audit.scoring import explanation_confidence_score
from audit.logging_utils import append_jsonl

st.set_page_config(page_title="XAI Reliability Audit", layout="wide")
st.title("XAI Reliability Audit Prototype")

# --- Sidebar controls ---
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox("Model", ["RandomForest", "XGBoost", "MLP"])  
# Option for non-tree explainability
use_kernel_shap = st.sidebar.checkbox("Enable KernelSHAP for non-tree models (slow)", value=False)
if use_kernel_shap:
    kernel_shap_nsamples = st.sidebar.slider("KernelSHAP nsamples", 25, 1000, 100, step=25)
else:
    kernel_shap_nsamples = None
num_features = st.sidebar.slider("Top-K features", 5, 20, 10)
lime_runs = st.sidebar.slider("LIME repeated runs", 3, 30, 10)
latency_target = st.sidebar.number_input("Real-time latency target (ms)", 50, 2000, 200)

seed = st.sidebar.number_input("Base random seed", 0, 9999, 42)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])


@st.cache_data
def load_demo_data():
    # MVP demo: synthetic dataset if no CSV is provided
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df, cols, "target"


def clean_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop common junk index columns
    junk_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)

    # Also drop a pure index column if it looks like 0..n-1
    # (only if it is numeric and unique)
    for c in df.columns[:2]:  # only check first couple columns to avoid surprises
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].is_unique:
            vals = df[c].values
            if len(vals) > 3 and np.all(vals[:3] == np.array([0, 1, 2])):  # quick heuristic
                df = df.drop(columns=[c])
                break

    return df


def preprocess_for_modeling(df: pd.DataFrame, target_col: str):
    """Prepare dataframe for modeling:
    - Convert target to numeric when it's clearly continuous
    - Drop rows with missing targets
    - Drop columns with excessive missing values
    - Keep numeric features; one-hot encode low-cardinality categoricals
    - Simple imputation for remaining missing values
    - Drop very high-cardinality non-numeric columns (like names) and warn
    Returns: df (modified), feature_cols (list), target_is_continuous (bool)
    """
    df = df.copy()

    # Try to robustly coerce target to numeric
    y_raw = df[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")

    # Heuristic: treat as continuous numeric target if many unique values
    target_is_continuous = False
    if y_num.notna().sum() >= len(df) * 0.9 and y_num.nunique(dropna=True) > 10:
        df[target_col] = y_num
        target_is_continuous = True
    else:
        # if target is fully numeric after coercion, keep numeric type
        if y_num.notna().all():
            df[target_col] = y_num
            target_is_continuous = y_num.nunique(dropna=True) > 10

    # Drop rows where target is NaN
    before = len(df)
    df = df.dropna(subset=[target_col])
    dropped = before - len(df)
    if dropped > 0:
        st.warning(f"Dropped {dropped} rows because the selected target could not be interpreted (NaN after coercion).")

    # Drop columns with a very high fraction of missing values (e.g., >50%)
    missing_frac = df.isna().mean()
    drop_cols = list(missing_frac[missing_frac > 0.5].index)
    if drop_cols:
        st.warning(f"Dropping columns with >50% missing values: {', '.join(drop_cols)}")
        df = df.drop(columns=drop_cols)

    # Features handling
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    low_card_cats = []
    high_card_cats = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            continue
        # Try coercing to numeric (e.g., numbers encoded as strings)
        conv = pd.to_numeric(df[c], errors="coerce")
        # If majority numeric, coerce and treat as numeric
        if conv.notna().sum() >= len(df) * 0.9:
            df[c] = conv
            numeric_cols.append(c)
            continue
        # Otherwise treat as categorical
        nunique = df[c].nunique(dropna=True)
        if nunique <= 20:
            low_card_cats.append(c)
        else:
            high_card_cats.append(c)

    if high_card_cats:
        st.warning(
            f"Dropping high-cardinality non-numeric columns (likely identifiers/text): {', '.join(high_card_cats)}"
        )
        df = df.drop(columns=high_card_cats)

    # Impute and encode
    # Numeric: fill missing with median
    for c in numeric_cols:
        if c in df.columns:
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)

    # Low-cardinality cats: fill missing with placeholder then one-hot
    for c in low_card_cats:
        if c in df.columns:
            df[c] = df[c].fillna("__MISSING__")

    if low_card_cats:
        # Only include columns still present (some may have been dropped)
        cols_to_encode = [c for c in low_card_cats if c in df.columns]
        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # Final feature list (exclude target)
    feature_cols = [c for c in df.columns if c != target_col]

    # Ensure no remaining missing values in features; fill any stray numerics with medians and others with a default
    if feature_cols:
        if df[feature_cols].isna().any().any():
            for c in feature_cols:
                if df[c].isna().any():
                    if pd.api.types.is_numeric_dtype(df[c]):
                        df[c] = df[c].fillna(df[c].median())
                    else:
                        df[c] = df[c].fillna("__MISSING__")

    return df, feature_cols, target_is_continuous


# ----------------------------
# Load Data + Choose Target
# ----------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df = clean_uploaded_df(df)

    st.sidebar.markdown("### Target selection")
    target_col = st.sidebar.selectbox(
        "Choose target column",
        options=df.columns.tolist()
    )

    # Preprocess uploaded data for modeling
    df, feature_cols, target_is_continuous = preprocess_for_modeling(df, target_col)

else:
    df, feature_cols, target_col = load_demo_data()
    target_is_continuous = False

# If the selected target appears continuous, offer auto-binning because the app is classification-focused
if target_is_continuous:
    st.warning(
        "The selected target appears to be continuous (regression). This app is designed for classification explanations (uses predict_proba etc.)."
    )
    auto_bin = st.sidebar.checkbox("Auto-bin continuous target into N classes for classification", value=False)
    if auto_bin:
        n_bins = st.sidebar.slider("Number of bins", 2, 20, 5)
        try:
            # Use quantile binning to create roughly balanced classes
            df[target_col] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
            # After binning, ensure it's treated as categorical
            if df[target_col].dtype.name.startswith('category'):
                df[target_col] = df[target_col].astype(float)
            # Recompute features since binning may have introduced NaNs
            df, feature_cols, target_is_continuous = preprocess_for_modeling(df, target_col)
            st.success(f"Binned target into {int(df[target_col].nunique())} classes.")
            target_is_continuous = False
        except Exception as e:
            st.error(f"Auto-binning failed: {e}. Choose a different number of bins or a categorical target.")
            st.stop()
    else:
        st.error("Selected target is continuous. Choose a categorical target or enable auto-binning to proceed.")
        st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), width='stretch')

# Ensure target has no missing values (we already dropped some during preprocessing)
# Ensure we have at least some rows and no missing values in features (impute if needed)
if len(df) == 0:
    st.error("No rows available after preprocessing. Check your target selection or data quality.")
    st.stop()

# Post-preprocessing sanity: if any feature values are still missing, impute simple values
if feature_cols and df[feature_cols].isna().any().any():
    st.warning("Some missing feature values remain; applying simple imputation (median for numeric, placeholder for others).")
    for c in feature_cols:
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna("__MISSING__")

# Final sanity: ensure at least 2 rows
if len(df) < 2:
    st.error("Not enough rows after preprocessing to train/test split. Provide more data or choose a different target.")
    st.stop()

# Convert features to a numeric numpy array (SHAP/ML code expects numeric inputs)
try:
    X = df[feature_cols].to_numpy(dtype=np.float64)
except Exception as e:
    # Identify problematic columns to give the user actionable feedback
    bad_cols = []
    for c in feature_cols:
        try:
            _ = pd.to_numeric(df[c], errors='raise')
        except Exception:
            bad_cols.append(c)
    st.error(
        "Could not convert feature columns to numeric types required by SHAP/modeling. "
        f"Problematic columns: {bad_cols}. Try re-running preprocessing or choosing a different target. Error: {e}"
    )
    st.stop()

y_raw = df[target_col].values

# Encode target for classification; keep numeric target raw for regression
if not target_is_continuous:
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]
else:
    y = y_raw
    class_names = [str(c) for c in np.unique(y)]

# Decide whether to stratify: only when every class has at least 2 members.
stratify = None
try:
    y_series = pd.Series(y)
    vc = y_series.value_counts()
    if vc.min() >= 2 and vc.size > 1 and not locals().get('target_is_continuous', False):
        stratify = y
    else:
        # warn if user expected stratification but it's unsafe
        if vc.min() < 2 and vc.size > 1:
            st.warning("Not stratifying split because some classes have fewer than 2 samples.")
except Exception:
    stratify = None

# Prevent classification with an extremely large number of classes
if not target_is_continuous:
    n_classes = len(np.unique(y))
    if n_classes > 50:
        st.error(f"Too many distinct classes for classification ({n_classes}). Consider binning the target or choosing a categorical target.")
        st.stop()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=int(seed), stratify=stratify
)

# Model
model = build_model(model_name, random_state=int(seed))
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
try:
    pipe.fit(X_train, y_train)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

predict_proba_fn = lambda X_: pipe.predict_proba(X_)
predict_fn = lambda X_: pipe.predict(X_)

# Choose instance
st.subheader("Select Instance to Explain")
idx = st.number_input("Test set index", min_value=0, max_value=len(X_test) - 1, value=0)
x_instance = X_test[int(idx)]
y_true = y_test[int(idx)]
pred = predict_fn(x_instance.reshape(1, -1))[0]
proba = predict_proba_fn(x_instance.reshape(1, -1))[0]

c1, c2, c3 = st.columns(3)
c1.metric("True label", int(y_true))
c2.metric("Predicted label", int(pred))
c3.write("Predicted probabilities:")
c3.write(proba)

run_btn = st.button("Run Reliability Audit")

if run_btn:
    st.info("Running audit...")

    # --- LIME repeated runs (stability + latency avg) ---
    lime_feature_sets = []
    lime_latencies = []
    lime_explanations = []

    for r in range(int(lime_runs)):
        lime_list, lat_ms = lime_explain_instance(
            X_train=X_train,
            feature_names=feature_cols,
            class_names=class_names,
            predict_proba_fn=predict_proba_fn,
            x_instance=x_instance,
            num_features=int(num_features),
            seed=int(seed) + r
        )
        lime_latencies.append(lat_ms)
        lime_explanations.append(lime_list)
        lime_feature_sets.append(set(topk_from_lime(lime_list, k=int(num_features))))

    lime_stability = mean_pairwise_jaccard(lime_feature_sets)
    lime_latency_ms = float(np.mean(lime_latencies))

    # --- LIME fidelity (local ridge proxy) ---
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(int(seed))
    n_neigh = 500
    noise = rng.normal(0, 0.5, size=(n_neigh, X_train.shape[1]))
    X_neigh = x_instance.reshape(1, -1) + noise

    probs = predict_proba_fn(X_neigh)
    if probs.shape[1] > 1:
        y_neigh = probs[:, 1]
    else:
        y_neigh = probs[:, 0]

    ridge = Ridge(alpha=1.0, random_state=int(seed))
    ridge.fit(X_neigh, y_neigh)
    y_hat = ridge.predict(X_neigh)
    lime_fidelity = fidelity_r2(y_neigh, y_hat)

    # --- SHAP (TreeSHAP only for tree models in MVP) ---
    shap_latency_ms = None
    shap_fidelity = None
    shap_stability = 1.0
    shap_topk = None

    if model_name in ["RandomForest", "XGBoost"]:
        # Refit tree model without scaler (cleaner for TreeSHAP)
        try:
            tree_model = build_model(model_name, random_state=int(seed))
            tree_model.fit(X_train, y_train)

            bg = X_train[np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)]

            # Use the helper (may raise) — wrap in try to avoid crashing the app
            try:
                shap_values, shap_latency_ms = shap_explain_instance_tree(tree_model, bg, x_instance)
            except Exception as e:
                shap_latency_ms = None
                st.warning(f"SHAP helper failed: {e}")

            # SHAP fidelity proxy via reconstruction (guarded)
            try:
                import shap as _shap
                # Convert background and instance to numeric arrays and impute NaN/Inf using X_train medians
                try:
                    bg_arr = np.asarray(bg, dtype=np.float64)
                    xi = np.asarray(x_instance.reshape(1, -1), dtype=np.float64)
                except Exception as e:
                    raise ValueError(f"Could not convert SHAP background/instance to float arrays: {e}")

                # Impute NaN/Inf in background using column medians from X_train
                if np.isnan(bg_arr).any() or np.isinf(bg_arr).any():
                    col_meds = np.nanmedian(X_train, axis=0)
                    mask_nan = np.isnan(bg_arr)
                    mask_inf = np.isinf(bg_arr)
                    for j in range(bg_arr.shape[1]):
                        if mask_nan[:, j].any() or mask_inf[:, j].any():
                            bg_arr[mask_nan[:, j], j] = col_meds[j]
                            bg_arr[mask_inf[:, j], j] = col_meds[j]

                if np.isnan(xi).any() or np.isinf(xi).any():
                    col_meds = np.nanmedian(X_train, axis=0)
                    for j in range(xi.shape[1]):
                        if np.isnan(xi[0, j]) or np.isinf(xi[0, j]):
                            xi[0, j] = col_meds[j]

                # Time the TreeExplainer call and compute shap values
                t0_shap = time.perf_counter()
                exp = _shap.TreeExplainer(tree_model, data=bg_arr)
                sv = exp.shap_values(xi)
                shap_latency_ms = (time.perf_counter() - t0_shap) * 1000.0

                probs = tree_model.predict_proba(xi)
                probs = probs[0]
                # Choose class index consistent with the model output (pick the most probable class)
                try:
                    class_idx = int(np.argmax(probs))
                except Exception:
                    class_idx = 0

                if isinstance(sv, list):
                    # pick the shap values corresponding to the chosen class index, but be safe about bounds
                    if len(sv) > class_idx:
                        sv_use = sv[class_idx]
                    else:
                        # If list is shorter than class_idx, use the last available shap values
                        sv_use = sv[-1] if len(sv) > 0 else sv[0]
                    
                    base = exp.expected_value
                    if isinstance(base, (list, np.ndarray)):
                        if len(base) > class_idx:
                            base_use = base[class_idx]
                        else:
                            base_use = base[-1] if len(base) > 0 else base[0]
                    else:
                        base_use = base
                else:
                    sv_use = sv
                    base_use = exp.expected_value

                # Ensure we end up with scalar base and scalar sum of shap values
                try:
                    base_scalar = np.asarray(base_use).squeeze()
                    if getattr(base_scalar, 'shape', ()) != ():
                        # If still a list/array, try to extract the scalar value
                        try:
                            base_scalar = base_scalar.item()
                        except Exception:
                            base_scalar = base_scalar[0] if len(base_scalar) > 0 else 0.0
                    base_scalar = float(base_scalar)
                    shap_sum = float(np.sum(sv_use))
                    shap_recon = base_scalar + shap_sum
                    model_out_scalar = float(probs[class_idx] if class_idx < len(probs) else probs[-1])
                    shap_fidelity = max(0.0, 1.0 - abs(model_out_scalar - shap_recon))
                except Exception as e:
                    # Fall back: cannot compute fidelity; leave as None but keep topk if available
                    st.warning(f"Could not compute SHAP fidelity due to shape/scalar conversion error: {e}")
                    shap_fidelity = None

                abs_sv = np.abs(np.array(sv_use)).ravel()
                order = np.argsort(-abs_sv)
                shap_topk = [feature_cols[i] for i in order[:int(num_features)]]
            except Exception as e:
                # Provide helpful warning and fall back to model feature importances for a proxy top-k
                st.warning(f"SHAP TreeExplainer failed: {e}")
                shap_fidelity = None
                try:
                    if hasattr(tree_model, 'feature_importances_'):
                        fi = np.array(tree_model.feature_importances_)
                        order = np.argsort(-fi)
                        shap_topk = [feature_cols[i] for i in order[:int(num_features)]]
                    else:
                        shap_topk = None
                except Exception:
                    shap_topk = None
        except Exception as e:
            # If the tree model itself fails (e.g., invalid labels for XGBoost), skip SHAP and warn
            st.error(f"Tree model training failed: {e}")
            shap_latency_ms = None
            shap_fidelity = None
            shap_topk = None

    # --- Confidence scores ---
    lime_score = explanation_confidence_score(
        lime_stability, lime_fidelity, lime_latency_ms, latency_target_ms=float(latency_target)
    )

    shap_score = None
    if shap_latency_ms is not None and shap_fidelity is not None:
        shap_score = explanation_confidence_score(
            shap_stability, shap_fidelity, float(shap_latency_ms), latency_target_ms=float(latency_target)
        )

    # --- Disagreement flag (Jaccard overlap of top-k) ---
    disagreement_flag = False
    overlap_jaccard = None
    if shap_topk is not None:
        lime_topk_run0 = set(topk_from_lime(lime_explanations[0], int(num_features)))
        shap_topk_set = set(shap_topk)
        overlap_jaccard = len(lime_topk_run0 & shap_topk_set) / max(1, len(lime_topk_run0 | shap_topk_set))
        disagreement_flag = (overlap_jaccard < 0.5)

    # --- Display results ---
    st.subheader("Audit Results")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### LIME")
        st.metric("Stability (mean pairwise Jaccard)", f"{lime_stability:.3f}")
        st.metric("Fidelity (local R² proxy)", f"{lime_fidelity:.3f}")
        st.metric("Latency (ms, avg)", f"{lime_latency_ms:.1f}")
        st.metric("Explanation Confidence Score", f"{lime_score:.3f}")

        st.write("Top-K features (Run 1):")
        st.dataframe(pd.DataFrame(lime_explanations[0], columns=["Feature", "Weight"]).head(int(num_features)),
                     width='stretch')

    with colB:
        st.markdown("### SHAP")
        if model_name not in ["RandomForest", "XGBoost"]:
            # For non-tree models (e.g., MLP) we can optionally run KernelSHAP (slow) when enabled in the sidebar
            if model_name == "MLP" and use_kernel_shap:
                st.info("Running KernelSHAP for MLP (this may be slow)...")
                try:
                    # small background sample for KernelSHAP
                    bg_k = X_train[np.random.choice(len(X_train), size=min(50, len(X_train)), replace=False)]
                    try:
                        shap_values, shap_latency_ms = shap_explain_instance_kernel(predict_proba_fn, bg_k, x_instance, nsamples=int(kernel_shap_nsamples or 100))

                        # Try to extract a per-feature importance vector robustly
                        _sv = shap_values
                        if isinstance(_sv, list):
                            # multiclass -> pick class 1 if present, else first
                            _sv = _sv[1] if len(_sv) > 1 else _sv[0]
                        _sv = np.array(_sv)
                        if _sv.ndim == 1:
                            shap_abs = np.abs(_sv)
                        else:
                            shap_abs = np.mean(np.abs(_sv), axis=0)

                        # Derive top-k features
                        shap_topk = [feature_cols[i] for i in np.argsort(-shap_abs)[:int(num_features)]]

                    except Exception as e:
                        shap_latency_ms = None
                        st.warning(f"KernelSHAP helper failed: {e}")
                except Exception as e:
                    st.warning(f"KernelSHAP preparation failed: {e}")
                    st.warning("TreeSHAP not available for MLP in this MVP. You can enable KernelSHAP in the sidebar (it's slow).")
            else:
                st.warning("TreeSHAP not available for MLP in this MVP. You can enable KernelSHAP in the sidebar (it's slow).")
        else:
            st.metric("Stability", "1.000 (deterministic)")
            st.metric("Fidelity (reconstruction proxy)", f"{shap_fidelity:.3f}" if shap_fidelity is not None else "N/A")
            st.metric("Latency (ms)", f"{float(shap_latency_ms):.1f}" if shap_latency_ms is not None else "N/A")
            st.metric("Explanation Confidence Score", f"{shap_score:.3f}" if shap_score is not None else "N/A")

            if shap_topk is not None:
                st.write("Top-K features (SHAP |abs|):")
                st.dataframe(pd.DataFrame({"Feature": shap_topk}), width='stretch')

    if overlap_jaccard is not None:
        st.write(f"**LIME vs SHAP Top-K overlap (Jaccard):** {overlap_jaccard:.3f}")

    if disagreement_flag:
        st.error("⚠️ Explanation disagreement detected (low overlap between LIME and SHAP top-K). Interpret with caution.")
    else:
        st.success("No major LIME vs SHAP disagreement detected under the current settings.")

    # --- Diagnostics plot: LIME latency distribution ---
    st.subheader("Diagnostics")
    fig = plt.figure()
    plt.hist(lime_latencies, bins=10)
    plt.xlabel("LIME Latency (ms)")
    plt.ylabel("Count")
    st.pyplot(fig)

    # --- Log run ---
    record = {
        "dataset": "uploaded_csv" if uploaded is not None else "synthetic_demo",
        "target_col": target_col,
        "model": model_name,
        "instance_idx": int(idx),
        "lime_runs": int(lime_runs),
        "top_k": int(num_features),
        "lime_stability_jaccard": float(lime_stability),
        "lime_fidelity_r2": float(lime_fidelity),
        "lime_latency_ms_avg": float(lime_latency_ms),
        "lime_confidence_score": float(lime_score),
        "shap_latency_ms": float(shap_latency_ms) if shap_latency_ms is not None else None,
        "shap_fidelity_proxy": float(shap_fidelity) if shap_fidelity is not None else None,
        "shap_confidence_score": float(shap_score) if shap_score is not None else None,
        "lime_shap_overlap_jaccard": float(overlap_jaccard) if overlap_jaccard is not None else None,
        "disagreement_flag": bool(disagreement_flag),
        "seed": int(seed),
    }
    append_jsonl(record)

    st.caption("Run logged to outputs/runs.jsonl")
