import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from audit.models import build_model
from audit.explainers import lime_explain_instance, shap_explain_instance_kernel, topk_from_lime
from audit.metrics import jaccard, fidelity_r2
from audit.scoring import explanation_confidence_score
from audit.logging_utils import append_jsonl

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
    df["target"] = y
    return df, cols, "target"

def clean_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    # Drop common junk index columns
    junk_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)
        
    # Also drop a pure index column if it looks like 0..n-1
    for c in df.columns[:2]:
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
    - Drop very high-cardinality non-numeric columns
    """
    df = df.copy()

    # Coerce target
    y_raw = df[target_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")
    
    # Heuristic for continuous vs classification target
    target_is_continuous = False
    if y_num.notna().sum() >= len(df) * 0.9 and y_num.nunique(dropna=True) > 10:
        df[target_col] = y_num
        target_is_continuous = True
    else:
        if y_num.notna().all():
            df[target_col] = y_num
            target_is_continuous = y_num.nunique(dropna=True) > 10

    # Drop missing taxrget
    df = df.dropna(subset=[target_col])
    
    # Drop empty cols
    missing_frac = df.isna().mean()
    drop_cols = [
        c for c in missing_frac[missing_frac > 0.5].index
        if c != target_col
    ]
    if drop_cols:
         st.warning(f"Dropping columns with >50% missing values: {', '.join(drop_cols)}")
         df = df.drop(columns=drop_cols)

    # Features
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = []
    low_card_cats = []
    high_card_cats = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            continue
        # Try coercing
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() >= len(df) * 0.9:
            df[c] = conv
            numeric_cols.append(c)
            continue
        
        # Categorical
        if df[c].nunique(dropna=True) <= 20: 
            low_card_cats.append(c)
        else:
            high_card_cats.append(c)
            
    if high_card_cats:
        st.warning(f"Dropping high-cardinality non-numeric columns: {', '.join(high_card_cats)}")
        df = df.drop(columns=high_card_cats)
        
    # Impute numeric
    for c in numeric_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
            
    # Impute/Encode cats
    if low_card_cats:
        cols_to_encode = [c for c in low_card_cats if c in df.columns]
        if cols_to_encode:
            # fillna before dummy
            for c in cols_to_encode:
                df[c] = df[c].fillna("__MISSING__")
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    feature_cols = [c for c in df.columns if c != target_col]
    
    # Final cleanup
    if feature_cols and df[feature_cols].isna().any().any():
        for c in feature_cols:
            if df[c].isna().any():
                 if pd.api.types.is_numeric_dtype(df[c]):
                      df[c] = df[c].fillna(0)
                 else:
                      df[c] = df[c].fillna("0")
                      
    return df, feature_cols, target_is_continuous

def training_cache_key(model_name: str, seed: int, X_train: np.ndarray, y_train: np.ndarray):
    sample_n = int(min(256, len(X_train)))
    x_sample = X_train[:sample_n]
    y_sample = y_train[:sample_n]
    return (
        model_name,
        int(seed),
        tuple(X_train.shape),
        tuple(y_train.shape),
        float(np.sum(x_sample)),
        float(np.mean(x_sample)),
        float(np.std(x_sample)),
        float(np.sum(y_sample)),
    )

st.title("XAI Reliability Audit Prototype")

# --- Sidebar controls ---
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox("Model", ["RandomForest", "HistGradientBoosting", "MLP"])
# Option for non-tree explainability
use_kernel_shap = False
kernel_shap_nsamples = 100
if model_name == "MLP":
    use_kernel_shap = st.sidebar.checkbox("Enable KernelSHAP for MLP (slow)", value=False)
    if use_kernel_shap:
        kernel_shap_nsamples = st.sidebar.slider("KernelSHAP nsamples", 25, 500, 100, step=25)
TOP_K = 5
Fidelity_N = 50
KERNEL_SHAP_BACKGROUND_SIZE = 50
BOUNDARY_THRESHOLD = 0.05
ADDITIVITY_WARNING_THRESHOLD = 1e-2
AUDIT_SCHEMA = "vB"
st.sidebar.caption(f"Top-K features fixed to {TOP_K} for evaluation consistency.")
lime_runs = st.sidebar.slider("LIME repeated runs", 10, 30, 10)
shap_runs = st.sidebar.slider("SHAP repeated runs", 10, 30, 10)
latency_target = st.sidebar.number_input("Real-time latency target (ms)", 50, 2000, 200)

seed = st.sidebar.number_input("Base random seed", 0, 9999, 42)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
clear_log_before_run = st.sidebar.checkbox("Clear audit log before run", value=False)

# Logic to handle data loading and target selection
if uploaded is not None:
    try:
        # Read plain dataframe first to get columns
        df_raw = pd.read_csv(uploaded)
        df_raw = clean_uploaded_df(df_raw)
        
        all_cols = list(df_raw.columns)
        if not all_cols:
            raise ValueError("The CSV has no usable columns.")
        
        # Heuristic for default target
        # Most tabular CSVs place the response column last. Use that as the
        # fallback instead of silently selecting the first feature or ID column.
        default_ix = max(0, len(all_cols) - 1)
        for i, c in enumerate(all_cols):
            if c.lower() in ["target", "label", "class", "y", "two_year_recid", "recid", "outcome"]:
                default_ix = i
                break
                
        target_col = st.sidebar.selectbox("Select Target Column", all_cols, index=default_ix)
        
        df, feature_cols, target_is_continuous = preprocess_for_modeling(df_raw, target_col)
        
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        st.stop()
else:
    df, feature_cols, target_col = load_demo_data()
    target_is_continuous = False

st.subheader("Dataset Preview")
st.dataframe(df.head(10), width="stretch")

X = df[feature_cols].values
if not feature_cols:
    st.error("No usable feature columns remain after preprocessing.")
    st.stop()
if len(df) < 4:
    st.error("At least 4 rows with non-missing target values are required.")
    st.stop()
try:
    X = X.astype(np.float64)
except ValueError as e:
    st.error(f"Failed to convert features to numeric. Check for remaining non-numeric columns. Error: {e}")
    st.stop()
y = df[target_col].values

# All models and explainers in this prototype are classification-only. Letting a
# continuous response reach train_test_split/model.fit produces scikit-learn's
# confusing "unique classes" warning (and eventually a training error).
if locals().get("target_is_continuous", False):
    st.error(
        f"'{target_col}' looks continuous or has too many unique numeric values "
        "for a classification target. Select a categorical/class-label target "
        "column, or convert this target into discrete classes before uploading."
    )
    st.stop()

# Encode target for classification
le = LabelEncoder()
y = le.fit_transform(y)
class_names = [str(c) for c in le.classes_]
if len(class_names) < 2:
    st.error("The target column must contain at least two distinct classes.")
    st.stop()

# split
stratify_y = y if (len(np.unique(y)) > 1 and np.min(np.unique(y, return_counts=True)[1]) > 1) else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=int(seed), stratify=stratify_y
)

# Train once per data/model/seed combo and reuse across Streamlit reruns.
train_key = training_cache_key(model_name, int(seed), X_train, y_train)
cached_key = st.session_state.get("trained_model_key")
if cached_key != train_key:
    model = build_model(model_name, random_state=int(seed))
    if model_name == "MLP":
        fitted_model = Pipeline([("scaler", StandardScaler()), ("model", model)])
    else:
        fitted_model = model
    fitted_model.fit(X_train, y_train)
    st.session_state["trained_model_key"] = train_key
    st.session_state["trained_model_obj"] = fitted_model
else:
    fitted_model = st.session_state["trained_model_obj"]

predict_proba_fn = lambda X_: fitted_model.predict_proba(X_)
predict_fn = lambda X_: fitted_model.predict(X_)
tree_model_main = fitted_model if model_name in ["RandomForest", "HistGradientBoosting"] else None

# choose instance
st.subheader("Select Instance to Explain")
idx = st.number_input("Test set index", min_value=0, max_value=len(X_test)-1, value=0)
x_instance = X_test[int(idx)]
y_true = y_test[int(idx)]
pred = predict_fn(x_instance.reshape(1,-1))[0]
proba = predict_proba_fn(x_instance.reshape(1,-1))[0]
pred_class_idx = int(np.argmax(proba))
prediction_confidence = float(proba[pred_class_idx])
near_boundary = bool(abs(prediction_confidence - 0.5) < BOUNDARY_THRESHOLD)

c1, c2, c3 = st.columns(3)
c1.metric("True label", int(y_true))
c2.metric("Predicted label", int(pred))
c3.write("Predicted probabilities:")
c3.write(proba)

run_btn = st.button("Run Reliability Audit")

if run_btn:
    st.info("Running audit...")
    runs_path = "outputs/runs.jsonl"
    if clear_log_before_run and os.path.exists(runs_path):
        os.remove(runs_path)
    def pairwise_jaccard_scores(feature_sets):
        vals = []
        n = len(feature_sets)
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(jaccard(feature_sets[i], feature_sets[j]))
        return vals

    def summarize_latencies(lat_ms):
        if not lat_ms:
            return None, None, None
        arr = np.array(lat_ms, dtype=float)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        p95_v = float(np.percentile(arr, 95))
        return mean_v, std_v, p95_v

    def lime_recon_for_instance(x_row, seed_val):
        lime_list_i, lime_map_i, _, lime_details_i = lime_explain_instance(
            X_train=X_train,
            feature_names=feature_cols,
            class_names=[str(c) for c in np.unique(y_train)],
            predict_proba_fn=predict_proba_fn,
            x_instance=x_row,
            num_features=int(TOP_K),
            seed=seed_val,
        )
        probs_i = predict_proba_fn(x_row.reshape(1, -1))[0]
        pred_label_i = int(np.argmax(probs_i))
        available_labels = list(lime_map_i.keys())
        target_label_i = pred_label_i if pred_label_i in lime_map_i else (available_labels[0] if available_labels else None)
        if target_label_i is None:
            return 0.0, lime_list_i, lime_map_i
        local_weights = lime_map_i.get(target_label_i, [])
        contrib_sum = float(np.sum([w for _, w in local_weights])) if local_weights else 0.0
        intercept_obj = lime_details_i.get("intercept", 0.0) if isinstance(lime_details_i, dict) else 0.0
        if isinstance(intercept_obj, dict):
            intercept = float(intercept_obj.get(target_label_i, 0.0))
        elif isinstance(intercept_obj, (list, np.ndarray)):
            idx_safe = target_label_i if target_label_i < len(intercept_obj) else 0
            intercept = float(intercept_obj[idx_safe])
        else:
            intercept = float(intercept_obj)
        recon = float(intercept + contrib_sum)
        return recon, lime_list_i, lime_map_i

    def normalize_shap_values(shap_values_obj, pred_class):
        if isinstance(shap_values_obj, list):
            idx = int(min(max(pred_class, 0), len(shap_values_obj) - 1))
            arr = np.array(shap_values_obj[idx], dtype=float)
        else:
            arr = np.array(shap_values_obj, dtype=float)
            if arr.ndim == 3:
                class_idx = int(min(max(pred_class, 0), arr.shape[2] - 1))
                arr = arr[0, :, class_idx]
            elif arr.ndim == 2:
                arr = arr[0]
        return arr.ravel()

    def normalize_shap_values_row(shap_values_obj, row_idx, pred_class):
        if isinstance(shap_values_obj, list):
            class_idx = int(min(max(pred_class, 0), len(shap_values_obj) - 1))
            arr = np.array(shap_values_obj[class_idx], dtype=float)
            if arr.ndim == 2:
                return arr[row_idx].ravel()
            return arr.ravel()
        arr = np.array(shap_values_obj, dtype=float)
        if arr.ndim == 3:
            class_idx = int(min(max(pred_class, 0), arr.shape[2] - 1))
            return arr[row_idx, :, class_idx].ravel()
        if arr.ndim == 2:
            return arr[row_idx].ravel()
        return arr.ravel()

    def pick_base_value(expected_value_obj, pred_class):
        if isinstance(expected_value_obj, (list, np.ndarray)):
            arr = np.array(expected_value_obj, dtype=float).ravel()
            if arr.size == 0:
                return 0.0
            idx = int(min(max(pred_class, 0), arr.size - 1))
            return float(arr[idx])
        return float(expected_value_obj)

    def model_output_for_class(model_obj, x_row, pred_class, output_space):
        if output_space == "probability":
            probs_local = model_obj.predict_proba(x_row.reshape(1, -1))[0]
            idx = int(min(max(pred_class, 0), len(probs_local) - 1))
            return float(probs_local[idx])
        raw_local = np.array(model_obj.decision_function(x_row.reshape(1, -1)), dtype=float)
        if raw_local.ndim == 0:
            return float(raw_local)
        if raw_local.ndim == 1:
            if raw_local.size == 1:
                return float(raw_local[0])
            idx = int(min(max(pred_class, 0), raw_local.size - 1))
            return float(raw_local[idx])
        if raw_local.shape[1] == 1:
            return float(raw_local[0, 0])
        idx = int(min(max(pred_class, 0), raw_local.shape[1] - 1))
        return float(raw_local[0, idx])

    # --- LIME repeated runs (stability + latency avg) ---
    lime_feature_sets = []
    lime_latencies = []
    lime_explanations = []
    lime_topk_indices_0 = []
    for r in range(int(lime_runs)):
        lime_list, lime_map, lat_ms, _ = lime_explain_instance(
            X_train=X_train,
            feature_names=feature_cols,
            class_names=[str(c) for c in np.unique(y_train)],
            predict_proba_fn=predict_proba_fn,
            x_instance=x_instance,
            num_features=int(TOP_K),
            seed=int(seed) + r,
        )
        lime_latencies.append(lat_ms)
        lime_explanations.append(lime_list)
        if r == 0:
            labels = list(lime_map.keys())
            if labels:
                first_label = labels[0]
                sorted_tuples = sorted(lime_map[first_label], key=lambda x: abs(x[1]), reverse=True)
                lime_topk_indices_0 = [idx for idx, _ in sorted_tuples[:int(TOP_K)]]
        lime_feature_sets.append(set(topk_from_lime(lime_list, k=int(TOP_K))))

    lime_jaccard_scores = pairwise_jaccard_scores(lime_feature_sets)
    lime_mean_jaccard = float(np.mean(lime_jaccard_scores)) if lime_jaccard_scores else 1.0
    lime_std_jaccard = float(np.std(lime_jaccard_scores)) if lime_jaccard_scores else 0.0
    lime_min_jaccard = float(np.min(lime_jaccard_scores)) if lime_jaccard_scores else 1.0
    lime_var_jaccard = float(np.var(lime_jaccard_scores)) if lime_jaccard_scores else 0.0
    lime_is_unstable = lime_var_jaccard > 0.01
    lime_supports_h1_jaccard = lime_mean_jaccard < 0.9
    lime_stability = lime_mean_jaccard
    lime_latency_ms_mean, lime_latency_ms_std, lime_latency_ms_p95 = summarize_latencies(lime_latencies)
    lime_latency_ms = float(lime_latency_ms_mean) if lime_latency_ms_mean is not None else None

    # --- SHAP repeated runs (stability + latency avg) ---
    shap_latencies = []
    shap_feature_sets = []
    shap_stability = None
    shap_latency_ms = None
    shap_topk = None
    shap_csv_data = None
    shap_jaccard_scores = []
    shap_mean_jaccard = None
    shap_std_jaccard = None
    shap_min_jaccard = None
    shap_var_jaccard = None
    shap_is_deterministic = None
    explainer_type = None
    shap_background_size_logged = None
    shap_values = None
    sv_use = None
    model_output_space = "probability"
    explained_class_idx = None
    shap_fx = None
    shap_reconstruction = None
    shap_additivity_error = None
    shap_additivity_warning = None
    kernel_nsamples_logged = None
    background_size_logged = None
    background_fixed_logged = None
    cold_start_logged = True
    shap_latency_ms_mean, shap_latency_ms_std, shap_latency_ms_p95 = None, None, None
    shap_succeeded = False

    if model_name in ["RandomForest", "HistGradientBoosting", "MLP"] and (model_name != "MLP" or use_kernel_shap):
        try:
            tree_model = None
            if model_name in ["RandomForest", "HistGradientBoosting"]:
                tree_model = tree_model_main
                model_output_space = "raw" if model_name == "HistGradientBoosting" else "probability"
                explainer_type = "TreeSHAP"
                rng_bg = np.random.default_rng(int(seed))
                bg_size_tree = int(min(200, len(X_train)))
                bg_idx_tree = rng_bg.choice(len(X_train), size=bg_size_tree, replace=False)
                bg_fixed = X_train[bg_idx_tree]
                shap_background_size_logged = int(bg_size_tree)
                background_size_logged = int(bg_size_tree)
                background_fixed_logged = True
            else:
                explainer_type = "KernelSHAP"
                shap_background_size_logged = int(min(KERNEL_SHAP_BACKGROUND_SIZE, len(X_train)))
                kernel_nsamples_logged = int(kernel_shap_nsamples)
                background_size_logged = int(shap_background_size_logged)
                background_fixed_logged = False

            rng = np.random.default_rng(int(seed))
            if model_name in ["RandomForest", "HistGradientBoosting"]:
                import shap
                t0_shap = time.perf_counter()
                tree_explainer = shap.TreeExplainer(tree_model, data=bg_fixed, model_output=model_output_space)
                shap_values_r = tree_explainer.shap_values(x_instance.reshape(1, -1))
                shap_latency_r = (time.perf_counter() - t0_shap) * 1000.0
                expected_value_r = tree_explainer.expected_value
                probs = tree_model.predict_proba(x_instance.reshape(1, -1))[0]
                pred_class = int(np.argmax(probs))
                sv_r = normalize_shap_values(shap_values_r, pred_class)
                abs_sv = np.abs(sv_r)
                order = np.argsort(-abs_sv)
                topk_features = [feature_cols[i] for i in order[:int(TOP_K)]]
                shap_feature_sets.append(set(topk_features))
                shap_latencies.append(shap_latency_r)
                sv_use = sv_r
                shap_values = shap_values_r
                explained_class_idx = int(pred_class)
                base_val_r = pick_base_value(expected_value_r, pred_class)
                shap_fx = model_output_for_class(tree_model, x_instance, pred_class, model_output_space)
                shap_reconstruction = float(base_val_r + np.sum(sv_r))
                shap_additivity_error = float(abs(shap_fx - shap_reconstruction))
                shap_additivity_warning = bool(shap_additivity_error > ADDITIVITY_WARNING_THRESHOLD)
            else:
                for r in range(int(shap_runs)):
                    shap_values_r = None
                    expected_value_r = None
                    shap_latency_r = 0.0
                    bg_idx = rng.choice(len(X_train), size=shap_background_size_logged, replace=False)
                    bg = X_train[bg_idx]
                    shap_values_r, expected_value_r, shap_latency_r, kernel_meta = shap_explain_instance_kernel(
                        predict_proba_fn, bg, x_instance, nsamples=int(kernel_shap_nsamples)
                    )
                    probs = predict_proba_fn(x_instance.reshape(1, -1))[0]
                    kernel_nsamples_logged = int(kernel_meta.get("kernel_nsamples", kernel_shap_nsamples))
                    background_size_logged = int(kernel_meta.get("background_size", shap_background_size_logged))

                    pred_class = int(np.argmax(probs))
                    sv_r = normalize_shap_values(shap_values_r, pred_class)
                    abs_sv = np.abs(sv_r)
                    order = np.argsort(-abs_sv)
                    topk_features = [feature_cols[i] for i in order[:int(TOP_K)]]
                    shap_feature_sets.append(set(topk_features))
                    shap_latencies.append(shap_latency_r)
                    sv_use = sv_r
                    shap_values = shap_values_r
                    explained_class_idx = int(pred_class)
                    base_val_r = pick_base_value(expected_value_r, pred_class)
                    shap_fx = float(probs[pred_class])
                    shap_reconstruction = float(base_val_r + np.sum(sv_r))
                shap_additivity_error = float(abs(shap_fx - shap_reconstruction))
                shap_additivity_warning = bool(shap_additivity_error > ADDITIVITY_WARNING_THRESHOLD)

            shap_succeeded = bool(shap_feature_sets and sv_use is not None)
            shap_jaccard_scores = pairwise_jaccard_scores(shap_feature_sets)
            shap_mean_jaccard = float(np.mean(shap_jaccard_scores)) if shap_jaccard_scores else 1.0
            shap_std_jaccard = float(np.std(shap_jaccard_scores)) if shap_jaccard_scores else 0.0
            shap_min_jaccard = float(np.min(shap_jaccard_scores)) if shap_jaccard_scores else 1.0
            shap_var_jaccard = float(np.var(shap_jaccard_scores)) if shap_jaccard_scores else 0.0
            shap_is_deterministic = bool(shap_min_jaccard == 1.0)

            shap_stability = shap_mean_jaccard
            shap_latency_ms_mean, shap_latency_ms_std, shap_latency_ms_p95 = summarize_latencies(shap_latencies)
            shap_latency_ms = float(shap_latency_ms_mean) if shap_latency_ms_mean is not None else None
            if sv_use is not None:
                abs_sv = np.abs(sv_use).ravel()
                order = np.argsort(-abs_sv)
                shap_topk = [feature_cols[i] for i in order[:int(TOP_K)]]
                shap_df = pd.DataFrame({"Feature": feature_cols, "SHAP Value": sv_use.ravel()})
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values("Abs", ascending=False).drop(columns=["Abs"])
                shap_csv_data = shap_df.to_csv(index=False)
        except Exception as e:
            st.error(f"SHAP Explainer Failed: {e}")
    else:
        shap_latency_ms_mean, shap_latency_ms_std, shap_latency_ms_p95 = None, None, None

    # --- Fidelity (R2 reconstruction over N instances) ---
    eval_n = int(min(Fidelity_N, len(X_test)))
    fidelity_indices = np.arange(eval_n)

    lime_true_probs = []
    lime_recon_probs = []
    for i in fidelity_indices:
        x_i = X_test[i]
        probs_i = predict_proba_fn(x_i.reshape(1, -1))[0]
        pred_label_i = int(np.argmax(probs_i))
        true_prob_i = float(probs_i[pred_label_i])
        recon_i, _, _ = lime_recon_for_instance(x_i, seed_val=int(seed) + i)
        lime_true_probs.append(true_prob_i)
        lime_recon_probs.append(float(recon_i))
    lime_fidelity = fidelity_r2(np.array(lime_true_probs), np.array(lime_recon_probs))

    shap_fidelity = None
    if shap_succeeded:
        shap_true_probs = []
        shap_recon_probs = []
        if model_name in ["RandomForest", "HistGradientBoosting"]:
            tree_model_f = tree_model_main
            import shap
            model_output_space_f = "raw" if model_name == "HistGradientBoosting" else "probability"
            # Supplying a background dataset selects interventional TreeSHAP.
            # Probability output is not supported by the default
            # tree_path_dependent perturbation mode.
            rng_tree_f = np.random.default_rng(int(seed))
            bg_size_tree_f = int(min(200, len(X_train)))
            bg_idx_tree_f = rng_tree_f.choice(
                len(X_train), size=bg_size_tree_f, replace=False
            )
            bg_tree_f = X_train[bg_idx_tree_f]
            tree_explainer_f = shap.TreeExplainer(
                tree_model_f,
                data=bg_tree_f,
                feature_perturbation="interventional",
                model_output=model_output_space_f,
            )
            expected_value = tree_explainer_f.expected_value
            X_eval = X_test[fidelity_indices]
            probs_eval = tree_model_f.predict_proba(X_eval)
            shap_vals_eval = tree_explainer_f.shap_values(X_eval)
            for row_i, _ in enumerate(fidelity_indices):
                probs_i = probs_eval[row_i]
                pred_class_i = int(np.argmax(probs_i))
                sv_i = normalize_shap_values_row(shap_vals_eval, row_i, pred_class_i)
                base_val = pick_base_value(expected_value, pred_class_i)
                recon_i = float(base_val + np.sum(sv_i))
                x_i = X_eval[row_i]
                shap_true_probs.append(model_output_for_class(tree_model_f, x_i, pred_class_i, model_output_space_f))
                shap_recon_probs.append(recon_i)
        else:
            rng_f = np.random.default_rng(int(seed))
            bg_idx_f = rng_f.choice(len(X_train), size=int(min(KERNEL_SHAP_BACKGROUND_SIZE, len(X_train))), replace=False)
            bg_f = X_train[bg_idx_f]
            import shap
            kernel_explainer_f = shap.KernelExplainer(predict_proba_fn, bg_f)
            for i in fidelity_indices:
                x_i = X_test[i]
                probs_i = predict_proba_fn(x_i.reshape(1, -1))[0]
                pred_class_i = int(np.argmax(probs_i))
                shap_vals_i = kernel_explainer_f.shap_values(x_i.reshape(1, -1), nsamples=int(kernel_shap_nsamples))
                sv_i = normalize_shap_values(shap_vals_i, pred_class_i)
                exp_val = kernel_explainer_f.expected_value
                base_val = pick_base_value(exp_val, pred_class_i)
                recon_i = float(base_val + np.sum(sv_i))
                shap_true_probs.append(float(probs_i[pred_class_i]))
                shap_recon_probs.append(recon_i)
        shap_fidelity = fidelity_r2(np.array(shap_true_probs), np.array(shap_recon_probs))

    fidelity_difference = float(shap_fidelity - lime_fidelity) if shap_fidelity is not None else None

    latency_ratio = None
    meets_10x_threshold = None
    if shap_latency_ms_mean is not None and lime_latency_ms_mean and lime_latency_ms_mean > 0:
        latency_ratio = float(shap_latency_ms_mean / lime_latency_ms_mean)
        meets_10x_threshold = bool(latency_ratio > 10.0)

    lime_score = explanation_confidence_score(
        lime_stability, lime_fidelity, float(lime_latency_ms_mean), latency_target_ms=float(latency_target)
    )
    shap_score = None
    if shap_latency_ms is not None and shap_fidelity is not None and shap_stability is not None:
        shap_score = explanation_confidence_score(
            shap_stability, shap_fidelity, float(shap_latency_ms_mean), latency_target_ms=float(latency_target)
        )

    disagreement_flag = False
    if shap_topk and lime_explanations:
        lime_topk_set = set([f for f, _ in lime_explanations[0][:int(TOP_K)]])
        shap_topk_set = set(shap_topk)
        inter = len(lime_topk_set & shap_topk_set)
        union = len(lime_topk_set | shap_topk_set)
        l_vs_s_jacc = inter / max(1, union)
        disagreement_flag = (l_vs_s_jacc < 0.5)

    # --- Prepare SHAP CSV Data (Fallback if not created in loop) ---
    if shap_csv_data is None and shap_topk is not None:
         shap_csv_data = pd.DataFrame({"Feature": shap_topk}).to_csv(index=False)


    # --- Display results ---
    st.subheader("Audit Results")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### LIME")
        st.metric("Stability Mean Jaccard", f"{lime_stability:.3f}")
        st.metric("Jaccard Std", f"{lime_std_jaccard:.4f}")
        st.metric("Jaccard Min", f"{lime_min_jaccard:.3f}")
        st.metric("Jaccard Variance", f"{lime_var_jaccard:.4f}")
        st.metric("Unstable (var > 0.01)", str(lime_is_unstable))
        st.metric("Mean Jaccard (< 0.9)", str(lime_supports_h1_jaccard))
        st.metric("Fidelity (R2)", f"{lime_fidelity:.3f}" if lime_fidelity is not None else "N/A")
        st.metric("Latency (ms, mean)", f"{lime_latency_ms_mean:.1f}" if lime_latency_ms_mean is not None else "N/A")

        st.metric("Explanation Confidence Score", f"{lime_score:.3f}")

        st.write("Top-K features (Run 1):")
        st.write(pd.DataFrame(lime_explanations[0], columns=["Feature", "Weight"]).head(int(TOP_K)))

    with colB:
        st.markdown("### SHAP")
        if model_name == "MLP" and not use_kernel_shap:
            st.warning("TreeSHAP not available for MLP. Enable KernelSHAP in the sidebar (slow).")
        else:
            stab_label = "Stability"
            if model_name == "MLP" and use_kernel_shap:
                stab_label = "Stability (KernelSHAP Jaccard)"
            
            st.metric(stab_label, f"{shap_stability:.3f}" if shap_stability is not None else "N/A")
            st.metric("Jaccard Std", f"{shap_std_jaccard:.4f}" if shap_std_jaccard is not None else "N/A")
            st.metric("Jaccard Min", f"{shap_min_jaccard:.3f}" if shap_min_jaccard is not None else "N/A")
            st.metric("Jaccard Variance", f"{shap_var_jaccard:.4f}" if shap_var_jaccard is not None else "N/A")
            st.metric("Deterministic", str(shap_is_deterministic) if shap_is_deterministic is not None else "N/A")
            st.metric("Fidelity (R2)", f"{shap_fidelity:.3f}" if shap_fidelity is not None else "N/A")
            st.metric("Latency (ms, mean)", f"{float(shap_latency_ms_mean):.1f}" if shap_latency_ms_mean is not None else "N/A")
            st.metric("Latency Ratio (SHAP/LIME)", f"{latency_ratio:.2f}" if latency_ratio is not None else "N/A")
            st.metric("Meets >10x threshold", str(meets_10x_threshold) if meets_10x_threshold is not None else "N/A")
            st.metric("Explanation Confidence Score", f"{shap_score:.3f}" if shap_score is not None else "N/A")
            if shap_topk:
                st.write("Top-K features (SHAP |abs|):")
                st.write(pd.DataFrame({"Feature": shap_topk}))
            if shap_additivity_warning:
                st.warning(
                    f"SHAP additivity warning: error={shap_additivity_error:.6f} > {ADDITIVITY_WARNING_THRESHOLD}"
                )

    st.caption(f"Fidelity evaluated as R2 reconstruction over N={eval_n} test instances. Top-K fixed at {TOP_K}.")
    if fidelity_difference is not None:
        st.caption(f"Fidelity difference (SHAP - LIME): {fidelity_difference:.4f}")

    if disagreement_flag:
        st.error("⚠️ Explanation disagreement detected (low overlap between LIME and SHAP top-K). Interpret with caution.")
    else:
        st.success("No major LIME vs SHAP disagreement detected under the current settings.")

    # --- Simple plot: LIME latency distribution ---
    st.subheader("Diagnostics")
    fig = plt.figure()
    plt.hist(lime_latencies, bins=10)
    plt.title("LIME Latency Distribution (Repeated Runs)")
    plt.xlabel("LIME Latency (ms)")
    plt.ylabel("Count")
    st.pyplot(fig)
    st.caption("SHAP is deterministic for tree models; distribution not shown.")
    
    # Save high-res plot to bytes for download
    plot_buf = io.BytesIO()
    fig.savefig(plot_buf, format="png", dpi=300, bbox_inches="tight")
    plot_buf.seek(0)
    plot_bytes = plot_buf.getvalue()

    # --- Log run ---
    record = {
        "dataset": "uploaded_csv" if uploaded is not None else "synthetic_demo",
        "audit_schema": AUDIT_SCHEMA,
        "model_name": model_name,
        "model": model_name,
        "instance_idx": int(idx),
        "lime_runs": int(lime_runs),
        "shap_runs": int(shap_runs),
        "top_k": int(TOP_K),
        "prediction_confidence": float(prediction_confidence),
        "near_boundary": bool(near_boundary),
        "fidelity_strategy": "r2_reconstruction_additive",
        "fidelity_eval_n": int(eval_n),
        "lime_stability_jaccard": float(lime_stability),
        "lime_mean_jaccard": float(lime_mean_jaccard),
        "lime_std_jaccard": float(lime_std_jaccard),
        "lime_min_jaccard": float(lime_min_jaccard),
        "lime_var_jaccard": float(lime_var_jaccard),
        "lime_is_unstable_var_gt_0_01": bool(lime_is_unstable),
        "lime_supports_h1_jaccard_lt_0_9": bool(lime_supports_h1_jaccard),
        "lime_r2_fidelity": float(lime_fidelity) if lime_fidelity is not None else None,
        "lime_latency_ms": float(lime_latency_ms_mean) if lime_latency_ms_mean is not None else None,
        "lime_latency_ms_avg": float(lime_latency_ms_mean) if lime_latency_ms_mean is not None else None,
        "lime_latency_ms_mean": float(lime_latency_ms_mean) if lime_latency_ms_mean is not None else None,
        "lime_latency_ms_std": float(lime_latency_ms_std) if lime_latency_ms_std is not None else None,
        "lime_latency_ms_p95": float(lime_latency_ms_p95) if lime_latency_ms_p95 is not None else None,
        "lime_confidence_score": lime_score,
        "shap_latency_ms": float(shap_latency_ms_mean) if shap_latency_ms_mean is not None else None,
        "shap_latency_ms_mean": float(shap_latency_ms_mean) if shap_latency_ms_mean is not None else None,
        "shap_latency_ms_std": float(shap_latency_ms_std) if shap_latency_ms_std is not None else None,
        "shap_latency_ms_p95": float(shap_latency_ms_p95) if shap_latency_ms_p95 is not None else None,
        "shap_r2_fidelity": float(shap_fidelity) if shap_fidelity is not None else None,
        "fidelity_difference": float(fidelity_difference) if fidelity_difference is not None else None,
        "shap_confidence_score": float(shap_score) if shap_score is not None else None,
        "shap_stability_jaccard": float(shap_stability) if shap_stability is not None else None,
        "shap_mean_jaccard": float(shap_mean_jaccard) if shap_mean_jaccard is not None else None,
        "shap_std_jaccard": float(shap_std_jaccard) if shap_std_jaccard is not None else None,
        "shap_min_jaccard": float(shap_min_jaccard) if shap_min_jaccard is not None else None,
        "shap_var_jaccard": float(shap_var_jaccard) if shap_var_jaccard is not None else None,
        "shap_is_deterministic": bool(shap_is_deterministic) if shap_is_deterministic is not None else None,
        "deterministic_flag": bool(shap_is_deterministic) if shap_is_deterministic is not None else None,
        "explainer_type": explainer_type,
        "model_output_space": model_output_space if shap_latency_ms is not None else None,
        "explained_class_idx": int(explained_class_idx) if explained_class_idx is not None else None,
        "shap_fx": float(shap_fx) if shap_fx is not None else None,
        "shap_reconstruction": float(shap_reconstruction) if shap_reconstruction is not None else None,
        "shap_additivity_error": float(shap_additivity_error) if shap_additivity_error is not None else None,
        "shap_additivity_warning": bool(shap_additivity_warning) if shap_additivity_warning is not None else None,
        "kernel_nsamples": int(kernel_nsamples_logged) if kernel_nsamples_logged is not None else None,
        "background_size": int(background_size_logged) if background_size_logged is not None else None,
        "background_fixed": bool(background_fixed_logged) if background_fixed_logged is not None else None,
        "cold_start": bool(cold_start_logged),
        "kernelshap_background_size": int(shap_background_size_logged) if shap_background_size_logged is not None else None,
        "latency_ratio": float(latency_ratio) if latency_ratio is not None else None,
        "meets_10x_threshold": bool(meets_10x_threshold) if meets_10x_threshold is not None else None,
        "explainer": "LIME",
        "mean_jaccard": float(lime_mean_jaccard),
        "var_jaccard": float(lime_var_jaccard),
        "r2_fidelity": float(lime_fidelity) if lime_fidelity is not None else None,
        "latency_ms": float(lime_latency_ms_mean) if lime_latency_ms_mean is not None else None,
        "disagreement_flag": bool(disagreement_flag),
        "seed": int(seed),
    }
    append_jsonl(record)

    # --- Save state for persistence ---
    st.session_state["audit_results"] = {
        "lime_stability": lime_stability,
        "lime_mean_jaccard": lime_mean_jaccard,
        "lime_std_jaccard": lime_std_jaccard,
        "lime_min_jaccard": lime_min_jaccard,
        "lime_var_jaccard": lime_var_jaccard,
        "lime_is_unstable": lime_is_unstable,
        "lime_supports_h1_jaccard": lime_supports_h1_jaccard,
        "lime_fidelity": lime_fidelity,
        "lime_latency_ms": lime_latency_ms,
        "lime_score": lime_score,
        "lime_explanations": lime_explanations,
        "lime_latencies": lime_latencies,
        "shap_stability": shap_stability,
        "shap_mean_jaccard": shap_mean_jaccard,
        "shap_std_jaccard": shap_std_jaccard,
        "shap_min_jaccard": shap_min_jaccard,
        "shap_var_jaccard": shap_var_jaccard,
        "shap_is_deterministic": shap_is_deterministic,
        "shap_fidelity": shap_fidelity,
        "shap_latency_ms": shap_latency_ms,
        "latency_ratio": latency_ratio,
        "meets_10x_threshold": meets_10x_threshold,
        "fidelity_difference": fidelity_difference,
        "explainer_type": explainer_type,
        "kernelshap_background_size": shap_background_size_logged,
        "shap_score": shap_score,
        "shap_topk": shap_topk,
        "shap_csv_data": shap_csv_data,
        "disagreement_flag": disagreement_flag,
        "sv_use": sv_use if 'sv_use' in locals() else None,
        "feature_cols": feature_cols,
        "model_name_run": model_name,
        "idx_run": idx,
        "plot_bytes": plot_bytes
    }

if "audit_results" in st.session_state:
    res = st.session_state["audit_results"]
    lime_stability = res["lime_stability"]
    lime_fidelity = res["lime_fidelity"]
    lime_latency_ms = res["lime_latency_ms"]
    lime_score = res["lime_score"]
    lime_explanations = res["lime_explanations"]
    lime_latencies = res["lime_latencies"]
    shap_stability = res["shap_stability"]
    shap_fidelity = res["shap_fidelity"]
    shap_latency_ms = res["shap_latency_ms"]
    shap_score = res["shap_score"]
    shap_topk = res["shap_topk"]
    shap_csv_data = res.get("shap_csv_data")
    disagreement_flag = res["disagreement_flag"]
    sv_use = res["sv_use"]
    feature_cols_run = res["feature_cols"]
    model_name_run = res["model_name_run"]
    idx_run = res["idx_run"]
    plot_bytes = res.get("plot_bytes", None)
    
    # append_jsonl(record) -> Removed to prevent re-logging on refresh

    st.caption("Run logged to outputs/runs.jsonl")

    # --- Download Buttons ---
    st.markdown("---")
    st.subheader("Downloads")
    
    # 1. Download full audit log
    runs_path = "outputs/runs.jsonl"
    try:
        with open(runs_path, "r") as f:
            runs_data = f.read()
        st.download_button(
            label="Download Full Audit Log (JSONL)",
            data=runs_data,
            file_name="audit_runs.jsonl",
            mime="application/json"
        )
    except FileNotFoundError:
        st.caption("No audit log found yet.")
        
    # 1b. Download audit log as CSV
    try:
        if runs_path: # check if exists logic from above roughly
             runs_df = pd.read_json(runs_path, lines=True)
             st.download_button(
                 label="Download Audit Log (CSV)",
                 data=runs_df.to_csv(index=False),
                 file_name="audit_runs.csv",
                 mime="text/csv"
             )
    except (FileNotFoundError, ValueError):
        pass
        
    # 1c. Download High-Res Plot
    if 'plot_bytes' in locals() and plot_bytes is not None:
        st.download_button(
            label="Download Diagnostic Plot (High-Res PNG)",
            data=plot_bytes,
            file_name=f"latency_plot_{model_name_run}_{idx_run}.png",
            mime="image/png"
        )

    # 2. Download current explanations
    lime_df = pd.DataFrame(lime_explanations[0], columns=["Feature", "Weight"])
    st.download_button(
        label="Download LIME Explanations (CSV)",
        data=lime_df.to_csv(index=False),
        file_name=f"lime_explanation_{model_name_run}_{idx_run}.csv",
        mime="text/csv"
    )

    if shap_csv_data:
        st.download_button(
            label="Download SHAP Explanations (CSV)",
            data=shap_csv_data,
            file_name=f"shap_explanation_{model_name_run}_{idx_run}.csv",
            mime="text/csv"
        )
