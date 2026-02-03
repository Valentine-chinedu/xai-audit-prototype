import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from audit.models import build_model
from audit.explainers import lime_explain_instance, shap_explain_instance_tree, shap_explain_instance_kernel, topk_from_lime
from audit.metrics import mean_pairwise_jaccard, fidelity_r2
from audit.scoring import explanation_confidence_score
from audit.logging_utils import append_jsonl

from audit.logging_utils import append_jsonl

def safe_float(val):
    """Robustly convert value to float, handling strings with brackets e.g. '[0.5]'."""
    try:
        return float(val)
    except (ValueError, TypeError):
        # handle string representations of lists/arrays
        if isinstance(val, str):
            val = val.strip(" []'\"")
            try:
                return float(val)
            except ValueError:
                pass
        return 0.0

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
    drop_cols = list(missing_frac[missing_frac > 0.5].index)
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
st.title("XAI Reliability Audit Prototype")

# --- Sidebar controls ---
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox("Model", ["RandomForest", "XGBoost", "MLP"])
# Option for non-tree explainability
use_kernel_shap = False
kernel_shap_nsamples = 100
if model_name == "MLP":
    use_kernel_shap = st.sidebar.checkbox("Enable KernelSHAP for MLP (slow)", value=False)
    if use_kernel_shap:
        kernel_shap_nsamples = st.sidebar.slider("KernelSHAP nsamples", 25, 500, 100, step=25)
num_features = st.sidebar.slider("Top-K features", 5, 20, 10)
lime_runs = st.sidebar.slider("LIME repeated runs", 3, 30, 10)
latency_target = st.sidebar.number_input("Real-time latency target (ms)", 50, 2000, 200)

seed = st.sidebar.number_input("Base random seed", 0, 9999, 42)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# Logic to handle data loading and target selection
if uploaded is not None:
    try:
        # Read plain dataframe first to get columns
        df_raw = pd.read_csv(uploaded)
        df_raw = clean_uploaded_df(df_raw)
        
        all_cols = list(df_raw.columns)
        
        # Heuristic for default target
        default_ix = 0
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
st.dataframe(df.head(10), use_container_width=True)

X = df[feature_cols].values
try:
    X = X.astype(np.float64)
except ValueError as e:
    st.error(f"Failed to convert features to numeric. Check for remaining non-numeric columns. Error: {e}")
    st.stop()
y = df[target_col].values

# Encode target for classification
if not locals().get('target_is_continuous', False):
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]
else:
    # Regression or manual binning
    class_names = [str(c) for c in np.unique(y)]

# split
# split
stratify_y = y if (len(np.unique(y)) > 1 and np.min(np.unique(y, return_counts=True)[1]) > 1) else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=int(seed), stratify=stratify_y
)

# pipeline: scale for MLP; scaling doesn't harm trees much for MVP
model = build_model(model_name, random_state=int(seed))
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
pipe.fit(X_train, y_train)

predict_proba_fn = lambda X_: pipe.predict_proba(X_)
predict_fn = lambda X_: pipe.predict(X_)

# choose instance
st.subheader("Select Instance to Explain")
idx = st.number_input("Test set index", min_value=0, max_value=len(X_test)-1, value=0)
x_instance = X_test[int(idx)]
y_true = y_test[int(idx)]
pred = predict_fn(x_instance.reshape(1,-1))[0]
proba = predict_proba_fn(x_instance.reshape(1,-1))[0]

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

    class_names = [str(c) for c in np.unique(y_train)]
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

    # --- LIME fidelity ---
    # Create neighbourhood and see how well LIME surrogate approximates model:
    # MVP approach: use LIME explanation weights as linear approximation proxy is complex;
    # For MVP, we compute "local fidelity" by sampling around instance and fitting a linear model.
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(int(seed))
    n_neigh = 500
    noise = rng.normal(0, 0.5, size=(n_neigh, X_train.shape[1]))
    X_neigh = x_instance.reshape(1, -1) + noise
    y_neigh = predict_proba_fn(X_neigh)[:, 1] if predict_proba_fn(X_neigh).shape[1] > 1 else predict_proba_fn(X_neigh)[:, 0]

    ridge = Ridge(alpha=1.0, random_state=int(seed))
    ridge.fit(X_neigh, y_neigh)
    y_hat = ridge.predict(X_neigh)
    lime_fidelity = fidelity_r2(y_neigh, y_hat)

    # --- SHAP (TreeSHAP for trees, KernelSHAP for MLP) ---
    shap_latency_ms = None
    shap_values = None
    shap_fidelity = None
    shap_stability = 1.0  # deterministic for TreeSHAP; KernelSHAP is approx but treated as deterministic for fixed seed

    if model_name in ["RandomForest", "XGBoost"] or (model_name == "MLP" and use_kernel_shap):
        try:
            shap_expected = None
            if model_name in ["RandomForest", "XGBoost"]:
                # background sample
                bg = X_train[np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)]
                # SHAP expects model, but we have a pipeline; use underlying model with scaled data:
                # easiest MVP: use raw model without scaling for trees by refitting a "tree_pipe" without scaler
                tree_model = build_model(model_name, random_state=int(seed))
                tree_model.fit(X_train, y_train)

                shap_values, shap_latency_ms = shap_explain_instance_tree(tree_model, bg, x_instance)
                
                # For fidelity proxy
                explainer = __import__("shap").TreeExplainer(tree_model, data=bg)
                shap_expected = explainer.expected_value
                model_used_for_fidelity = tree_model

            else: # MLP / KernelSHAP
                 if use_kernel_shap:
                    st.info(f"Running KernelSHAP with {kernel_shap_nsamples} samples...")
                    bg_k = X_train[np.random.choice(len(X_train), size=min(50, len(X_train)), replace=False)]
                    
                    # Optimized call: returns (shap_values, expected_value, latency)
                    shap_values, shap_expected, shap_latency_ms = shap_explain_instance_kernel(
                        predict_proba_fn, bg_k, x_instance, nsamples=int(kernel_shap_nsamples)
                    )
                    
                    # No need to re-instantiate explainer for expected_value!
                    model_used_for_fidelity = None 

            # SHAP fidelity: approximate local prediction by sum(shap)+base_value vs model output
            try:
                # For fidelity check, we need the model output
                # If tree model, we used the tree_model directly. If KernelSHAP, we used predict_proba_fn
                if model_name == "MLP":
                    probs = predict_proba_fn(x_instance.reshape(1, -1))[0]
                else: 
                     # Tree model
                    probs = model_used_for_fidelity.predict_proba(x_instance.reshape(1, -1))[0]
                
                # Dynamic class selection
                pred_class = np.argmax(probs)
                model_out = probs[pred_class]

                if isinstance(shap_values, list):
                    # Multi-output model (e.g. classifier): shap_values is list of arrays
                    sv_use = shap_values[pred_class]
                else:
                    # Single output (e.g. binary classifier just outputting logit or prob)
                    sv_use = shap_values

                # Clean shap_values array (values might be strings '[0.01]')
                try:
                    sv_use = np.array(sv_use, dtype=float)
                except (ValueError, TypeError):
                    # Fallback to element-wise cleaning
                    vf = np.vectorize(safe_float)
                    sv_use = vf(sv_use)

                # Robustly extract scalar base value
                base_use = shap_expected
                if hasattr(shap_expected, "__iter__") and not isinstance(shap_expected, str):
                     base_arr = np.array(shap_expected).ravel()
                     if len(base_arr) > 1:
                         # Use same class index
                         base_use = base_arr[pred_class] if pred_class < len(base_arr) else base_arr[0]
                     elif len(base_arr) == 1:
                         base_use = base_arr[0]
                
                # Use safe_float to handle brackets in string output
                base_val = safe_float(base_use)
                
                # Clamp reconstruction for probabilities
                shap_recon = float(base_val + np.sum(sv_use))
                shap_recon = max(0.0, min(1.0, shap_recon))
                
                # fidelity here as 1 - abs error (bounded), MVP proxy
                shap_fidelity = max(0.0, 1.0 - abs(model_out - shap_recon))
            except Exception as e:
                st.warning(f"SHAP Fidelity Calculation Failed: {e}")
                shap_fidelity = None
        except Exception as e:
            st.error(f"SHAP Explainer Failed: {e}")
            shap_fidelity = None
            shap_values = None

    # --- Scores ---
    lime_score = explanation_confidence_score(lime_stability, lime_fidelity, lime_latency_ms, latency_target_ms=float(latency_target))
    shap_score = None
    if shap_latency_ms is not None and shap_fidelity is not None:
        shap_score = explanation_confidence_score(shap_stability, shap_fidelity, float(shap_latency_ms), latency_target_ms=float(latency_target))

    # --- Disagreement flag (simple, defendable) ---
    disagreement_flag = False
    # Compare top-k from LIME run-0 against SHAP top-k if available
    shap_topk = None
    if model_name in ["RandomForest", "XGBoost"] or (model_name == "MLP" and use_kernel_shap):
        try:
            import shap as _shap
            
            # Reconstruct top-k from shap_values calculated above
            if shap_values is not None:
                if isinstance(shap_values, list):
                    sv_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    sv_use = shap_values
                abs_sv = np.abs(sv_use).ravel()
                order = np.argsort(-abs_sv)
                shap_topk = [feature_cols[i] for i in order[:int(num_features)]]
                j = len(set(shap_topk) & set(topk_from_lime(lime_explanations[0], int(num_features)))) / max(1, len(set(shap_topk) | set(topk_from_lime(lime_explanations[0], int(num_features)))))
                disagreement_flag = (j < 0.5)
        except Exception:
            shap_topk = None

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
        st.write(pd.DataFrame(lime_explanations[0], columns=["Feature", "Weight"]).head(int(num_features)))

    with colB:
        st.markdown("### SHAP")
        if model_name == "MLP" and not use_kernel_shap:
            st.warning("TreeSHAP not available for MLP. Enable KernelSHAP in the sidebar (slow).")
        else:
            stab_label = "Stability"
            if model_name == "MLP" and use_kernel_shap:
                stab_label = "Approximation (KernelSHAP)"
                st.metric(stab_label, f"Running with {kernel_shap_nsamples} samples")
            else:
                st.metric(stab_label, "1.000 (deterministic)")
            st.metric("Fidelity (reconstruction proxy)", f"{shap_fidelity:.3f}" if shap_fidelity is not None else "N/A")
            st.metric("Latency (ms)", f"{float(shap_latency_ms):.1f}" if shap_latency_ms is not None else "N/A")
            st.metric("Explanation Confidence Score", f"{shap_score:.3f}" if shap_score is not None else "N/A")
            if shap_topk:
                st.write("Top-K features (SHAP |abs|):")
                st.write(pd.DataFrame({"Feature": shap_topk}))

    if disagreement_flag:
        st.error("⚠️ Explanation disagreement detected (low overlap between LIME and SHAP top-K). Interpret with caution.")
    else:
        st.success("No major LIME vs SHAP disagreement detected under the current settings.")

    # --- Simple plot: LIME latency distribution ---
    st.subheader("Diagnostics")
    fig = plt.figure()
    plt.hist(lime_latencies, bins=10)
    plt.xlabel("LIME Latency (ms)")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Save high-res plot to bytes for download
    plot_buf = io.BytesIO()
    fig.savefig(plot_buf, format="png", dpi=300, bbox_inches="tight")
    plot_buf.seek(0)
    plot_bytes = plot_buf.getvalue()

    # --- Log run ---
    record = {
        "dataset": "uploaded_csv" if uploaded is not None else "synthetic_demo",
        "model": model_name,
        "instance_idx": int(idx),
        "lime_runs": int(lime_runs),
        "top_k": int(num_features),
        "lime_stability_jaccard": lime_stability,
        "lime_fidelity_r2": lime_fidelity,
        "lime_latency_ms_avg": lime_latency_ms,
        "lime_confidence_score": lime_score,
        "shap_latency_ms": float(shap_latency_ms) if shap_latency_ms is not None else None,
        "shap_fidelity_proxy": float(shap_fidelity) if shap_fidelity is not None else None,
        "shap_confidence_score": float(shap_score) if shap_score is not None else None,
        "disagreement_flag": bool(disagreement_flag),
        "seed": int(seed),
    }
    append_jsonl(record)
    
    # --- Save state for persistence ---
    st.session_state["audit_results"] = {
        "lime_stability": lime_stability,
        "lime_fidelity": lime_fidelity,
        "lime_latency_ms": lime_latency_ms,
        "lime_score": lime_score,
        "lime_explanations": lime_explanations,
        "lime_latencies": lime_latencies,
        "shap_stability": shap_stability,
        "shap_fidelity": shap_fidelity,
        "shap_latency_ms": shap_latency_ms,
        "shap_score": shap_score,
        "shap_topk": shap_topk,
        "disagreement_flag": disagreement_flag,
        "sv_use": sv_use if 'sv_use' in locals() else None,
        "feature_cols": feature_cols,
        "model_name_run": model_name,
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
    disagreement_flag = res["disagreement_flag"]
    sv_use = res["sv_use"]
    feature_cols_run = res["feature_cols"]
    model_name_run = res["model_name_run"]
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

    if shap_topk:
        # Reconstruct full SHAP df if possible, or just top-k
        # For MVP we just reused top-k in display, but let's try to pass the full values if available
        # We reused sv_use and feature_cols earlier
        try:
            # sv_use is already clean numpy array here
            # feature_cols is full list
             shap_df = pd.DataFrame({
                 "Feature": feature_cols_run,
                 "SHAP Value": sv_use.ravel()
             })
             # Sort by magnitude
             shap_df["Abs"] = shap_df["SHAP Value"].abs()
             shap_df = shap_df.sort_values("Abs", ascending=False).drop(columns=["Abs"])
             
             st.download_button(
                label="Download SHAP Explanations (CSV)",
                data=shap_df.to_csv(index=False),
                file_name=f"shap_explanation_{model_name_run}_{idx_run}.csv",
                mime="text/csv"
            )
        except Exception:
            pass
