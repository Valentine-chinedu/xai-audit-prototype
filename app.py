import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from audit.models import build_model
from audit.explainers import lime_explain_instance, shap_explain_instance_tree, topk_from_lime
from audit.metrics import mean_pairwise_jaccard, fidelity_r2
from audit.scoring import explanation_confidence_score
from audit.logging_utils import append_jsonl

st.set_page_config(page_title="XAI Reliability Audit", layout="wide")
st.title("XAI Reliability Audit Prototype")

# --- Sidebar controls ---
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox("Model", ["RandomForest", "XGBoost", "MLP"])
num_features = st.sidebar.slider("Top-K features", 5, 20, 10)
lime_runs = st.sidebar.slider("LIME repeated runs", 3, 30, 10)
latency_target = st.sidebar.number_input("Real-time latency target (ms)", 50, 2000, 200)

seed = st.sidebar.number_input("Base random seed", 0, 9999, 42)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
target_col = st.sidebar.text_input("Target column name (for CSV)", value="target")

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

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in uploaded CSV.")
        st.stop()
    feature_cols = [c for c in df.columns if c != target_col]
else:
    df, feature_cols, target_col = load_demo_data()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

X = df[feature_cols].values
y = df[target_col].values

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=int(seed), stratify=y if len(np.unique(y)) > 1 else None
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

    # --- SHAP (TreeSHAP only for tree models) ---
    shap_latency_ms = None
    shap_values = None
    shap_fidelity = None
    shap_stability = 1.0  # deterministic for TreeSHAP

    if model_name in ["RandomForest", "XGBoost"]:
        # background sample
        bg = X_train[np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)]
        # SHAP expects model, but we have a pipeline; use underlying model with scaled data:
        # easiest MVP: use raw model without scaling for trees by refitting a "tree_pipe" without scaler
        from sklearn.pipeline import Pipeline
        tree_model = build_model(model_name, random_state=int(seed))
        tree_model.fit(X_train, y_train)

        shap_values, shap_latency_ms = shap_explain_instance_tree(tree_model, bg, x_instance)

        # SHAP fidelity: approximate local prediction by sum(shap)+base_value vs model output
        # This varies by SHAP output structure; MVP handles binary case commonly.
        try:
            explainer = __import__("shap").TreeExplainer(tree_model, data=bg)
            base = explainer.expected_value
            sv = explainer.shap_values(x_instance.reshape(1, -1))
            model_out = tree_model.predict_proba(x_instance.reshape(1, -1))[0, 1] if len(tree_model.classes_) > 1 else tree_model.predict_proba(x_instance.reshape(1, -1))[0, 0]

            if isinstance(sv, list):
                sv_use = sv[1] if len(sv) > 1 else sv[0]
                base_use = base[1] if isinstance(base, (list, np.ndarray)) and len(base) > 1 else base
            else:
                sv_use = sv
                base_use = base

            shap_recon = float(base_use + np.sum(sv_use))
            # fidelity here as 1 - abs error (bounded), MVP proxy
            shap_fidelity = max(0.0, 1.0 - abs(model_out - shap_recon))
        except Exception:
            shap_fidelity = None

    # --- Scores ---
    lime_score = explanation_confidence_score(lime_stability, lime_fidelity, lime_latency_ms, latency_target_ms=float(latency_target))
    shap_score = None
    if shap_latency_ms is not None and shap_fidelity is not None:
        shap_score = explanation_confidence_score(shap_stability, shap_fidelity, float(shap_latency_ms), latency_target_ms=float(latency_target))

    # --- Disagreement flag (simple, defendable) ---
    disagreement_flag = False
    # Compare top-k from LIME run-0 against SHAP top-k if available
    shap_topk = None
    if model_name in ["RandomForest", "XGBoost"]:
        try:
            import shap as _shap
            tree_model = build_model(model_name, random_state=int(seed))
            tree_model.fit(X_train, y_train)
            bg = X_train[np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)]
            exp = _shap.TreeExplainer(tree_model, data=bg)
            sv = exp.shap_values(x_instance.reshape(1, -1))
            if isinstance(sv, list):
                sv_use = sv[1] if len(sv) > 1 else sv[0]
            else:
                sv_use = sv
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
        if model_name not in ["RandomForest", "XGBoost"]:
            st.warning("TreeSHAP not available for MLP in this MVP (we can add KernelSHAP later).")
        else:
            st.metric("Stability", "1.000 (deterministic)")
            st.metric("Fidelity (reconstruction proxy)", f"{shap_fidelity:.3f}" if shap_fidelity is not None else "N/A")
            st.metric("Latency (ms)", f"{float(shap_latency_ms):.1f}")
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

    st.caption("Run logged to outputs/runs.jsonl")
