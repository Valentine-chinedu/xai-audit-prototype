import time
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

def lime_explain_instance(X_train, feature_names, class_names, predict_proba_fn,
                          x_instance, num_features=10, seed=0):
    # LIME has randomness; we control via numpy seed (still may vary due to internals)
    np.random.seed(seed)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
        random_state=seed
    )
    t0 = time.perf_counter()
    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=predict_proba_fn,
        num_features=num_features
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # exp.as_list() returns list of tuples (feature_str, weight)
    # also return map to get raw feature indices for fidelity check
    as_list = exp.as_list()
    as_map = exp.as_map()
    details = {
        "intercept": exp.intercept,
        "local_pred": getattr(exp, "local_pred", None),
        "predict_proba": getattr(exp, "predict_proba", None),
    }
    return as_list, as_map, latency_ms, details

def shap_explain_instance_tree(model, X_background, x_instance, model_output="probability"):
    # TreeSHAP for tree models (RF, XGBoost)
    t0 = time.perf_counter()
    if X_background is None:
        explainer = shap.TreeExplainer(model, model_output=model_output)
    else:
        explainer = shap.TreeExplainer(model, data=X_background, model_output=model_output)
    shap_values = explainer.shap_values(x_instance.reshape(1, -1))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # shap_values can be list for multiclass; for binary, may be array or list
    return shap_values, explainer.expected_value, latency_ms

def shap_explain_instance_kernel(predict_proba_fn, X_background, x_instance, nsamples=100):
    """Kernel SHAP for non-tree models."""
    t0 = time.perf_counter()
    # KernelExplainer expects a function and a background dataset
    explainer = shap.KernelExplainer(predict_proba_fn, X_background)
    
    # Reshape to 2D (1, n_features) as safe practice
    x_2d = x_instance.reshape(1, -1)
    
    shap_values = explainer.shap_values(x_2d, nsamples=nsamples)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    metadata = {
        "kernel_nsamples": int(nsamples),
        "background_size": int(len(X_background)),
    }
    return shap_values, explainer.expected_value, latency_ms, metadata

def topk_from_lime(lime_list, k=10):
    # lime_list: [(feature_desc, weight), ...]
    # feature_desc from LIME may include thresholds like "age <= 25"
    return [f for f, _w in lime_list[:k]]
