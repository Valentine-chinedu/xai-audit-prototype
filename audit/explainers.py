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
    as_list = exp.as_list()
    return as_list, latency_ms

def shap_explain_instance_tree(model, X_background, x_instance):
    # TreeSHAP for tree models (RF, XGBoost)
    t0 = time.perf_counter()
    # Coerce background and instance to numeric arrays; provide helpful error if conversion fails
    try:
        import pandas as _pd
        if isinstance(X_background, _pd.DataFrame):
            Xb = X_background.values.astype(np.float64)
        else:
            Xb = np.asarray(X_background, dtype=np.float64)
        xi = np.asarray(x_instance.reshape(1, -1), dtype=np.float64)
    except Exception as e:
        raise ValueError(
            "SHAP TreeExplainer requires numeric feature arrays (float). "
            "Ensure preprocessing has converted all features to numeric types. "
            f"Conversion error: {e}"
        )

    explainer = shap.TreeExplainer(model, data=Xb)
    shap_values = explainer.shap_values(xi)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # shap_values can be list for multiclass; for binary, may be array or list
    return shap_values, latency_ms


def shap_explain_instance_kernel(predict_proba_fn, X_background, x_instance, nsamples=100):
    """Kernel SHAP for non-tree models. """
    t0 = time.perf_counter()
    # Coerce background and instance to numeric arrays; provide helpful error if conversion fails
    try:
        import pandas as _pd
        if isinstance(X_background, _pd.DataFrame):
            Xb = X_background.values.astype(np.float64)
        else:
            Xb = np.asarray(X_background, dtype=np.float64)
        xi = np.asarray(x_instance.reshape(1, -1), dtype=np.float64)
    except Exception as e:
        raise ValueError(
            "KernelSHAP requires numeric feature arrays (float). Ensure preprocessing produces numeric features. "
            f"Conversion error: {e}"
        )

    # KernelExplainer can be slow; use a small background sample.
    explainer = shap.KernelExplainer(predict_proba_fn, Xb)
    shap_values = explainer.shap_values(xi, nsamples=nsamples)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return shap_values, latency_ms

def topk_from_lime(lime_list, k=10):
    # lime_list: [(feature_desc, weight), ...]
    # feature_desc from LIME may include thresholds like "age <= 25"
    return [f for f, _w in lime_list[:k]]
