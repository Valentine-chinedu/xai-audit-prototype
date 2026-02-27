import json
import os
from datetime import datetime

AUDIT_SCHEMA_COLUMNS = [
    "dataset",
    "model_name",
    "model",
    "instance_idx",
    "lime_runs",
    "shap_runs",
    "top_k",
    "seed",
    "audit_schema",
    "prediction_confidence",
    "near_boundary",
    "fidelity_strategy",
    "fidelity_eval_n",
    "lime_stability_jaccard",
    "lime_mean_jaccard",
    "lime_std_jaccard",
    "lime_min_jaccard",
    "lime_var_jaccard",
    "lime_is_unstable_var_gt_0_01",
    "lime_supports_h1_jaccard_lt_0_9",
    "lime_r2_fidelity",
    "lime_fidelity_r2",
    "lime_confidence_score",
    "lime_latency_ms",
    "lime_latency_ms_avg",
    "lime_latency_ms_mean",
    "lime_latency_ms_std",
    "lime_latency_ms_p95",
    "shap_stability_jaccard",
    "shap_mean_jaccard",
    "shap_std_jaccard",
    "shap_min_jaccard",
    "shap_var_jaccard",
    "shap_is_deterministic",
    "deterministic_flag",
    "shap_r2_fidelity",
    "shap_fidelity_proxy",
    "shap_confidence_score",
    "shap_latency_ms",
    "shap_latency_ms_mean",
    "shap_latency_ms_std",
    "shap_latency_ms_p95",
    "latency_ratio",
    "meets_10x_threshold",
    "explainer",
    "explainer_type",
    "mean_jaccard",
    "var_jaccard",
    "r2_fidelity",
    "latency_ms",
    "model_output_space",
    "explained_class_idx",
    "shap_additivity_error",
    "shap_additivity_warning",
    "shap_fx",
    "shap_reconstruction",
    "kernel_nsamples",
    "background_size",
    "background_fixed",
    "cold_start",
    "kernelshap_background_size",
    "fidelity_difference",
    "disagreement_flag",
]


def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)

def _normalize_record_schema(record: dict) -> dict:
    out = dict(record)
    if "audit_schema" not in out:
        out["audit_schema"] = "vB"
    for col in AUDIT_SCHEMA_COLUMNS:
        out.setdefault(col, None)
    return out

def append_jsonl(record: dict, filepath="outputs/runs.jsonl"):
    ensure_outputs_dir(os.path.dirname(filepath) or "outputs")
    record = _normalize_record_schema(record)
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def write_jsonl(record: dict, filepath="outputs/runs.jsonl"):
    """Write (overwrite) a record to a JSONL file."""
    ensure_outputs_dir(os.path.dirname(filepath) or "outputs")
    record = _normalize_record_schema(record)
    record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
