from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

def build_model(model_name: str, random_state: int = 42):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        )
    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed.")
        return XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1, eval_metric="logloss"
        )
    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=random_state,
            max_iter=500
        )
    raise ValueError(f"Unknown model: {model_name}")
