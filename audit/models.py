from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



def build_model(model_name: str, random_state: int = 42):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        )
    if model_name == "HistGradientBoosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            l2_regularization=1.0, random_state=random_state
        )
    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=random_state,
            max_iter=500
        )
    raise ValueError(f"Unknown model: {model_name}")
