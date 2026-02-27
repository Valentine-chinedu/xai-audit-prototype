
import sys
import numpy as np
import pandas as pd
from audit.models import build_model
from sklearn.datasets import make_classification
import shap

def verify():
    print("Verifying HistGradientBoostingClassifier...")
    try:
        model = build_model("HistGradientBoosting")
        print("Model built successfully.")
    except Exception as e:
        print(f"Failed to build model: {e}")
        return

    # Check params
    print(f"Model params: {model.get_params()}")

    # Train
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model.fit(X, y)
    print("Model trained successfully.")

    # SHAP check
    print("Checking SHAP compatibility...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:5])
        print("SHAP TreeExplainer ran successfully.")
    except Exception as e:
        print(f"SHAP TreeExplainer failed: {e}")
        # Note: HistGradientBoosting might not be supported by TreeExplainer directly in some versions
        # fallback to Explainer
        try:
            print("Trying generic shap.Explainer...")
            explainer = shap.Explainer(model)
            shap_values = explainer(X[:5])
            print("SHAP Explainer ran successfully.")
        except Exception as e2:
            print(f"SHAP generic Explainer failed: {e2}")

if __name__ == "__main__":
    verify()
