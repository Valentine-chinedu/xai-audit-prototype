import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from audit.models import build_model

def test_fidelity_calculation():
    # 1. Setup Data & Model (similar to app.py)
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Using RandomForest as per default/common case
    model = build_model("RandomForest", random_state=42)
    # Note: app.py uses a pipeline with StandardScaler
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    
    predict_proba_fn = lambda X_: pipe.predict_proba(X_)
    
    # 2. Select an instance
    idx = 0
    x_instance = X_test[int(idx)]
    
    # Feature indices
    cols = [f"f{i}" for i in range(X.shape[1])]

    def calculate_drop(strategy="mean"):
        top_k_indices = [0, 1, 2, 3, 4] # Top 5 informative
        
        # Original prediction
        probs_orig = predict_proba_fn(x_instance.reshape(1, -1))[0]
        pred_class = np.argmax(probs_orig)
        prob_orig = probs_orig[pred_class]
        
        # Masked prediction
        x_masked = x_instance.copy()
        
        if strategy == "mean":
            means = np.mean(X_train, axis=0)
            for idx in top_k_indices:
                x_masked[idx] = means[idx]
        elif strategy == "zero":
             for idx in top_k_indices:
                x_masked[idx] = 0.0
        elif strategy == "max_inverse":
             # Set to max value in opposite direction or just min/max of global
             for idx in top_k_indices:
                 if x_instance[idx] > np.mean(X_train[:, idx]):
                     x_masked[idx] = np.min(X_train[:, idx])
                 else:
                     x_masked[idx] = np.max(X_train[:, idx])

        probs_masked = predict_proba_fn(x_masked.reshape(1, -1))[0]
        # We care about the probability of the *original* predicted class dropping
        prob_masked = probs_masked[pred_class]
        
        drop = prob_orig - prob_masked
        print(f"Strategy: {strategy:12} | Orig: {prob_orig:.4f} | Masked: {prob_masked:.4f} | Drop: {drop:.4f}")
        return drop

    print("\n--- Testing Fidelity Strategies ---")
    calculate_drop("mean")
    calculate_drop("zero")
    calculate_drop("max_inverse")

if __name__ == "__main__":
    test_fidelity_calculation()
