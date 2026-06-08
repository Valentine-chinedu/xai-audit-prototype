import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from audit.models import build_model

def test_aopc_calculation():
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
    
    model = build_model("RandomForest", random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)
    
    predict_proba_fn = lambda X_: pipe.predict_proba(X_)
    
    idx = 0
    x_instance = X_test[int(idx)]
    
    # 2. Define AOPC Function (Copied from app.py)
    def calculate_aopc(model_predict_proba_fn, x, top_k_indices, X_background):
        probs_orig = model_predict_proba_fn(x.reshape(1, -1))[0]
        pred_class = np.argmax(probs_orig)
        prob_orig = probs_orig[pred_class]
        
        mins = np.min(X_background, axis=0)
        maxs = np.max(X_background, axis=0)
        means = np.mean(X_background, axis=0)
        
        x_masked = x.copy()
        cumulative_drop = 0.0
        
        drops = []
        for i, idx in enumerate(top_k_indices):
            if x[idx] > means[idx]:
                x_masked[idx] = mins[idx]
            else:
                x_masked[idx] = maxs[idx]
            
            probs_masked = model_predict_proba_fn(x_masked.reshape(1, -1))[0]
            prob_masked = probs_masked[pred_class]
            
            drop_k = max(0.0, prob_orig - prob_masked)
            cumulative_drop += drop_k
            drops.append(drop_k)
            print(f"Step {i+1}: Feature {idx} masked. Drop={drop_k:.4f}")
            
        if not top_k_indices:
            return 0.0
        return cumulative_drop / len(top_k_indices)

    # 3. Test with top 5 features
    
    top_k_indices = [0, 1, 2, 3, 4]
    
    print("\n--- Testing AOPC Calculation ---")
    aopc = calculate_aopc(predict_proba_fn, x_instance, top_k_indices, X_train)
    
    print(f"\nFinal AOPC: {aopc:.4f}")
    
    if aopc > 0:
        print("SUCCESS: AOPC is positive.")
    else:
        print("FAIL: AOPC is 0.0 or negative.")

if __name__ == "__main__":
    test_aopc_calculation()
