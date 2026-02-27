import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_lime_parsing():
    # 1. Setup Data & Model
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=42
    )
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Train simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # 2. Run LIME
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=cols,
        class_names=['0', '1'],
        discretize_continuous=True, # This causes "feature > X" strings
        mode="classification",
        random_state=42
    )
    
    x_instance = X[0]
    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model.predict_proba,
        num_features=5
    )
    
    lime_list = exp.as_list()
    print("\nLIME Output (as_list):")
    for item in lime_list:
        print(item)
        
    # 3. Simulate `topk_from_lime` logic from app.py
    # def topk_from_lime(lime_list, k=10):
    #     return [f for f, _w in lime_list[:k]]
    
    top_k_extracted = [f for f, _w in lime_list[:5]]
    print(f"\nExtracted Keys: {top_k_extracted}")
    
    # 4. Simulate `calculate_prediction_drop` finding indices
    # top_k_indices = [all_feature_names.index(f) for f in top_k_names if f in all_feature_names]
    
    matched_indices = []
    for f in top_k_extracted:
        if f in cols:
            matched_indices.append(cols.index(f))
        else:
            print(f"MISSING: '{f}' not found in columns!")
            
    print(f"\nMatched Indices Count: {len(matched_indices)} / 5")
    
    if len(matched_indices) == 0:
        print("FAIL: No LIME features matched raw column names. This explains why fidelity drop is 0.0.")
    else:
        print("SUCCESS: Features matched.")

if __name__ == "__main__":
    test_lime_parsing()
