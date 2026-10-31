"""
Standalone Command-Line Patient Diagnosis Prediction Tool
Allows user/examiner to evaluate test cases from data/test_data.csv or custom profiles.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_loader import DataLoader
from src.models_sklearn import get_sklearn_models

def main():
    parser = argparse.ArgumentParser(description="Patient Biopsy Diagnostic Predictor")
    parser.add_argument("--model", type=str, default="knn",
                        choices=["dt", "nb", "knn", "mlp"],
                        help="Choose model: dt (Decision Tree), nb (Naive Bayes), knn (k-NN), mlp (Neural Net)")
    args = parser.parse_args()

    model_map = {
        "dt": "Decision Tree (Unit 2)",
        "nb": "Naive Bayes (Unit 3)",
        "knn": "k-Nearest Neighbors (Unit 4)",
        "mlp": "Artificial Neural Net (Unit 2)"
    }
    chosen_key = model_map[args.model]

    print("=" * 75)
    print("            AUTOMATED CLINICAL INFERENCE & DIAGNOSIS")
    print(f"            Active Model: {chosen_key}")
    print("=" * 75)

    # 1. Ingest via modular DataLoader
    loader = DataLoader(test_size=0.20, random_state=42)
    X, y = loader.load_data(export_csv=False)
    splits = loader.prepare_splits(X, y, export_splits=False)

    models = get_sklearn_models()
    model = models[chosen_key]
    model.fit(splits["X_train_scaled"], splits["y_train"])

    # 2. Test Cases from actual clinical test split (data/test_data.csv)
    test_csv_path = os.path.join(loader.data_dir, "test_data.csv")
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        # Select one Malignant (target=0) and one Benign (target=1) test instance
        sample_mal = test_df[test_df["target"] == 0].drop(columns=["target", "diagnosis"]).iloc[0].values
        sample_ben = test_df[test_df["target"] == 1].drop(columns=["target", "diagnosis"]).iloc[0].values
    else:
        sample_mal = np.percentile(X[y == 0], 75, axis=0)
        sample_ben = np.percentile(X[y == 1], 25, axis=0)

    test_cases = [
        ("Clinical Test Sample A (from data/test_data.csv)", sample_mal),
        ("Clinical Test Sample B (from data/test_data.csv)", sample_ben)
    ]

    for label, features in test_cases:
        scaled_feat = splits["scaler"].transform(features.reshape(1, -1))
        pred = model.predict(scaled_feat)[0]
        status = "BENIGN (Non-Cancerous / Low Risk)" if pred == 1 else "MALIGNANT (Cancerous / High Risk)"
        
        print(f"\nEvaluating: {label}")
        print(f" -> Mean Cell Radius    : {features[0]:.2f}")
        print(f" -> Mean Cell Texture   : {features[1]:.2f}")
        print(f" -> Mean Cell Perimeter : {features[2]:.2f}")
        print(f" -> Mean Cell Area      : {features[3]:.2f}")
        print(f" -> Mean Smoothness     : {features[4]:.4f}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_feat)[0]
            print(f" -> Class Probabilities: P(Malignant)={probs[0]:.2%}, P(Benign)={probs[1]:.2%}")

        print(f" >>> PREDICTED DIAGNOSIS: {status}")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    main()
