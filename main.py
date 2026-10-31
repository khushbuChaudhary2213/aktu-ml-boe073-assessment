"""
AKTU Machine Learning (BOE-073) Assessment Project
Project: Comparative Study and Evaluation of Machine Learning Algorithms for Diagnostic Classification
Mapped Syllabus Units:
  - Unit II : Decision Tree Learning & Artificial Neural Networks (MLP / Backpropagation)
  - Unit III: Evaluating Hypotheses (Accuracy, k-Fold CV, Paired t-tests) & Bayesian Learning (Naive Bayes)
  - Unit IV : Instance-Based Learning (k-Nearest Neighbors)
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_loader import DataLoader
from src.models_sklearn import get_sklearn_models
from src.hypothesis_testing import HypothesisEvaluator
from src.visualizer import Visualizer

def run_project():
    print("=" * 85)
    print("           AKTU 4th Year B.Tech ML (BOE-073) Assessment Project")
    print("      Comparative Study & Evaluation of ML Algorithms for Disease Classification")
    print("=" * 85)

    # 1. Load Dataset
    print("\n[Step 1/6] Ingesting & Preprocessing Dataset (Wisconsin Breast Cancer)...")
    loader = DataLoader(test_size=0.20, random_state=42)
    X, y = loader.load_data(export_csv=True, csv_filename="dataset.csv")
    stats = loader.get_summary_stats(X, y)

    print(f" -> Total Instances    : {stats['total_samples']}")
    print(f" -> Number of Features : {stats['num_features']}")
    print(f" -> Class Distribution : {stats['class_counts']} ({stats['class_percentages']})")
    print(f" -> Missing Values     : {stats['missing_values']} (Clean)")
    print(f" -> Saved to Directory : 'data/dataset.csv'")

    # 2. Train-Test Split & Feature Scaling
    print("\n[Step 2/6] Preparing Stratified Splits & Standardizing Features...")
    splits = loader.prepare_splits(X, y, export_splits=True)
    X_train_s = splits["X_train_scaled"]
    X_test_s = splits["X_test_scaled"]
    X_all_s = splits["X_all_scaled"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    print(f" -> Training Instances : {X_train_s.shape[0]} (Saved to 'data/train_data.csv')")
    print(f" -> Testing Instances  : {X_test_s.shape[0]} (Saved to 'data/test_data.csv')")

    # 3. Model Initialization
    print("\n[Step 3/6] Initializing Models mapped to Tom Mitchell Curriculum...")
    models = get_sklearn_models()
    for name in models.keys():
        print(f"    • {name}")

    # 4. Model Training & Basic Metrics
    print("\n[Step 4/6] Training models & computing classification metrics...")
    results = []
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 85)
    print("                      1. MODEL PERFORMANCE SUMMARY TABLE")
    print("=" * 85)
    print(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format
    }))
    print("=" * 85)

    # 5. Unit III: Hypothesis Testing & Statistical Significance
    print("\n[Step 5/6] Executing Unit III Hypothesis Testing (10-Fold CV & Paired t-tests)...")
    evaluator = HypothesisEvaluator(n_splits=10, random_state=42)
    cv_summary, cv_distributions = evaluator.evaluate_cross_validation(models, X_all_s, y)

    print("\n" + "-" * 85)
    print("          2. 10-FOLD CROSS-VALIDATION & VARIANCE ESTIMATION (UNIT III)")
    print("-" * 85)
    print(cv_summary.to_string(index=False, formatters={
        "Mean Accuracy": "{:.4f}".format,
        "Sample Variance": "{:.6f}".format,
        "Std Error": "{:.4f}".format
    }))
    print("-" * 85)

    print("\n" + "-" * 85)
    print("          3. PAIRED STUDENT'S t-TEST: COMPARING LEARNING ALGORITHMS (UNIT III)")
    print("-" * 85)
    pairwise_tests = evaluator.run_all_pairwise_tests(cv_distributions)
    print(pairwise_tests.to_string(index=False))
    print("-" * 85)

    # 6. Generate Plots
    print("\n[Step 6/6] Generating High-Resolution Evaluation Figures in 'outputs/'...")
    visualizer = Visualizer()
    p1 = visualizer.plot_metric_comparison(results_df)
    p2 = visualizer.plot_confusion_matrices(confusion_matrices, loader.target_names)
    p3 = visualizer.plot_roc_curves(models, X_test_s, y_test)
    p4 = visualizer.plot_cv_distribution(cv_distributions)

    print(f" -> [Plot 1] Metric comparison bar chart : 'outputs/{os.path.basename(p1)}'")
    print(f" -> [Plot 2] Confusion matrices grid     : 'outputs/{os.path.basename(p2)}'")
    print(f" -> [Plot 3] ROC & AUC curves            : 'outputs/{os.path.basename(p3)}'")
    print(f" -> [Plot 4] 10-Fold CV boxplot variance : 'outputs/{os.path.basename(p4)}'")

    print("\n[SUCCESS] Assessment execution finished successfully!")
    print("   All datasets, models, tests, and plots are neatly organized into their folders.")

if __name__ == "__main__":
    run_project()
