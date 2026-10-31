"""
Visualizer Module: Generates publication-ready figures for assessment report.
Saves all figures into the dedicated outputs/ directory.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class Visualizer:
    def __init__(self, output_dir=None):
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_metric_comparison(self, results_df, filename="model_comparison.png"):
        """Bar chart comparing Accuracy, Precision, Recall, F1 across models."""
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        plot_df = results_df.set_index("Algorithm")[metrics]

        plot_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.75)
        plt.title("Algorithm Performance Comparison (AKTU BOE-073)", fontsize=13, weight="bold")
        plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
        plt.xlabel("Machine Learning Algorithm", fontsize=11)
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0.85, 1.02)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_confusion_matrices(self, confusion_matrices, target_names, filename="confusion_matrices.png"):
        """2x2 Grid of Confusion Matrices."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for idx, (name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                xticklabels=target_names, yticklabels=target_names, cbar=False
            )
            axes[idx].set_title(name, fontsize=11, weight="bold")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")

        plt.suptitle("Confusion Matrices Across Models", fontsize=14, weight="bold")
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_roc_curves(self, models, X_test, y_test, filename="roc_curves.png"):
        """ROC curves with Area Under Curve (AUC) scores."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_probs = model.decision_function(X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=13, weight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_cv_distribution(self, cv_distributions, filename="cv_boxplots.png"):
        """Box plot of 10-fold cross validation score distributions."""
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [scores for scores in cv_distributions.values()]
        labels = list(cv_distributions.keys())

        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#cbe4ff", color="#1c4e80"),
                   medianprops=dict(color="#d9534f", linewidth=2))
        ax.set_title("Cross-Validation Accuracy Distribution (10 Folds)", fontsize=13, weight="bold")
        ax.set_ylabel("Accuracy", fontsize=11)
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath
