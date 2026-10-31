"""
Hypothesis Testing & Statistical Significance Module
Directly aligns with AKTU Syllabus Unit III:
  - Estimating Hypotheses Accuracy
  - Basics of Sampling Theory
  - Comparing Learning Algorithms via Paired t-test
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

class HypothesisEvaluator:
    def __init__(self, n_splits=10, random_state=42):
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def evaluate_cross_validation(self, models, X, y):
        """
        Estimates generalization accuracy and standard error across 10 folds
        (Tom Mitchell, Chapter 5: Evaluating Hypotheses).
        """
        cv_summary = []
        cv_distributions = {}

        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=self.kfold, scoring="accuracy")
            mean_acc = np.mean(scores)
            variance = np.var(scores, ddof=1)
            std_err = np.std(scores, ddof=1) / np.sqrt(self.n_splits)
            # 95% Confidence Interval using Student's t distribution
            t_crit = stats.t.ppf(0.975, df=self.n_splits - 1)
            ci_lower = mean_acc - t_crit * std_err
            ci_upper = mean_acc + t_crit * std_err

            cv_distributions[name] = scores
            cv_summary.append({
                "Algorithm": name,
                "Mean Accuracy": mean_acc,
                "Sample Variance": variance,
                "Std Error": std_err,
                "95% Conf. Interval": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            })

        return pd.DataFrame(cv_summary), cv_distributions

    def paired_t_test(self, scores_model1, scores_model2, name1="Model 1", name2="Model 2", alpha=0.05):
        """
        Conducts Paired Student's t-test to determine if performance difference is statistically significant.
        Null Hypothesis (H0): Mean performance difference d_bar = 0.
        """
        t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)
        diff_mean = np.mean(scores_model1 - scores_model2)
        significant = p_value < alpha

        return {
            "Comparison": f"{name1} vs {name2}",
            "Mean Difference": round(diff_mean, 4),
            "t-Statistic": round(t_stat, 4),
            "p-Value": round(p_value, 5),
            "Statistically Significant?": "Yes (Reject H0)" if significant else "No (Fail to Reject H0)"
        }

    def run_all_pairwise_tests(self, cv_distributions):
        """Runs pairwise t-tests across all model combinations."""
        names = list(cv_distributions.keys())
        comparisons = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                res = self.paired_t_test(
                    cv_distributions[names[i]],
                    cv_distributions[names[j]],
                    names[i],
                    names[j]
                )
                comparisons.append(res)
        return pd.DataFrame(comparisons)

