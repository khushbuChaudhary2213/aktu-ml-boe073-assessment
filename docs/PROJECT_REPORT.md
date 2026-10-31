# ACADEMIC ASSESSMENT REPORT
## Subject: Machine Learning (BOE-073)
### Open Elective - II | B.Tech 4th Year, 7th Semester
**Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow**

---

# Title of Project
## **Comparative Empirical Study and Hypothesis Evaluation of Classical Machine Learning Algorithms for Diagnostic Medical Classification**

---

### Candidate & Submission Information
* **Course:** Bachelor of Technology (Computer Science & Engineering / Information Technology)
* **Semester:** 7th Semester (Final Year)
* **Academic Year:** 2026–2027
* **Subject Code:** BOE-073 / KOE-073: Machine Learning
* **Reference Text:** *Machine Learning*, Tom M. Mitchell, McGraw-Hill Education

---

## 1. ABSTRACT
Automated medical classification systems play a critical role in augmenting clinical diagnostic workflows. This project investigates and benchmarks four fundamental machine learning paradigms prescribed in the AKTU B.Tech Machine Learning (BOE-073) syllabus:
1. **Decision Tree Learning** (Unit II: ID3/Entropy/Information Gain heuristic)
2. **Artificial Neural Networks** (Unit II: Multi-Layer Perceptrons trained via Backpropagation gradient descent)
3. **Bayesian Learning** (Unit III: Gaussian Naïve Bayes under the Conditional Independence assumption)
4. **Instance-Based Learning** (Unit IV: $k$-Nearest Neighbors via Euclidean distance metric)

Using the Wisconsin Breast Cancer Diagnostic Dataset (569 biopsy samples, 30 continuous dimensional features), models are evaluated under the strict mathematical framework of **Unit III (Evaluating Hypotheses)**. In addition to conventional metrics (Accuracy, Precision, Recall, and F1-score), we conduct **10-Fold Cross-Validation** with confidence intervals and **Paired Student’s $t$-Tests** to evaluate whether observed performance discrepancies are statistically significant. Furthermore, pure from-scratch implementations of $k$-NN, Naïve Bayes, and Perceptrons are benchmarked against library models to validate algorithmic fidelity.

---

## 2. FORMAL PROBLEM FORMULATION (TOM M. MITCHELL'S $T, P, E$ TRIPLE)
According to Tom M. Mitchell (Chapter 1, *Machine Learning*):
> *"A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."*

* **Task ($T$):** Binary classification of digitized fine needle aspirate (FNA) cell biopsy samples into `Malignant` (cancerous) or `Benign` (non-cancerous).
* **Performance Measure ($P$):** 
  - Classification Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
  - Sensitivity / Recall: $\frac{TP}{TP + FN}$
  - Precision: $\frac{TP}{TP + FP}$
  - F1-Score: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  - Area Under the ROC Curve (ROC-AUC)
  - Paired Student's $t$-statistic and $p$-value across cross-validation folds.
* **Experience ($E$):** Supervised dataset containing 569 patient tissue instances characterized by 30 continuous diagnostic attributes.

---

## 3. SYLLABUS MAPPING & THEORETICAL FOUNDATIONS

### 3.1 Unit II: Decision Tree Learning
* **Concept:** Non-parametric supervised hierarchical partitioning.
* **Inductive Bias:** Prefers shorter trees over deeper trees (Occam’s razor); prefers trees placing attributes with highest Information Gain near the root.
* **Mathematical Formulations:**
  - **Entropy ($H$):**
    $$H(S) = - \sum_{i=1}^{c} p_i \log_2 (p_i)$$
  - **Information Gain ($Gain$):**
    $$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
* **Avoidance of Overfitting:** Handled via pre-pruning constraint (`max_depth = 4`).

### 3.2 Unit II: Artificial Neural Networks (Backpropagation)
* **Concept:** Feedforward Multi-Layer Perceptron (MLP) mapping non-linear input spaces.
* **Architecture:** Input layer (30 features) $\to$ Hidden Layer 1 (16 neurons, ReLU) $\to$ Hidden Layer 2 (8 neurons, ReLU) $\to$ Output (Sigmoid / Softmax).
* **Backpropagation Weight Update:**
  $$\Delta w_{ji} = \eta \, \delta_j \, x_{ji} + \alpha \, \Delta w_{ji}^{(prev)}$$
  where $\delta_j$ represents the backpropagated error gradient.

### 3.3 Unit III: Bayesian Learning
* **Concept:** Maximum A Posteriori (MAP) classification using Bayes' rule.
* **Inductive Bias:** Conditional independence of all features given the target class $Y$.
* **Formulation:**
  $$P(Y=c | X) \propto P(Y=c) \prod_{j=1}^{d} P(x_j | Y=c)$$
  For continuous features, likelihood $P(x_j | Y=c)$ is modeled via a Gaussian distribution:
  $$P(x_j | Y=c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left( - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$

### 3.4 Unit IV: Instance-Based Learning ($k$-NN)
* **Concept:** Lazy learning algorithm delaying generalization until query time.
* **Distance Metric:** Euclidean Distance:
  $$d(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}$$
* **Inductive Bias:** Target function values for neighboring points in feature space are identical.

---

## 4. SYSTEM ARCHITECTURE & PROJECT STRUCTURE

```text
ML_Disease_Classification_Benchmark/
│
├── src/                               # Modular Source Code Package
│   ├── __init__.py                    # Python Package Initialization
│   ├── data_loader.py                 # Ingestion, EDA, Standardization, Splitter
│   ├── models_sklearn.py              # Scikit-learn Unit 2, 3, 4 Ensembles
│   ├── models_scratch.py              # Pure Python From-Scratch (KNN, Bayes, Perceptron)
│   ├── hypothesis_testing.py          # 10-Fold CV & Paired t-test (Unit 3)
│   └── visualizer.py                  # High-Resolution Publication Plotting Engine
│
├── main.py                            # Master End-to-End Execution Pipeline
├── evaluate_from_scratch.py           # Verification: Custom Scratch vs Library
├── predict.py                         # CLI Inference Tool for Custom Biopsies
├── app.py                             # Interactive Web UI (Streamlit Dashboard)
├── Diagnostic_Benchmark_Notebook.ipynb# Interactive Jupyter Notebook for Lab Submissions
├── dataset.csv                        # Processed Local Data File (569 rows)
├── requirements.txt                   # Dependency Manifest
├── README.md                          # Quick Start & Overview
└── PROJECT_REPORT.md                  # Comprehensive Assessment Document
```

---

## 5. EXPERIMENTAL RESULTS & STATISTICAL ANALYSIS

### 5.1 Test-Set Performance Matrix (80-20 Stratified Split)

| Algorithm | Syllabus Unit | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (ID3/Entropy)** | Unit II | 93.86% | 95.77% | 94.44% | 95.10% |
| **Naive Bayes (Gaussian)** | Unit III | 92.98% | 94.44% | 94.44% | 94.44% |
| **$k$-Nearest Neighbors ($k=5$)**| Unit IV | **95.61%** | 95.89% | **97.22%** | **96.55%** |
| **Artificial Neural Net (MLP)** | Unit II | **95.61%** | **98.55%** | 94.44% | 96.45% |

### 5.2 10-Fold Cross-Validation & Variance Estimation (Unit III)
Cross-validation prevents optimistic partition bias:
* **Artificial Neural Net (MLP):** Mean Accuracy = **96.66%**, Variance = $0.000104$, $95\%$ CI = $[95.93\%, 97.39\%]$
* **$k$-Nearest Neighbors:** Mean Accuracy = **96.31%**, Variance = $0.000135$, $95\%$ CI = $[95.48\%, 97.14\%]$
* **Decision Tree:** Mean Accuracy = **94.20%**, Variance = $0.000110$, $95\%$ CI = $[93.45\%, 94.95\%]$
* **Naive Bayes:** Mean Accuracy = **92.79%**, Variance = $0.000416$, $95\%$ CI = $[91.33\%, 94.25\%]$

### 5.3 Hypothesis Testing via Paired $t$-Test
* **$H_0$ (Null Hypothesis):** There is no significant difference between the true error rates of the two algorithms ($d = 0$).
* **Decision Tree vs. Neural Network:** $p$-value = $0.00004 < 0.05$ $\to$ **Reject $H_0$** (Neural Network performs significantly better than Decision Tree).
* **$k$-NN vs. Neural Network:** $p$-value = $0.34 > 0.05$ $\to$ **Fail to Reject $H_0$** (Performance difference between $k$-NN and Neural Net is not statistically significant).

---
