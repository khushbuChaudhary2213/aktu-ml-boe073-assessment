# AKTU Machine Learning (BOE-073) Assessment Project

## Project Title
**Comparative Study and Empirical Evaluation of Classical Machine Learning Algorithms for Diagnostic Medical Classification**

---

### Course & Curriculum Details
* **University:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
* **Degree / Semester:** B.Tech 4th Year, 7th Semester
* **Subject Code:** BOE-073 / KOE-073 (Open Elective - II: Machine Learning)
* **Prescribed Reference:** *Machine Learning* by Tom M. Mitchell (McGraw-Hill)

---

## 1. Syllabus Mapping & Justification

This project directly implements core algorithms across the official AKTU ML syllabus:

| Syllabus Unit | Core Topic | Implementation in Project |
| :--- | :--- | :--- |
| **Unit II: Decision Tree Learning** | Decision tree algorithm, Information Gain, Entropy, Pruning | `DecisionTreeClassifier` with `criterion="entropy"`, limited depth |
| **Unit II: Artificial Neural Networks** | Perceptron, Delta Rule, Multilayer Networks, Backpropagation | `MLPClassifier` with multi-layer architecture and backpropagation gradient descent |
| **Unit III: Bayesian Learning** | Bayes Theorem, Naïve Bayes Classifier | `GaussianNB` and custom `ScratchGaussianNaiveBayes` (MLE) |
| **Unit III: Evaluating Hypotheses** | Estimating accuracy, sample variance, Comparing Learning Algorithms | Confusion Matrix, Precision, Recall, F1-Score, 10-Fold CV & Paired Student's $t$-tests |
| **Unit IV: Instance-Based Learning** | $k$-Nearest Neighbor ($k$-NN) Learning, Euclidean distance | `KNeighborsClassifier` ($k=5$) and custom `ScratchKNN` |

---

## 2. Complete Modular Project Architecture

```text
ML_Disease_Classification_Benchmark/
│
├── data/                                  # 📁 Datasets Directory
│   ├── dataset.csv                        # Full Wisconsin Breast Cancer dataset (569 rows)
│   ├── train_data.csv                     # Stratified training split (80% = 455 rows)
│   └── test_data.csv                      # Stratified testing split (20% = 114 rows)
│
├── src/                                   # 📁 Core Source Code Package
│   ├── __init__.py                        # Package Initializer
│   ├── data_loader.py                     # Data Ingestion, EDA, Standardization, Splitter
│   ├── models_sklearn.py                  # Scikit-learn Model Ensembles (Units II, III, IV)
│   ├── models_scratch.py                  # ⭐ Pure Python Math Models (k-NN, Naïve Bayes, Perceptron)
│   ├── hypothesis_testing.py              # ⭐ Unit III: 10-Fold CV & Paired Student's t-Test
│   └── visualizer.py                      # Multi-Figure Publication Plotting Engine
│
├── tests/                                 # 📁 Automated Verification & Test Suite
│   ├── __init__.py                        # Test Package Initializer
│   ├── test_pipeline.py                   # Automated Unit Tests (unittest framework)
│   └── test_scratch_models.py             # Scratch vs. Scikit-Learn Verification Test
│
├── notebooks/                             # 📁 Interactive Notebooks Directory
│   └── Diagnostic_Benchmark_Notebook.ipynb# Complete Jupyter Notebook for Lab/Colab Submissions
│
├── docs/                                  # 📁 Reports & Academic Documentation
│   ├── PROJECT_REPORT.md                  # Comprehensive Formal Assessment Report with Viva Q&A
│   ├── PRESENTATION_SLIDES.md             # Slide-by-slide script & viva talking points
│   └── Project_Presentation.pptx          # 📊 High-Quality 16:9 PowerPoint Presentation Deck
│
├── outputs/                               # 📁 Generated Figures & Visual Assets
│   ├── model_comparison.png               # High-resolution performance comparison chart
│   ├── confusion_matrices.png             # 2x2 Grid of Confusion Matrix Heatmaps
│   ├── roc_curves.png                     # Multi-Model ROC Curves with AUC Scores
│   └── cv_boxplots.png                    # 10-Fold Cross-Validation Accuracy Distribution Boxplots
│
├── main.py                                # Master End-to-End Execution Pipeline
├── predict.py                             # CLI Inference Tool for Test Cases
├── evaluate_from_scratch.py               # Root shortcut for scratch verification
├── app.py                                 # Interactive Web Dashboard (Streamlit GUI)
└── requirements.txt                       # Project Dependency Manifest
```

---

## 3. How to Run Every Component

### 1. Run Master Pipeline (Ingestion, Training, Statistical Tests, Plots)
```bash
python main.py
```
*Outputs: Generates `data/` splits, evaluates models, prints 10-fold CV & paired $t$-test tables, and saves 4 plots to `outputs/`.*

### 2. Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Run Algorithmic Proof: Scratch vs. Scikit-Learn Verification
```bash
python evaluate_from_scratch.py
```
*Verifies 100% agreement between custom pure-Python mathematical models and Scikit-Learn.*

### 4. Run Patient Diagnosis Prediction on `data/test_data.csv`
```bash
python predict.py --model knn
python predict.py --model mlp
```

### 5. Launch Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```

### 6. Interactive Jupyter Notebook
Open `notebooks/Diagnostic_Benchmark_Notebook.ipynb` in VS Code or Jupyter.
