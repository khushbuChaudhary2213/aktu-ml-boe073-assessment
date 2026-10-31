"""
Streamlit Web Application for AKTU ML (BOE-073) Assessment Project
Interactive Diagnostic Classifier & Model Comparator
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from src.data_loader import DataLoader
from src.models_sklearn import get_sklearn_models

# Page Configuration
st.set_page_config(
    page_title="AKTU ML (BOE-073) Diagnostic Benchmark",
    page_icon="🔬",
    layout="wide"
)

# Title & Header
st.title("🔬 Medical Diagnostic Benchmark & Algorithm Comparison")
st.markdown("""
**Course:** AKTU B.Tech 7th Sem - Machine Learning (`BOE-073`)  
**Curriculum Reference:** Tom M. Mitchell, *Machine Learning*  
**Syllabus Coverage:** Unit II (Decision Trees, MLP), Unit III (Naïve Bayes & Hypothesis Evaluation), Unit IV ($k$-NN)
""")
st.divider()

# Ingest Data via Modular DataLoader
@st.cache_data
def get_data():
    loader = DataLoader(test_size=0.20, random_state=42)
    X, y = loader.load_data(export_csv=True)
    return loader, X, y

loader, X, y = get_data()

# Sidebar Controls
st.sidebar.header("⚙️ Experiment Controls")
test_size = st.sidebar.slider("Test Set Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
selected_model_name = st.sidebar.selectbox(
    "Choose Primary Model for Detailed Inspection",
    [
        "Decision Tree (Unit 2)",
        "Naive Bayes (Unit 3)",
        "k-Nearest Neighbors (Unit 4)",
        "Artificial Neural Net (Unit 2)"
    ]
)

# Prepare dynamic splits
loader.test_size = test_size
splits = loader.prepare_splits(X, y, export_splits=True)
models = get_sklearn_models()

# Compute Comparative Metrics
metrics_list = []
cms = {}

for name, model in models.items():
    model.fit(splits["X_train_scaled"], splits["y_train"])
    preds = model.predict(splits["X_test_scaled"])
    acc = accuracy_score(splits["y_test"], preds)
    prec = precision_score(splits["y_test"], preds)
    rec = recall_score(splits["y_test"], preds)
    f1 = f1_score(splits["y_test"], preds)

    metrics_list.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })
    cms[name] = confusion_matrix(splits["y_test"], preds)

metrics_df = pd.DataFrame(metrics_list)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Comparative Benchmark", "🔍 Detailed Model View", "🧪 Interactive Patient Test"])

with tab1:
    st.subheader("1. Cross-Model Performance Matrix (Unit III: Comparing Learning Algorithms)")
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}"
    }).highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Accuracy & F1-Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_df = metrics_df.set_index("Model")[["Accuracy", "F1-Score"]]
        plot_df.plot(kind="bar", ax=ax, colormap="tab10", width=0.6)
        plt.ylim(0.85, 1.02)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.ylabel("Score")
        st.pyplot(fig)

    with col2:
        st.write("##### Multi-Model ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(splits["X_test_scaled"])[:, 1]
            elif hasattr(model, "decision_function"):
                y_probs = model.decision_function(splits["X_test_scaled"])
            else:
                continue
            fpr, tpr, _ = roc_curve(splits["y_test"], y_probs)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{name[:12]} (AUC={roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax_roc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_roc)

with tab2:
    st.subheader(f"Detailed Evaluation: {selected_model_name}")
    selected_row = metrics_df[metrics_df["Model"] == selected_model_name].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{selected_row['Accuracy']:.2%}")
    m2.metric("Precision", f"{selected_row['Precision']:.2%}")
    m3.metric("Recall", f"{selected_row['Recall']:.2%}")
    m4.metric("F1-Score", f"{selected_row['F1-Score']:.2%}")

    st.write("##### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cms[selected_model_name], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
        xticklabels=loader.target_names, yticklabels=loader.target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig_cm)

with tab3:
    st.subheader("🧪 Live Diagnostic Prediction")
    st.write("Test the active model with sample patient feature values:")

    sample_patient = X.mean().to_dict()
    col_a, col_b = st.columns(2)
    with col_a:
        radius_mean = st.number_input("Mean Radius", value=float(sample_patient['mean radius']))
        texture_mean = st.number_input("Mean Texture", value=float(sample_patient['mean texture']))
        perimeter_mean = st.number_input("Mean Perimeter", value=float(sample_patient['mean perimeter']))
    with col_b:
        area_mean = st.number_input("Mean Area", value=float(sample_patient['mean area']))
        smoothness_mean = st.number_input("Mean Smoothness", value=float(sample_patient['mean smoothness']))
        compactness_mean = st.number_input("Mean Compactness", value=float(sample_patient['mean compactness']))

    # Update sample
    sample_patient['mean radius'] = radius_mean
    sample_patient['mean texture'] = texture_mean
    sample_patient['mean perimeter'] = perimeter_mean
    sample_patient['mean area'] = area_mean
    sample_patient['mean smoothness'] = smoothness_mean
    sample_patient['mean compactness'] = compactness_mean

    sample_array = np.array(list(sample_patient.values())).reshape(1, -1)
    sample_array_scaled = splits["scaler"].transform(sample_array)

    if st.button("Predict Diagnostic Result", type="primary"):
        active_model = models[selected_model_name]
        pred_label = active_model.predict(sample_array_scaled)[0]
        diagnosis = loader.target_names[pred_label]
        
        if pred_label == 1:
            st.success(f"### Predicted Diagnosis: **{diagnosis.upper()}** (Low Risk / Non-Cancerous)")
        else:
            st.error(f"### Predicted Diagnosis: **{diagnosis.upper()}** (High Risk / Cancerous)")

st.sidebar.markdown("---")
st.sidebar.caption("AKTU Machine Learning B.Tech Project Assessment | Dr. A.P.J. Abdul Kalam Technical University")
