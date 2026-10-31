"""
Data Loader & Preprocessing Module
AKTU Machine Learning (BOE-073) Assessment
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, test_size=0.20, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.raw_data = None
        self.feature_names = None
        self.target_names = None
        self.scaler = StandardScaler()
        # Path to the data directory: ML_Disease_Classification_Benchmark/data/
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_data(self, export_csv=True, csv_filename="dataset.csv"):
        """Loads Wisconsin Breast Cancer Dataset and exports local CSV to data/ folder."""
        self.raw_data = load_breast_cancer()
        self.feature_names = list(self.raw_data.feature_names)
        self.target_names = list(self.raw_data.target_names)
        
        X = pd.DataFrame(self.raw_data.data, columns=self.feature_names)
        y = pd.Series(self.raw_data.target, name="target")
        
        if export_csv:
            csv_path = os.path.join(self.data_dir, csv_filename)
            export_df = X.copy()
            export_df["diagnosis"] = y.map({0: "Malignant", 1: "Benign"})
            export_df.to_csv(csv_path, index=False)
            
        return X, y

    def get_summary_stats(self, X, y):
        """Generates descriptive statistics and class balance report."""
        stats = {
            "total_samples": len(y),
            "num_features": X.shape[1],
            "class_counts": {
                "Malignant (Class 0)": int((y == 0).sum()),
                "Benign (Class 1)": int((y == 1).sum())
            },
            "class_percentages": {
                "Malignant %": round((y == 0).mean() * 100, 2),
                "Benign %": round((y == 1).mean() * 100, 2)
            },
            "missing_values": int(X.isnull().sum().sum())
        }
        return stats

    def prepare_splits(self, X, y, export_splits=True):
        """
        Splits into stratified train/test partitions, exports train_data.csv and test_data.csv
        to data/ folder, and standardizes features using StandardScaler.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        if export_splits:
            train_df = X_train.copy()
            train_df["target"] = y_train
            train_df["diagnosis"] = y_train.map({0: "Malignant", 1: "Benign"})
            train_path = os.path.join(self.data_dir, "train_data.csv")
            train_df.to_csv(train_path, index=False)

            test_df = X_test.copy()
            test_df["target"] = y_test
            test_df["diagnosis"] = y_test.map({0: "Malignant", 1: "Benign"})
            test_path = os.path.join(self.data_dir, "test_data.csv")
            test_df.to_csv(test_path, index=False)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_all_scaled = self.scaler.transform(X)
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "X_all_scaled": X_all_scaled,
            "scaler": self.scaler
        }
