"""
Automated Unit Tests for ML Pipeline
Verifies data loading, train/test splitting, and model execution.
"""

import unittest
import os
import numpy as np
from src.data_loader import DataLoader
from src.models_sklearn import get_sklearn_models
from src.models_scratch import ScratchKNN, ScratchGaussianNaiveBayes

class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(test_size=0.20, random_state=42)
        cls.X, cls.y = cls.loader.load_data(export_csv=True)
        cls.splits = cls.loader.prepare_splits(cls.X, cls.y, export_splits=True)

    def test_data_splits_and_shapes(self):
        """Test dataset shape and train-test partition ratio."""
        self.assertEqual(self.X.shape[0], 569)
        self.assertEqual(self.X.shape[1], 30)
        self.assertEqual(self.splits["X_train_scaled"].shape[0], 455)
        self.assertEqual(self.splits["X_test_scaled"].shape[0], 114)

    def test_exported_datasets_exist(self):
        """Test that data/ folder contains dataset.csv, train_data.csv, and test_data.csv."""
        data_dir = self.loader.data_dir
        self.assertTrue(os.path.exists(os.path.join(data_dir, "dataset.csv")))
        self.assertTrue(os.path.exists(os.path.join(data_dir, "train_data.csv")))
        self.assertTrue(os.path.exists(os.path.join(data_dir, "test_data.csv")))

    def test_sklearn_models_training(self):
        """Test all four syllabus models train and produce valid accuracy scores."""
        models = get_sklearn_models()
        for name, model in models.items():
            model.fit(self.splits["X_train_scaled"], self.splits["y_train"])
            preds = model.predict(self.splits["X_test_scaled"])
            acc = np.mean(preds == self.splits["y_test"])
            self.assertGreater(acc, 0.90, f"Model {name} accuracy fell below expected 90% benchmark")

    def test_scratch_knn(self):
        """Test custom Scratch k-NN model."""
        s_knn = ScratchKNN(k=5)
        s_knn.fit(self.splits["X_train_scaled"], self.splits["y_train"].values)
        preds = s_knn.predict(self.splits["X_test_scaled"])
        acc = np.mean(preds == self.splits["y_test"])
        self.assertGreater(acc, 0.92)

    def test_scratch_naive_bayes(self):
        """Test custom Scratch Gaussian Naive Bayes model."""
        s_nb = ScratchGaussianNaiveBayes()
        s_nb.fit(self.splits["X_train_scaled"], self.splits["y_train"].values)
        preds = s_nb.predict(self.splits["X_test_scaled"])
        acc = np.mean(preds == self.splits["y_test"])
        self.assertGreater(acc, 0.90)

if __name__ == "__main__":
    unittest.main()

