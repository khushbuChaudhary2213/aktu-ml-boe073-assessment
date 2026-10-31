"""
From-Scratch Algorithmic Implementations (Pure Python & NumPy)
Implements Tom M. Mitchell's mathematical formulations for:
  1. ScratchKNN (Unit IV: Instance-Based Learning)
  2. ScratchGaussianNaiveBayes (Unit III: Bayesian Learning)
  3. ScratchPerceptron (Unit II: Perceptrons & Delta Rule)
"""

import numpy as np

class ScratchKNN:
    """
    Instance-Based Learning: k-Nearest Neighbor from scratch.
    Inductive Bias: Instances near each other in Euclidean feature space share identical class labels.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Lazy learner: memorizes the training instances
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances between x and all training examples
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Identify indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority voting
        counts = np.bincount(k_nearest_labels)
        return np.argmax(counts)


class ScratchGaussianNaiveBayes:
    """
    Bayesian Learning: Gaussian Naive Bayes from scratch using Maximum Likelihood Estimation (MLE).
    Inductive Bias: Attributes are conditionally independent given the target class value.
    """
    def __init__(self):
        self.classes = None
        self.parameters = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        n_samples = len(y)

        # Estimate prior, mean, and variance for each class
        for c in self.classes:
            X_c = X[y == c]
            prior = len(X_c) / n_samples
            mean = np.mean(X_c, axis=0)
            # Add small epsilon to variance to prevent division by zero
            var = np.var(X_c, axis=0) + 1e-9
            
            self.parameters[c] = {
                "prior": prior,
                "mean": mean,
                "var": var
            }

    def _calculate_gaussian_log_likelihood(self, class_idx, x):
        mean = self.parameters[class_idx]["mean"]
        var = self.parameters[class_idx]["var"]
        # Gaussian PDF in log-space: log P(x_i | c)
        log_pdf = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
        return np.sum(log_pdf)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.parameters[c]["prior"])
            likelihood = self._calculate_gaussian_log_likelihood(c, x)
            posterior = prior + likelihood
            posteriors.append((posterior, c))
        # Return class with Maximum A Posteriori (MAP)
        posteriors.sort(key=lambda item: item[0], reverse=True)
        return posteriors[0][1]


class ScratchPerceptron:
    """
    Artificial Neural Networks: Single-layer Perceptron with Delta / Hebbian update rule.
    Unit II: Rosenblatt Perceptron.
    """
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.array(X)
        # Convert binary classes {0, 1} to {-1, +1} for standard perceptron rule
        y_binary = np.where(np.array(y) == 0, -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = 1 if linear_output >= 0 else -1
                update = self.lr * (y_binary[i] - prediction)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

