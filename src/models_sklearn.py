"""
Scikit-Learn Model Ensembles mapped to AKTU BOE-073 Syllabus
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def get_sklearn_models():
    """
    Returns a dictionary of models mapped directly to Tom Mitchell's textbook chapters:
      - Unit II : Decision Trees (Entropy/Information Gain)
      - Unit II : Artificial Neural Networks (Backpropagation / MLP)
      - Unit III: Bayesian Learning (Gaussian Naive Bayes)
      - Unit IV : Instance-Based Learning (k-Nearest Neighbors)
    """
    return {
        "Decision Tree (Unit 2)": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=4,
            random_state=42
        ),
        "Naive Bayes (Unit 3)": GaussianNB(),
        "k-Nearest Neighbors (Unit 4)": KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean"
        ),
        "Artificial Neural Net (Unit 2)": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=600,
            activation="relu",
            solver="adam",
            random_state=42
        )
    }

