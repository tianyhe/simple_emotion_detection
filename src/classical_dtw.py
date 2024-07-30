import numpy as np
import json
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real
from helpers import setup_logger


def classical_dtw(x, y, dist=euclidean):
    """Compute Dynamic Time Warping (DTW) distance between two sequences."""
    nx, ny = len(x), len(y)
    cost_matrix = np.zeros((nx + 1, ny + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            cost = dist(x[i - 1], y[j - 1])
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]
            )

    return cost_matrix[nx, ny]


def classical_dtw_distance(x, y):
    """Wrapper function to compute DTW distance between two sequences."""
    return classical_dtw(x.T, y.T, dist=euclidean)


def derivative_dtw_distance(x, y):
    """Compute DTW distance between derivatives of two sequences."""
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
    return classical_dtw_distance(dx, dy)


def classical_dtw_matrix(x, y, dist=euclidean):
    """Compute Dynamic Time Warping (DTW) distance and cost matrix between two sequences."""
    nx, ny = len(x), len(y)
    cost_matrix = np.zeros((nx + 1, ny + 1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            cost = dist(x[i - 1], y[j - 1])
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]
            )

    return cost_matrix[1:, 1:]  # Exclude the initial padding


def traceback(D):
    """
    Traceback for classical DTW to get the optimal path.

    Parameters:
    - D (numpy array): Cost matrix from DTW.

    Returns:
    - path (list): Optimal path for alignment.
    """
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while i > 0 and j > 0:
        tb = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    while i > 0:
        i -= 1
        p.insert(0, i)
        q.insert(0, 0)
    while j > 0:
        j -= 1
        p.insert(0, 0)
        q.insert(0, j)
    return list(zip(p, q))


def preprocess_features(feature_data, scaler=None):
    """Preprocess feature data."""
    if isinstance(feature_data, str):
        try:
            feature_array = np.array(json.loads(feature_data))
        except json.JSONDecodeError:
            print(f"Invalid JSON: {feature_data}")
            raise
    elif isinstance(feature_data, np.ndarray):
        feature_array = feature_data
    else:
        raise TypeError(f"Unexpected type for feature_data: {type(feature_data)}")

    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(1, -1)

    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(feature_array.T).T, scaler
    return scaler.transform(feature_array.T).T, scaler


def classify_emotion(sample, templates, distance_func, feature_weights):
    """Classify emotion based on distance to templates."""
    distances = {}
    features = ["mfccs", "chroma", "spectral_centroid", "zero_crossing_rate", "rms"]

    for emotion, template in templates.items():
        total_distance = 0
        for i, feature in enumerate(features):
            sample_feature, _ = preprocess_features(sample[feature])
            template_feature, _ = preprocess_features(template[feature])

            distance = distance_func(sample_feature, template_feature)
            total_distance += feature_weights[i] * distance

        distances[emotion] = total_distance

    return min(distances, key=distances.get)


class MoodDetectionModel:
    """Mood detection model using DTW."""

    def __init__(self, templates, distance_func, feature_weights=None):
        self.templates = templates
        self.distance_func = distance_func
        self.feature_weights = (
            feature_weights if feature_weights is not None else [1.0] * 5
        )

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [
            classify_emotion(
                sample, self.templates, self.distance_func, self.feature_weights
            )
            for _, sample in X.iterrows()
        ]

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            "templates": self.templates,
            "distance_func": self.distance_func,
            "feature_weights": self.feature_weights,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def optimize_model(df, templates, distance_func, n_iter=50, logger=None):
    """Optimize model parameters using Bayesian optimization."""
    X = df.drop("emotion", axis=1)
    y = df["emotion"]

    model = MoodDetectionModel(templates, distance_func)

    search_space = {
        f"feature_weights_{i}": Real(0.01, 10.0, prior="log-uniform") for i in range(5)
    }

    opt = BayesSearchCV(
        model,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    opt.fit(X, y)

    best_params = [opt.best_params_[f"feature_weights_{i}"] for i in range(5)]
    best_score = opt.best_score_

    if logger:
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")

    results = []
    for i in range(len(opt.cv_results_["params"])):
        results.append(
            {
                "params": opt.cv_results_["params"][i],
                "mean_test_score": opt.cv_results_["mean_test_score"][i],
                "std_test_score": opt.cv_results_["std_test_score"][i],
                "rank_test_score": opt.cv_results_["rank_test_score"][i],
            }
        )

    return best_params, best_score, results


def run_experiments(selected_df, emotion_templates, n_iter=50):
    """Run experiments for classical DTW and derivative DTW."""
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # Set up logging
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/dtw_experiment_{current_time}.log"
    logger = setup_logger("dtw_experiment", log_file)

    distance_metrics = {
        "classical_dtw": classical_dtw_distance,
        "derivative_dtw": derivative_dtw_distance,
    }

    results = []

    for distance_name, distance_func in distance_metrics.items():
        logger.info(f"\nOptimizing for {distance_name} distance:")
        best_params, best_score, opt_results = optimize_model(
            selected_df, emotion_templates, distance_func, n_iter=n_iter, logger=logger
        )

        for result in opt_results:
            results.append(
                {
                    "distance_metric": distance_name,
                    "params": result["params"],
                    "mean_test_score": result["mean_test_score"],
                    "std_test_score": result["std_test_score"],
                    "rank_test_score": result["rank_test_score"],
                }
            )

    results_df = pd.DataFrame(results)
    results_file = f"../results/classical_dtw_experiment_results_{current_time}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)

    logger.info(f"\nResults saved to {results_file}")

    for distance_name in distance_metrics.keys():
        distance_results = results_df[results_df["distance_metric"] == distance_name]
        best_result = distance_results.loc[distance_results["mean_test_score"].idxmax()]
        logger.info(f"\n{distance_name.capitalize()} Distance Results:")
        logger.info(f"Best parameters: {best_result['params']}")
        logger.info(f"Best score: {best_result['mean_test_score']}")
