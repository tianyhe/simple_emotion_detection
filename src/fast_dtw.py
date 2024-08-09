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
from collections import defaultdict
import numbers

def fastdtw(x, y, radius=1, dist=None):
    """
    Compute the approximate distance between two time series using FastDTW.
    
    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    radius : int
        Radius of the window used for approximation.
    dist : function or int
        Distance metric. If int, p-norm is used. If function, it computes distance between points.

    Returns
    -------
    distance : float
        Approximate distance between the two time series.
    path : list
        List of index pairs representing the optimal path.
    """
    x, y, dist = _prep_inputs(x, y, dist)
    return _fastdtw(x, y, radius, dist)

def _prep_inputs(x, y, dist):
    x = np.asarray(x, dtype='float')
    y = np.asarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('Second dimension of x and y must be the same')
    
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        dist = np.abs
    elif isinstance(dist, numbers.Number):
        dist = lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), ord=dist)
    
    return x, y, dist

def _fastdtw(x, y, radius, dist):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = _reduce_by_half(x)
    y_shrinked = _reduce_by_half(y)
    distance, path = _fastdtw(x_shrinked, y_shrinked, radius, dist)
    window = _expand_window(path, len(x), len(y), radius)
    return _dtw(x, y, window, dist)

def _reduce_by_half(x):
    return [(x[i] + x[i + 1]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

def _expand_window(path, len_x, len_y, radius):
    path_set = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius + 1)
                     for b in range(-radius, radius + 1)):
            path_set.add((a, b))

    window_set = set()
    for i, j in path_set:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_set.add((a, b))

    window = []
    start_j = 0
    for i in range(len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_set:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window

def dtw(x, y, dist=None):
    """
    Compute the exact distance between two time series using DTW.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.
    dist : function or int
        Distance metric. If int, p-norm is used. If function, it computes distance between points.

    Returns
    -------
    distance : float
        Exact distance between the two time series.
    path : list
        List of index pairs representing the optimal path.
    """
    x, y, dist = _prep_inputs(x, y, dist)

    return _dtw(x, y, None, dist)

def _dtw(x, y, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    
    for i, j in window:
        dt = dist(x[i - 1], y[j - 1])
        D[i, j] = min(
            (D[i - 1, j][0] + dt, i - 1, j),
            (D[i, j - 1][0] + dt, i, j - 1),
            (D[i - 1, j - 1][0] + dt, i - 1, j - 1),
            key=lambda a: a[0]
        )
    
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    
    return (D[len_x, len_y][0], path)


def euclidean_distance(x, y):
    """Compute Euclidean distance between two sequences using FastDTW."""
    distance, _ = fastdtw(x.T, y.T, radius=1, dist=euclidean)
    return distance



def derivative_dtw_distance(x, y):
    """Compute DTW distance between derivatives of two sequences using FastDTW."""
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    distance, _ = fastdtw(
        dx.T, dy.T, radius=1, dist=euclidean)  
    return distance


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
    """Mood detection model using FastDTW."""

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
    """Run experiments for FastDTW and derivative FastDTW."""
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    # Set up logging
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/fastdtw_experiment_{current_time}.log"
    logger = setup_logger("fastdtw_experiment", log_file)

    distance_metrics = {
        "fastdtw": euclidean_distance,
        "derivative_fastdtw": derivative_dtw_distance,
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
    results_file = f"../results/fastdtw_experiment_results_{current_time}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)

    logger.info(f"\nResults saved to {results_file}")

    for distance_name in distance_metrics.keys():
        distance_results = results_df[results_df["distance_metric"] == distance_name]
        best_result = distance_results.loc[distance_results["mean_test_score"].idxmax()]
        logger.info(f"\n{distance_name.capitalize()} Distance Results:")
        logger.info(f"Best parameters: {best_result['params']}")
        logger.info(f"Best score: {best_result['mean_test_score']}")
