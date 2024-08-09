import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean


def euclidean_distance(x, y):
    """Compute Euclidean distance between two sequences using FastDTW."""
    distance, _ = fastdtw(x.T, y.T, dist=scipy_euclidean)
    return distance


def derivative_dtw_distance(x, y):
    """Compute DTW distance between derivatives of two sequences using FastDTW."""
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    distance, _ = fastdtw(dx.T, dy.T, dist=scipy_euclidean)
    return distance
