import numpy as np
from scipy.spatial.distance import euclidean


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
