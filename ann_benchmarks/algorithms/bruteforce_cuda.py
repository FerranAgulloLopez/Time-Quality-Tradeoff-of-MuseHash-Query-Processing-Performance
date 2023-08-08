import cuml.neighbors
import numpy as np

from .base import BaseANN


class BruteForceCuda(BaseANN):
    def __init__(self, metric, workers=-1):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForceCuda doesn't support metric %s" % metric)
        self._metric = metric
        self.workers = workers
        self.name = "BruteForce()"
        print(f'Number of workers: {self.workers}')

    def fit(self, X):
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[self._metric]
        self._nbrs = cuml.neighbors.NearestNeighbors(metric=metric)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))
