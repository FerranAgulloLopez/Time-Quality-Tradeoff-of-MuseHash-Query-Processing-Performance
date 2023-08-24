import numpy as np
import sklearn.neighbors

from .base import BaseANN


class balltree(BaseANN):
    def __init__(self, metric, workers=-1, query_processes=None, leaf_size=40):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[metric]
        self.name = "balltree()"
        self.leaf_size = leaf_size
        if workers != 1:
            raise Exception('BallTree algorirthm does not implement data parallelism')
        super().__init__(query_processes)

    def fit(self, index, X):
        self.nbrs = sklearn.neighbors.BallTree(X, metric=self.metric, leaf_size=self.leaf_size)

    def query(self, v, n):
        return self.nbrs.query(np.expand_dims(v, axis=0), return_distance=False, k=n)[0]

    def query_batch(self, v, n):
        return self.nbrs.query(v, return_distance=False, k=n)

    def query_with_distances(self, v, n):
        (distances, positions) = self.nbrs.query(np.expand_dims(v, axis=0), return_distance=True, k=n)
        return zip(list(positions[0]), list(distances[0]))
