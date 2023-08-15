import numpy as np
import sklearn.neighbors

from .base import BaseANN


class bruteforce(BaseANN):
    def __init__(self, metric, workers=-1, query_threads=None):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[metric]
        self.workers = workers
        self.name = "bruteforce()"
        print(f'Number of workers: {self.workers}')
        super().__init__(query_threads)

    def fit_single(self, index, X):
        self.nbrs = sklearn.neighbors.NearestNeighbors(algorithm="brute", metric=self.metric, n_jobs=self.workers)
        self.nbrs.fit(X)

    def query(self, v, n):
        return list(self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))
