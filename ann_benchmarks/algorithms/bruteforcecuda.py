import os

import numpy as np

from .base import BaseANN


class bruteforcecuda(BaseANN):
    def __init__(self, metric, workers=-1, query_processes=None):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForceCuda doesn't support metric %s" % metric)
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[metric]
        self.workers = workers  # not used -> data parallelism not implemented
        self.name = "bruteforcecuda()"
        super().__init__(query_processes)

    def fit(self, index, X):
        #print(f'* index: {index}; position: 1; env var value: {os.environ["CUDA_VISIBLE_DEVICES"]} *')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        import cuml  # to get new value for env var
        self.nbrs = cuml.neighbors.NearestNeighbors(metric=self.metric)
        self.nbrs.fit(X)

    def query(self, v, n):
        return self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=False, n_neighbors=n)[0]

    def query_batch(self, v, n):
        return self.nbrs.kneighbors(v, return_distance=False, n_neighbors=n)

    def query_with_distances(self, v, n):
        (distances, positions) = self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))
