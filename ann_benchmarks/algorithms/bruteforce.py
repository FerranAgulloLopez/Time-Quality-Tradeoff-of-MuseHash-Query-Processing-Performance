import numpy as np
from time import time
import numpy
import sklearn.neighbors

from ..distance import metrics as pd
from .base import BaseANN


class BruteForce(BaseANN):
    def __init__(self, metric, workers=-1, query_threads=None):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.workers = workers
        self.name = "BruteForce()"
        print(f'Number of workers: {self.workers}')
        super().__init__(query_threads)

    def fit(self, X):
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[self._metric]
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm="brute", metric=metric, n_jobs=self.workers)
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=False, n_neighbors=n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        # os.environ['OPENBLAS_NUM_THREADS'] = '1'

        if metric not in ("angular", "euclidean", "hamming", "jaccard"):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        elif metric == "hamming" and precision != numpy.bool_:
            raise NotImplementedError(
                "BruteForceBLAS doesn't support precision" " %s with Hamming distances" % precision
            )
        self._metric = metric
        self._precision = precision
        self.name = "BruteForceBLAS()"

        #self.total_time = 0
        #self.time_1 = 0
        #self.time_2 = 0
        #self.time_3 = 0
        #self.time_4 = 0
        #self.time_5 = 0

    def fit(self, X):
        """Initialize the search index."""
        if self._metric == "angular":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == "hamming":
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(numpy.float32)
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=numpy.float32)
            self.lengths = numpy.ascontiguousarray(lens, dtype=numpy.float32)
        elif self._metric == "euclidean":
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == "jaccard":
            self.index = X
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        #init_time = time()
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""

        #init_time_1 = time()
        if self._metric != "jaccard":
            # use same precision for query as for index
            v = numpy.ascontiguousarray(v, dtype=self.index.dtype)
        #self.time_1 += time() - init_time_1

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        #init_time_2 = time()
        if self._metric == "angular":
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -numpy.dot(self.index, v)
        elif self._metric == "euclidean":
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == "hamming":
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == "jaccard":
            dists = [pd[self._metric]["distance"](v, e) for e in self.index]
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        #self.time_2 += time() - init_time_2

        # partition-sort by distance, get `n` closest
        #init_time_3 = time()
        nearest_indices = numpy.argpartition(dists, n)[:n]
        #self.time_3 += time() - init_time_3

        #init_time_4 = time()
        indices = [idx for idx in nearest_indices if pd[self._metric]["distance_valid"](dists[idx])]
        #self.time_4 += time() - init_time_4

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric]["distance"](ep, ev))

        #init_time_5 = time()
        output = map(fix, indices)
        #self.time_5 += time() - init_time_5

        #self.total_time += time() - init_time

        #print(f'total_time: {self.total_time}; time_1: {self.time_1}; time_2: {self.time_2}; time_3: {self.time_3}; time_4: {self.time_4}; time_5: {self.time_5}')
        return output
