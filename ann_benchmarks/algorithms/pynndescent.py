import numpy as np
from pynndescent import NNDescent
import scipy.sparse

from .base import BaseANN


class pynndescent(BaseANN):
    def __init__(
            self,
            metric,
            workers=-1,
            query_processes=None,
            n_neighbors=10,
            pruning_degree_multiplier=1.5,
            diversify_prob=1.0,
            leaf_size=20,
            n_search_trees=1,
            epsilon=0.1
    ):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = {
            'angular': 'dot',
            'euclidean': 'euclidean',
            'hamming': 'hamming',
            'jaccard': 'jaccard'
        }[metric]
        self.name = "pynndescent()"

        self.n_neighbors = n_neighbors
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.diversify_prob = diversify_prob
        self.leaf_size = leaf_size
        self.epsilon = epsilon
        self.n_search_trees = n_search_trees

        if workers != 1:
            raise Exception('PyNNDescent algorirthm does not implement data parallelism')
        super().__init__(query_processes)

    def fit(self, index, X):
        if self.metric == "jaccard":
            # Convert to sparse matrix format
            if type(X) == list:
                sizes = [len(x) for x in X]
                n_cols = max([max(x) for x in X]) + 1
                matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
                matrix.indices = np.hstack(X).astype(np.int32)
                matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
                matrix.data = np.ones(matrix.indices.shape[0], dtype=np.float32)
                matrix.sort_indices()
                X = matrix
            else:
                X = scipy.sparse.csr_matrix(X)

            self._query_matrix = scipy.sparse.csr_matrix((1, X.shape[1]), dtype=np.float32)

        self._index = NNDescent(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            low_memory=True,
            leaf_size=self.leaf_size,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            diversify_prob=self.diversify_prob,
            n_search_trees=self.n_search_trees,
            compressed=True,
            verbose=True
        )
        if hasattr(self._index, "prepare"):
            self._index.prepare()
        else:
            self._index._init_search_graph()
            if self._index._is_sparse:
                if hasattr(self._index, "_init_sparse_search_function"):
                    self._index._init_sparse_search_function()
            else:
                if hasattr(self._index, "_init_search_function"):
                    self._index._init_search_function()

    def query(self, v, n):
        if self._index._is_sparse:
            # convert index array to sparse matrix format and query;
            # the overhead of direct conversion is high for single
            # queries (converting the entire test dataset and sending
            # single rows is better), so we just populate the required
            # structures.
            if v.dtype == np.bool_:
                self._query_matrix.indices = np.flatnonzero(v).astype(np.int32)
            else:
                self._query_matrix.indices = v.astype(np.int32)
            size = self._query_matrix.indices.shape[0]
            self._query_matrix.indptr = np.array([0, size], dtype=np.int32)
            self._query_matrix.data = np.ones(size, dtype=np.float32)
            ind, dist = self._index.query(self._query_matrix, k=n, epsilon=self.epsilon)
        else:
            ind, dist = self._index.query(v.reshape(1, -1).astype("float32"), k=n, epsilon=self.epsilon)
        return ind[0]

    def query_batch(self, v, n):
        raise NotImplementedError()
