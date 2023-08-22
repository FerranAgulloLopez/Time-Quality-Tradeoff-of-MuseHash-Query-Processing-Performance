import numpy as np
import sklearn.neighbors

from .base import BaseANN

# solve sklearn joblib issue with nested parallelism
def _patch_joblib_loky_backend():
    import joblib._parallel_backends
    from joblib._parallel_backends import mp, cpu_count

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel"""
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        elif mp is None or n_jobs is None:
            # multiprocessing is not available or disabled, fallback
            # to sequential mode
            return 1
        elif n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        return n_jobs

    # Monkey-patch to allow daemonic thread to spawn processes
    joblib._parallel_backends.LokyBackend.effective_n_jobs = effective_n_jobs


class bruteforce(BaseANN):
    def __init__(self, metric, workers=-1, query_processes=None):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForce doesn't support metric %s" % metric)
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[metric]
        self.workers = workers
        self.name = "bruteforce()"
        print(f'Number of workers: {self.workers}')
        _patch_joblib_loky_backend()
        super().__init__(query_processes)

    def fit(self, index, X):
        self.nbrs = sklearn.neighbors.NearestNeighbors(algorithm="brute", metric=self.metric, n_jobs=self.workers)
        self.nbrs.fit(X)

    def query(self, v, n):
        return self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=False, n_neighbors=n)[0]

    def query_batch(self, v, n):
        return self.nbrs.kneighbors(v, return_distance=False, n_neighbors=n)

    def query_with_distances(self, v, n):
        (distances, positions) = self.nbrs.kneighbors(np.expand_dims(v, axis=0), return_distance=True, n_neighbors=n)
        return zip(list(positions[0]), list(distances[0]))
