import multiprocessing
import os

import psutil


class BaseANN(object):

    def __init__(self, query_threads=None):
        self.query_threads = query_threads
        if self.query_threads is None:
            self.query_threads = os.cpu_count()

    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        pass

    def query(self, q, n):
        return []  # array of candidate indices

    def batch_query(self, X, n):
        """Provide all queries at once and let algorithm figure out
        how to handle it. Default implementation uses a ThreadPool
        to parallelize query processing."""
        multiprocessing.set_start_method('spawn')
        print(f'Making queries in batch with {self.query_threads} simultaneous threads')
        pool = multiprocessing.pool.ThreadPool(processes=self.query_threads)
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
