import os

import numpy as np


class BaseANN(object):

    def __init__(self, query_processes=None):
        self.query_processes = query_processes
        if self.query_processes is None:
            self.query_processes = os.cpu_count()
        print(f'Number of query threads: {self.query_processes}')

    def done(self):
        pass

    def get_query_processes(self):
        return self.query_processes

    def fit(self, index, X):
        raise NotImplementedError()

    def query(self, q, n):
        raise NotImplementedError()

    def query_batch(self, q, n):
        output = np.zeros((q.shape[0], n), dtype=int)
        for index in range(q.shape[0]):
            output[index] = self.query(q[index], n)
        return output

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
