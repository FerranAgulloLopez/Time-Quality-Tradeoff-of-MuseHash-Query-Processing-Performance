import os


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

    def batch_query(self, X, n):
        raise NotImplementedError()

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name
