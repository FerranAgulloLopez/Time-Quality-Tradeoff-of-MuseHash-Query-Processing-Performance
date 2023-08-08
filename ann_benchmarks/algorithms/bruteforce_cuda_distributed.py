import os

import cuml.neighbors
import cupy as cp
import dask_cudf
import numpy as np
from cudf import DataFrame
from cuml.dask.common.utils import persist_across_workers
from dask.distributed import wait
from dask_cuda import LocalCUDACluster
from distributed import Client

from .base import BaseANN

os.environ["DASK_RMM__POOL_SIZE"] = "500M"
os.environ["DASK_UCX__CUDA_COPY"] = "True"
os.environ["DASK_UCX__TCP"] = "True"
os.environ["DASK_UCX__NVLINK"] = "True"
os.environ["DASK_UCX__INFINIBAND"] = "True"
os.environ["DASK_UCX__NET_DEVICES"] = "ib0"


def distribute_data(client, np_array, n_workers=None, partitions_per_worker=1):
    # Get workers on cluster
    workers = list(client.has_what().keys())
    # Select only n_workers workers
    if n_workers:
        workers = workers[:n_workers]
    # Compute number of partitions
    n_partitions = partitions_per_worker * len(workers)
    # From host to device
    cp_array = cp.array(np_array)
    # From cuPy array to cuDF Dataframe
    cudf_df = DataFrame(cp_array)
    # From cuDF Dataframe to distributed Dask Dataframe
    dask_cudf_df = dask_cudf.from_cudf(cudf_df, npartitions=n_partitions)
    dask_cudf_df, = persist_across_workers(client, [dask_cudf_df], workers=workers)
    wait(dask_cudf_df)
    return dask_cudf_df


class BruteForceCudaDistributed(BaseANN):
    def __init__(self, metric, workers=-1):
        if metric not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError("BruteForceCuda doesn't support metric %s" % metric)
        self._metric = metric
        self.workers = workers
        self.name = "BruteForce()"
        print(f'Number of workers: {self.workers}')
        cluster = LocalCUDACluster(
            protocol="ucx",
            enable_tcp_over_ucx=True,
            enable_nvlink=True,
            enable_infiniband=False
        )
        self.client = Client(cluster)

    def fit(self, X):
        metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[self._metric]
        self._nbrs = cuml.neighbors.NearestNeighbors(metric=metric, client=self.client)
        dist_X = distribute_data(self.client, X, n_workers=2)
        self._nbrs.fit(dist_X)

    def query(self, v, n):
        v = np.expand_dims(v, axis=0)
        dist_v = distribute_data(self.client, v, n_workers=2)
        (distances, positions) = self._nbrs.kneighbors(dist_v, return_distance=False, n_neighbors=n)
        (distances, positions) = self.client.compute([distances, positions])
        return list(distances)

    def query_with_distances(self, v, n):
        v = np.expand_dims(v, axis=0)
        dist_v = distribute_data(self.client, v, n_workers=2)
        (distances, positions) = self._nbrs.kneighbors(np.expand_dims(dist_v, axis=0), return_distance=True, n_neighbors=n)
        (distances, positions) = self.client.compute([distances, positions])
        return zip(list(positions[0]), list(distances[0]))
