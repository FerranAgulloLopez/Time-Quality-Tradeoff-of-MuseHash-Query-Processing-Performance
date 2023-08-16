import argparse
import json
import time
import multiprocessing
import multiprocessing.pool
from multiprocessing import freeze_support

import numpy as np

from ann_benchmarks.algorithms.bruteforce import bruteforce

import numpy

from .algorithms.definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset
from .distance import dataset_transform, metrics
from .results import store_results

multiprocessing.set_start_method('spawn', force=True)


def initialize_par(algo_init, n_init):
    global algo, n
    algo = algo_init
    n = n_init
    time.sleep(5)


def inference_par(v):
    global algo, n
    return algo.query(v, n)


def inference_batch_par(v):
    global algo, n
    return algo.query_batch(v, n)


def run_individual_query(algo, X_train, X_test, distance, count, run_count, batch, pool):
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(X):
            if prepared_queries:
                raise NotImplementedError()
            else:

                # INFERENCE START

                query_processes = algo.get_query_processes()
                pool = multiprocessing.pool.Pool(processes=query_processes)
                pool.starmap(initialize_par, [(algo, count)] * query_processes)

                print('starts inference')
                start = time.time()

                results = pool.map(inference_par, [X[index] for index in range(X.shape[0])])

                # INFERENCE END

                total = time.time() - start

            candidates = [
                [(int(idx), float(metrics[distance]["distance"](v, X_train[idx]))) for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(total / float(len(X)), v) for v in candidates]

        def batch_query(X):
            if prepared_queries:
                raise NotImplementedError()
            else:

                # INFERENCE START

                query_processes = algo.get_query_processes()
                pool = multiprocessing.pool.Pool(processes=query_processes)
                pool.starmap(initialize_par, [(algo, count)] * query_processes)

                print('starts inference')
                start = time.time()

                results = pool.map(inference_batch_par, [split for split in np.array_split(X, query_processes)])
                results = np.concatenate(results, axis=0)

                # INFERENCE END

                total = time.time() - start

            candidates = [
                [(int(idx), float(metrics[distance]["distance"](v, X_train[idx]))) for idx in single_results]  # noqa
                for v, single_results in zip(X, results)
            ]
            return [(total / float(len(X)), v) for v in candidates]

        if batch:
            results = batch_query(X_test)
        else:
            results = single_query(X_test)

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def run(definition, dataset, count, run_count, batch):
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), """\
error: query argument groups have been specified for %s.%s(%s), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function""" % (
        definition.module,
        definition.constructor,
        definition.arguments,
    )

    D, dimension = get_dataset(dataset)
    X_train = numpy.array(D["train"])
    X_test = numpy.array(D["test"])
    distance = D.attrs["distance"]
    print("got a train set of size (%d * %d)" % (X_train.shape[0], dimension))
    print("got %d queries" % len(X_test))

    X_train, X_test = dataset_transform(D)

    pool = None
    try:
        if hasattr(algo, "supports_prepared_queries"):
            algo.supports_prepared_queries()

        t0 = time.time()

        # FIT START

        # query_processes = algo.get_query_processes()
        # pool = multiprocessing.pool.Pool(processes=query_processes)
        # pool.starmap(fit_par, [(definition, index, X_train) for index in range(query_processes)])
        algo.fit(0, X_train)

        # FIT END

        build_time = time.time() - t0
        print("Built index in", build_time)

        query_argument_groups = definition.query_argument_groups
        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [[]]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." % (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch, pool)
            print(f"Queries per second: {1.0 / descriptor['best_search_time']}")
            descriptor["build_time"] = build_time
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            store_results(dataset, count, definition, query_arguments, descriptor, results, batch)
    finally:
        algo.done()
        if pool is not None:
            pool.close()


def run_from_cmdline():
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    parser.add_argument(
        "--count", help="K: Number of nearest neighbours for the algorithm to return.", required=True, type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()
    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)
