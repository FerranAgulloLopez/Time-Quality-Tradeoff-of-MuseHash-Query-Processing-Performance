from __future__ import absolute_import
import os

import numpy as np


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def epsilon_threshold(data, count, epsilon):
    return data[count - 1] * (1 + epsilon)


def get_recall_values(dataset_distances, run_distances, count, threshold, epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = threshold(dataset_distances[i], count, epsilon)
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual
    return (np.mean(recalls) / float(count), np.std(recalls) / float(count), recalls)


def knn(dataset_distances, run_distances, count, metrics, epsilon=1e-3):
    if "knn" not in metrics:
        print("Computing knn metrics")
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, knn_threshold, epsilon)
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached result")
    return metrics["knn"]


def muse_hash(metrics, neighbors):
    DATA_INFO_PATH = './data/dataset/'
    LABELS_LENGTH = 8

    def compute_precision_recall(query_label, result_labels):
        # Convert query and result labels to numpy arrays
        query_label = np.array(query_label)
        result_labels = np.array(result_labels)

        # Compute true positives, false positives, and false negatives
        true_positives = np.sum(query_label * result_labels)
        false_positives = np.sum((1 - query_label) * result_labels)
        false_negatives = np.sum(query_label * (1 - result_labels))

        # Compute precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return precision, recall

    if "muse_hash" not in metrics:
        print("Computing museHash metrics")
        muse_hash_metrics = metrics.create_group("muse_hash")

        # load test split info
        test_split = np.genfromtxt(os.path.join(DATA_INFO_PATH, 'test.txt'), dtype=int)

        # load retrieval link info
        retrieval_link = np.genfromtxt(os.path.join(DATA_INFO_PATH, 'retrieval.txt'), dtype=int)

        # compute precision and recall for each test sample / query
        precisions = np.zeros(neighbors.shape[0])
        recalls = np.zeros(neighbors.shape[0])
        for index in range(neighbors.shape[0]):
            # match label to query
            # query_label = np.genfromtxt(os.path.join(DATA_INFO_PATH, 'labels', f'label_{test_split[index]}.txt'))
            query_label = np.genfromtxt(os.path.join(DATA_INFO_PATH, 'labels', f'label_{test_split[index]}.txt'))

            # match label to neighbours
            result_labels = np.zeros((neighbors.shape[1], LABELS_LENGTH))
            for index_2 in range(neighbors.shape[1]):
                link = retrieval_link[neighbors[index, index_2]]
                result_labels[index_2] = np.genfromtxt(os.path.join(DATA_INFO_PATH, 'labels', f'label_{link}.txt'))

            # compute precision and recall
            precision, recall = compute_precision_recall(query_label, result_labels)

            # store results
            precisions[index] = precision
            recalls[index] = recall

        muse_hash_metrics.attrs["mean_precisions"] = np.mean(precisions)
        muse_hash_metrics.attrs["std_precisions"] = np.std(precisions)
        muse_hash_metrics.attrs["mean_recalls"] = np.mean(recalls)
        muse_hash_metrics.attrs["std_recalls"] = np.std(recalls)
        muse_hash_metrics["precisions"] = precisions
        muse_hash_metrics["recalls"] = recalls

    else:
        print("Found cached result")
    return metrics["muse_hash"]


def epsilon(dataset_distances, run_distances, count, metrics, epsilon=0.01):
    s = "eps" + str(epsilon)
    if s not in metrics:
        print("Computing epsilon metrics")
        epsilon_metrics = metrics.create_group(s)
        mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, epsilon_threshold, epsilon)
        epsilon_metrics.attrs["mean"] = mean
        epsilon_metrics.attrs["std"] = std
        epsilon_metrics["recalls"] = recalls
    else:
        print("Found cached result")
    return metrics[s]


def rel(dataset_distances, run_distances, metrics):
    if "rel" not in metrics.attrs:
        print("Computing rel metrics")
        total_closest_distance = 0.0
        total_candidate_distance = 0.0
        for true_distances, found_distances in zip(dataset_distances, run_distances):
            total_closest_distance += np.sum(true_distances)
            total_candidate_distance += np.sum(found_distances)
        if total_closest_distance < 0.01:
            metrics.attrs["rel"] = float("inf")
        else:
            metrics.attrs["rel"] = total_candidate_distance / total_closest_distance
    else:
        print("Found cached result")
    return metrics.attrs["rel"]


def queries_per_second(queries, attrs):
    return 1.0 / attrs["best_search_time"]


def percentile_50(times):
    return np.percentile(times, 50.0) * 1000.0


def percentile_95(times):
    return np.percentile(times, 95.0) * 1000.0


def percentile_99(times):
    return np.percentile(times, 99.0) * 1000.0


def percentile_999(times):
    return np.percentile(times, 99.9) * 1000.0


def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)


def build_time(queries, attrs):
    return attrs["build_time"]


def candidates(queries, attrs):
    return attrs["candidates"]


def dist_computations(queries, attrs):
    return attrs.get("dist_comps", 0) / (attrs["run_count"] * len(queries))


all_metrics = {
    "k-nn": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: knn(
            true_distances, run_distances, run_attrs["count"], metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "muse-hash-precision": {
        "description": "Precision",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: muse_hash(
            metrics, neighbors
        ).attrs[
            "mean_precisions"
        ],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "muse-hash-recall": {
        "description": "Recall",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: muse_hash(
            metrics, neighbors
        ).attrs[
            "mean_recalls"
        ],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "epsilon": {
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: epsilon(
            true_distances, run_distances, run_attrs["count"], metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    "largeepsilon": {
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: epsilon(
            true_distances, run_distances, run_attrs["count"], metrics, 0.1
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    "rel": {
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: rel(
            true_distances, run_distances, metrics
        ),  # noqa
        "worst": float("inf"),
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: queries_per_second(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("-inf"),
    },
    "p50": {
        "description": "Percentile 50 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: percentile_50(times),  # noqa
        "worst": float("inf"),
    },
    "p95": {
        "description": "Percentile 95 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: percentile_95(times),  # noqa
        "worst": float("inf"),
    },
    "p99": {
        "description": "Percentile 99 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: percentile_99(times),  # noqa
        "worst": float("inf"),
    },
    "p999": {
        "description": "Percentile 99.9 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: percentile_999(times),  # noqa
        "worst": float("inf"),
    },
    "distcomps": {
        "description": "Distance computations",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: dist_computations(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    "build": {
        "description": "Build time (s)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: build_time(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    "candidates": {
        "description": "Candidates generated",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: candidates(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    "indexsize": {
        "description": "Index size (kB)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: index_size(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("inf"),
    },
    "queriessize": {
        "description": "Index size (kB)/Queries per second (s)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs, neighbors: index_size(
            true_distances, run_attrs
        )
        / queries_per_second(true_distances, run_attrs),  # noqa
        "worst": float("inf"),
    },
}
