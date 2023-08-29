import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np


def __plot_lines(lines, y_label, x_label, x_ticks, title, path):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, y_error, x, label = line
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_error, y + y_error, alpha=0.5)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel(y_label)
    plt.title(title)
    ax.legend(loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')


def __obtain_info_dimension_importance_chart(path: str, ignore_directories: set, ignore_indexes: set):
    result_files = list(Path(path).glob('*/results.csv'))

    if ignore_directories is not None:
        result_files_to_delete = []
        for result_file in result_files:
            if result_file.parent.name in ignore_directories:
                result_files_to_delete.append(result_file)
        for result_file in result_files_to_delete:
            result_files.remove(result_file)

    multiple_labels = []
    multiple_indexes_results = []
    multiple_mean_results = []
    multiple_error_margins_results = []
    for result_file in result_files:
        with open(result_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            all_csv_lines = list(csv_reader)

            indexes_results = np.asarray(all_csv_lines[0][1:], dtype=int)
            mean_results = np.asarray(all_csv_lines[1][1:], dtype=float)
            error_margin_results = np.asarray(all_csv_lines[2][1:], dtype=float)

            if ignore_indexes is not None:
                indexes_to_delete = []
                for index in range(indexes_results.shape[0]):
                    if indexes_results[index] in ignore_indexes:
                        indexes_to_delete.append(index)
                indexes_to_delete = sorted(indexes_to_delete, reverse=True)
                for index in indexes_to_delete:
                    indexes_results = np.delete(indexes_results, index, axis=0)
                    mean_results = np.delete(mean_results, index, axis=0)
                    error_margin_results = np.delete(error_margin_results, index, axis=0)

            multiple_labels.append(result_file.parent.name)
            multiple_indexes_results.append(indexes_results)
            multiple_mean_results.append(mean_results)
            multiple_error_margins_results.append(error_margin_results)

    multiple_indexes_results = np.asarray(multiple_indexes_results)
    multiple_mean_results = np.asarray(multiple_mean_results)
    multiple_error_margins_results = np.asarray(multiple_error_margins_results)

    multiple_indexes_results = np.transpose(multiple_indexes_results)
    multiple_mean_results = np.transpose(multiple_mean_results)
    multiple_error_margins_results = np.transpose(multiple_error_margins_results)

    multiple_labels = np.asarray(multiple_labels, dtype=int)
    sorted_indexes = multiple_labels.argsort()
    multiple_labels = multiple_labels[sorted_indexes]
    for index in range(multiple_indexes_results.shape[0]):
        multiple_indexes_results[index] = multiple_indexes_results[index][sorted_indexes]
        multiple_mean_results[index] = multiple_mean_results[index][sorted_indexes]
        multiple_error_margins_results[index] = multiple_error_margins_results[index][sorted_indexes]

    return multiple_labels, multiple_indexes_results, multiple_mean_results, multiple_error_margins_results


def create_dimension_importance_chart(path: str, info, labels):
    multiple_labels, multiple_indexes_results, multiple_mean_results, multiple_error_margins_results = info

    __plot_lines(
        [(
            multiple_mean_results[index],
            multiple_error_margins_results[index],
            multiple_labels,
            labels[index]
        ) for index in range(multiple_indexes_results.shape[0])],
        'queries per second',
        'dimension size',
        multiple_labels,
        f'Comparison of the performance of the algorithm with different dimension sizes for the dataset fake-large',
        os.path.join(path, 'comparison_dimension_sizes.png')
    )

    for index in range(multiple_indexes_results.shape[0]):
        mean_results = multiple_mean_results[index]
        error_margin_results = multiple_error_margins_results[index]

        baseline_mean = mean_results[0]
        baseline_error_margin = error_margin_results[0]

        speedup_results = np.divide(mean_results, np.repeat([baseline_mean], mean_results.shape[0], axis=0))
        relative_uncertainty_1 = np.divide(np.repeat([baseline_error_margin], error_margin_results.shape[0], axis=0), np.repeat([baseline_mean], mean_results.shape[0], axis=0))
        relative_uncertainty_2 = np.divide(error_margin_results, mean_results)
        uncertainty_results = np.add(relative_uncertainty_1, relative_uncertainty_2)

        multiple_mean_results[index] = speedup_results
        multiple_error_margins_results[index] = uncertainty_results

    __plot_lines(
        [(
            multiple_mean_results[index],
            multiple_error_margins_results[index],
            multiple_labels,
            labels[index]
        ) for index in range(multiple_indexes_results.shape[0])],
        'speedup',
        'dimension size',
        multiple_labels,
        f'Comparison of the performance of the algorithm with different dimension sizes for the dataset fake-large',
        os.path.join(path, 'comparison_dimension_sizes_speedup.png')
    )


def merge_info(info_1, info_2):
    multiple_labels_1, multiple_indexes_results_1, multiple_mean_results_1, multiple_error_margins_results_1 = info_1
    multiple_labels_2, multiple_indexes_results_2, multiple_mean_results_2, multiple_error_margins_results_2 = info_2

    if not np.array_equal(multiple_labels_1, multiple_labels_2):
        raise Exception('Label arrays not equal')

    multiple_indexes_results = np.concatenate((multiple_indexes_results_1, multiple_indexes_results_2), axis=0)
    multiple_mean_results = np.concatenate((multiple_mean_results_1, multiple_mean_results_2), axis=0)
    multiple_error_margins_results = np.concatenate((multiple_error_margins_results_1, multiple_error_margins_results_2), axis=0)

    return multiple_labels_1, multiple_indexes_results, multiple_mean_results, multiple_error_margins_results


# recursive
def obtain_info(path: str, *splits, ignore_directories=None, ignore_indexes=None):
    if len(splits) == 0:
        return __obtain_info_dimension_importance_chart(path, ignore_directories, ignore_indexes)
    else:
        split = splits[0]
        info = None
        for folder in split:
            aux_info = obtain_info(os.path.join(path, folder), *splits[1:], ignore_directories=ignore_directories, ignore_indexes=ignore_indexes)
            if info is None:
                info = aux_info
            else:
                info = merge_info(info, aux_info)
        return info


if __name__ == '__main__':
    bruteforce_info = obtain_info(
        'query_parallelism',
        ['fake-large'],
        ignore_directories={'128', '512', '2048'},
        ignore_indexes={2, 4, 8, 16, 32}
    )
    ball_tree_info = obtain_info(
        'ball_tree_dimension_importance',
        ['fake-large']
    )
    info = merge_info(bruteforce_info, ball_tree_info)
    create_dimension_importance_chart('', info, ['bruteforce', 'ball_tree'])
