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


def __create_qps_chart(path: str, ignore_directories=None):
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

            multiple_labels.append(result_file.parent.name)
            multiple_indexes_results.append(indexes_results)
            multiple_mean_results.append(mean_results)
            multiple_error_margins_results.append(error_margin_results)

    __plot_lines(
        [(
            multiple_mean_results[index],
            multiple_error_margins_results[index],
            multiple_indexes_results[index],
            multiple_labels[index]
        ) for index in range(len(multiple_labels))],
        'queries per second',
        'number of processes',
        multiple_indexes_results[0],
        f'Comparison of the scalability of multiple hash codes for the dataset {path.split("/")[-1]}',
        os.path.join(path, 'comparison_qps.png')
    )


# recursive
def generate_charts(path: str, *splits, ignore_directories=None):
    if len(splits) == 0:
        __create_qps_chart(path, ignore_directories)
    else:
        split = splits[0]
        for folder in split:
            generate_charts(os.path.join(path, folder), *splits[1:], ignore_directories=ignore_directories)


if __name__ == '__main__':
    generate_charts(
        'query_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        ignore_directories={'1', '2', '4', '8', '16'}
    )
    generate_charts(
        'query_parallelism_cuda',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air']
    ),
    generate_charts(
        'data_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air']
    )
