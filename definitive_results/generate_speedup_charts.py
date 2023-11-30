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


def __create_speedup_chart(path: str, baseline=None, ignore_directories=None):
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
    multiple_speedup_results = []
    multiple_uncertainty_results = []
    for result_file in result_files:
        with open(result_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            all_csv_lines = list(csv_reader)

            indexes_results = np.asarray(all_csv_lines[0][1:], dtype=int)
            mean_results = np.asarray(all_csv_lines[1][1:], dtype=float)
            error_margin_results = np.asarray(all_csv_lines[2][1:], dtype=float)

            if baseline is None:
                baseline_mean = mean_results[0]
                baseline_error_margin = error_margin_results[0]
            else:
                baseline_mean, baseline_error_margin = baseline[result_file.parent.name]

            speedup_results = np.divide(mean_results, np.repeat([baseline_mean], mean_results.shape[0], axis=0))
            relative_uncertainty_1 = np.divide(np.repeat([baseline_error_margin], error_margin_results.shape[0], axis=0), np.repeat([baseline_mean], mean_results.shape[0], axis=0))
            relative_uncertainty_2 = np.divide(error_margin_results, mean_results)
            uncertainty_results = np.add(relative_uncertainty_1, relative_uncertainty_2)

            multiple_labels.append(result_file.parent.name)
            multiple_indexes_results.append(indexes_results)
            multiple_speedup_results.append(speedup_results)
            multiple_uncertainty_results.append(uncertainty_results)

    __plot_lines(
        [(
            multiple_speedup_results[index],
            multiple_uncertainty_results[index],
            multiple_indexes_results[index],
            multiple_labels[index]
        ) for index in range(len(multiple_labels))] +
        [(
            multiple_indexes_results[0],
            np.zeros(multiple_indexes_results[0].shape[0]),
            multiple_indexes_results[0],
            'optimal scalability'
        )],
        'speedup',
        'number of processes',
        multiple_indexes_results[0],
        f'Comparison of the scalability of multiple hash codes for the dataset {path.split("/")[-1]}',
        os.path.join(path, 'comparison_speedup.png')
    )


# recursive
def generate_charts(path: str, *splits, baseline=None, ignore_directories=None):
    if len(splits) == 0:
        __create_speedup_chart(path, baseline, ignore_directories)
    else:
        split = splits[0]
        for folder in split:
            generate_charts(os.path.join(path, folder), *splits[1:], baseline=None if baseline is None else baseline[folder], ignore_directories=ignore_directories)


if __name__ == '__main__':
    generate_charts(
        'query_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        ignore_directories={'1', '2', '4', '8', '16'}
    )
    generate_charts(
        'query_parallelism_cuda',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        baseline={
            'fake-large': {
                '32': (5.839611474993033, 0.05301108559528854),
                '128': (1.873244693686018, 0.010735223090178669),
                '512': (0.4971154043701028, 0.0064679272563895),
                '2048': (0.12553415261935685, 0.0027742345216329894)
            },
            'fake-medium': {
                '32': (24.215453191805313, 0.3997191680019935),
                '128': (7.190787563457468, 0.11924760725946783),
                '512': (1.9397325231515237, 0.054127487673176176),
                '2048': (0.5074463370006881, 0.011553609530322667)
            },
            'fake-small': {
                '32': (99.64858497288404, 0.32702373063754414),
                '128': (29.846570636519107, 0.304613545742876),
                '512': (7.76824087033913, 0.15017123043339883),
                '2048': (1.995862837306532, 0.043873197397447)
            },
            'muse-hash-visual-temporal-spatial-euclidean-au_air': {
                '32': (344.1591326777886, 0.33284600289534816),
                '128': (119.84918995808007, 0.21589480099177497),
                '512': (32.08709432305078, 0.2785519010746276),
                '2048': (8.100334426542304, 0.09070412806227318)
            }
        }
    ),
    generate_charts(
        'data_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air']
    ),
    generate_charts(
        'query_parallelism_pynndescent',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air']
    )
