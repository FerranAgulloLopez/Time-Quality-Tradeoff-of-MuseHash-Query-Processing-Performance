import csv
import math
import os
import re
from pathlib import Path

import numpy as np


def __read_results(path: str):
    output_files = list(Path(path).rglob('*.out'))
    if len(output_files) != 5:
        print(f'WARNING. Output files in {path} are {len(output_files)}')

    qps_list = []
    for output_file in output_files:
        with open(output_file, mode='r') as file:
            file_data = file.read()
        qps = re.findall('Queries per second: (.*)\n', file_data)
        if len(qps) != 1:
            raise Exception(f'Unable to correctly extract results from output file {output_file}')
        qps = float(qps[0])
        qps_list.append(qps)

    with open(os.path.join(path, 'extracted_results.csv'), 'w') as f:
        write = csv.writer(f, delimiter=',')
        write.writerows([
            list(range(1, len(qps_list) + 1)),
            qps_list
        ])


def __compress_results(path: str):
    result_files = list(Path(path).rglob('*extracted_results.csv'))

    final_results_mean = []
    final_results_error_margin = []
    final_results_labels = []
    for result_file in result_files:
        with open(result_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            results = np.asarray(list(csv_reader)[1], dtype=float)

            z = 1.96  # 95 %
            std = results.std()
            error_margin = z * std / math.sqrt(results.shape[0])

            final_results_mean.append(results.mean())
            final_results_error_margin.append(error_margin)
            final_results_labels.append(int(result_file.parent.name))

    zipped = list(zip(final_results_labels, final_results_mean, final_results_error_margin))
    zipped.sort()
    final_results_labels, final_results_mean, final_results_error_margin = zip(*zipped)

    with open(os.path.join(path, 'results.csv'), 'w') as f:
        write = csv.writer(f, delimiter=',')
        write.writerows([
            [''] + list(final_results_labels),
            ['mean'] + list(final_results_mean),
            ['error_margin'] + list(final_results_error_margin)
        ])


# recursive
def extract_results(path: str, *splits):
    if len(splits) == 0:
        __read_results(path)
    else:
        split = splits[0]
        for folder in split:
            extract_results(os.path.join(path, folder), *splits[1:])
        if len(splits) == 1:
            __compress_results(path)


if __name__ == '__main__':
    extract_results(
        'query_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        ['32', '128', '512', '2048'],
        ['1', '2', '4', '8', '16', '32']
    )
    extract_results(
        'query_parallelism_cuda',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        ['32', '128', '512', '2048'],
        ['1', '2', '4']
    ),
    extract_results(
        'data_parallelism',
        ['fake-large', 'fake-medium', 'fake-small', 'muse-hash-visual-temporal-spatial-euclidean-au_air'],
        ['32', '128', '512', '2048'],
        ['1', '2', '4', '8', '16', '32']
    )
