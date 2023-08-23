# MuseHash Comparison
_This adaptation was created as a results of the publication ..._

### Introduction
This repository contains an adaptation of the code from [ANN Benchmark](http://github.com/erikbern/ann-benchmarks/). The modifications are to perform scalability experiments with the MuseHash method.

The main changes are the following:
- the addition of two new datasets (vgg16-features and muse-hash)
- the addition of a new evaluation metric (muse-hash).
- the addition of the comparison between query and data parallelism
- the addition of a GPU algorithm that allows query parallelism
- the adaptation of the code to work on high performance computing environments, specifically MareNostrum IV from the Barcelona Supercomputing Center.

### How to set up
- Install python3.9
- Install requirements.txt
- Install cuml ([check official documentation](https://docs.rapids.ai/install#selector)) -> pip3 install cuml-cu11 --extra-index-url=https://pypi.nvidia.com 
- Create a data directory and populate it like the following:
```
- ./data/
  - au_air_dataset/
    - retrieval.txt
    - test.txt
    - train.txt
    - labels/
        - label_1.txt
        - label_2.txt
        - ...
    - hash_codes/
      - spatial/
        - 16bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - 32bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - ...
      - temporal/
        - 16bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - 32bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - ...
      - visual/
        - 16bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - 32bit/
          - bin_feature_1.txt
          - bin_feature_2.txt
          - ...
        - ...
    - vgg16_features/
      - feature_1.txt
      - feature_2.txt
      - ...
  - lcs_dataset/
    - ...
```

### How to create the datasets
For creating the dataset with vgg16 features:
```
python3 create_dataset.py --dataset vgg16-features-[...]
```

For creating the dataset with hash codes:
```
python3 create_dataset.py --dataset muse-hash-[...]
```

### How to run
For running:
```
python3 run_algorithm.py --dataset DATASET_NAME --runs RUNS --count NEIGHBOURS --algorithm ALGORITHM_NAME --module ann_benchmarks.algorithms.ALGORITHM_NAME --constructor ALGORITHM_NAME "[\"METRIC\", WORKERS_DATA_PARALLELISM, WORKERS_QUERY_PARALLELISM]" --batch
```
**remember to populate the variables DATASET_NAME, ALGORITHM_NAME, NEIGHBOURS, RUNS, METRIC, WORKERS_QUERY_PARALLELISM and WORKERS_DATA_PARALLELISM**, like the following:
```
python3 run_algorithm.py --dataset fake-small-32 --runs 1 --count 10 --algorithm bruteforce --module ann_benchmarks.algorithms.bruteforce --constructor bruteforce "[\"euclidean\", 1, 1]" --batch
```

For running all the experiments needed for the paper, check the file `mn_launcher.sh`. It contains the launch commands to launch all experiments in one of the MareNostrum clusters. The last can be updated to work in other platforms.

### How to plot
For plotting:
```
python3 plot.py --dataset DATASET_NAME --count 10 --recompute -x METRIC
```
**remember to populate the variables DATASET_NAME and METRIC**, like the following:
```
python3 plot.py --dataset vgg16-features --count 10 --recompute -x muse-hash-recall
```