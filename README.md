# MuseHash Scalability
_This adaptation was created as a result of the research publication ..._

### Introduction
This repository contains an adaptation of the code from [ANN Benchmark](http://github.com/erikbern/ann-benchmarks/). The modifications are meant to perform scalability experiments with the MuseHash method.

The main changes to the code are the following:
- the addition of two new datasets (vgg16-features and muse-hash) for the [AU-AIR](https://bozcani.github.io/auairdataset) and [LSC'23](http://lifelogsearch.org/lsc/) public datasets
- the addition of random synthetic datasets of arbitrary size (fake-*)
- the addition of a new evaluation metric (muse-hash).
- the addition of the comparison between query and data parallelism
- the addition of a GPU algorithm that allows query parallelism (cuml)
- the adaptation of the code to work on high performance computing environments, specifically MareNostrum IV from the Barcelona Supercomputing Center (BSC). The docker behaviour has been removed, as well as many of the original thread pool implementation. 

### How to set up the code
- Install python3.9
- Install requirements.txt
- [opt] If willing to use GPUs, install cuml ([check official documentation](https://docs.rapids.ai/install#selector))

### How to set up the data
If using the vgg16-features or muse-hash datasets, it will be needed to populate the *data* directory in the following manner (please contact the creators of this repository to access these two datasets):
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

If only using the synthetic datasets there is no need for any data preparation. For the other datasets, check the documentation of the [original implementation]((http://github.com/erikbern/ann-benchmarks/)).

### How to create the datasets
For creating the datasets with vgg16 features:
```
python3 create_dataset.py --dataset vgg16-features-ORIGINAL_DATASET
```
remember to populate the variable ORIGINAL_DATASET (possible values: *au_air* and *lcs*)

For creating the datasets with hash codes:
```
python3 create_dataset.py --dataset muse-hash-MODALITIES-HASH_CODE_SIZE-euclidean-ORIGINAL_DATASET
```
remember to populate the variables:
- MODALITIES (possible values: *visual*, *visual-temporal* and *visual-temporal-spatial*)
- HASH_CODE_SIZE (possible values: *16*, *32*, *64*, *128*, *256*, *512*, *1024* and *2048*)
- ORIGINAL_DATASET (possible values: *au_air* and *lcs*)

For creating the synthetic datasets:
```
python3 create_dataset.py --dataset fake-SAMPLE_SIZE-HASH_CODE_SIZE
```
remember to populate the variables:
- SAMPLE_SIZE (possible values: *small* {28000}, *medium* {112000} and *large* {448000})
- HASH_CODE_SIZE (possible values: *32*, *128*, *512* and *2048*)

Other combinations of values for the three datasets can be included modifying the dictionary available at the end of the file *ann_benchmarks/datasets.py*.

### How to run
For running:
```
python3 run_algorithm.py --dataset DATASET_NAME --runs RUNS --count NEIGHBOURS --algorithm ALGORITHM_NAME --module ann_benchmarks.algorithms.ALGORITHM_NAME --constructor ALGORITHM_NAME "[\"METRIC\", WORKERS_DATA_PARALLELISM, WORKERS_QUERY_PARALLELISM]" --batch
```
remember to populate the variables DATASET_NAME, ALGORITHM_NAME, NEIGHBOURS, RUNS, METRIC, WORKERS_QUERY_PARALLELISM and WORKERS_DATA_PARALLELISM, like the following:
```
python3 run_algorithm.py --dataset fake-small-32 --runs 1 --count 10 --algorithm bruteforce --module ann_benchmarks.algorithms.bruteforce --constructor bruteforce "[\"euclidean\", 1, 1]" --batch
```

For running all the scalability experiments needed for the paper, check the file `definitive_results/mn_launcher.sh`. It contains the run commands to launch all experiments in one of the MareNostrum clusters. The last can be updated to work in other platforms. Check the directory *definitive_results* for more information.

### How to plot
For plotting:
```
python3 plot.py --dataset DATASET_NAME --count 10 --recompute -x METRIC
```
remember to populate the variables DATASET_NAME and METRIC, like the following:
```
python3 plot.py --dataset vgg16-features --count 10 --recompute -x muse-hash-recall
```

Take into account that this feature is not used in the scalability experiments. It is maintained for backward compatibility with the original implementation and for assuring the good functioning of the code with the computation of the precision and recall metrics. Check the *definitive_results* directory for the information of the plots / charts produced for the scalability experiments.