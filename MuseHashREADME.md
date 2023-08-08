# MuseHash Comparison
_This adaptation was created as a results of the publication ..._

### Introduction
This repository contains an adaptation of the code from [ANN Benchmark](http://github.com/erikbern/ann-benchmarks/). The modifications are small and are meant to compare the method MuseHash with the baseline.

The main change consists in the addition of two new datasets (vgg16-features and muse-hash) and a new evaluation metric (muse-hash).

### How to set up
- Install python3.9
- Install requirements.txt
- Install cuml ([check official documentation](https://docs.rapids.ai/install#selector)) -> pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com 
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
python3 run.py --dataset DATASET_NAME --local --runs 1 --algorithm ALGORITHM_NAME --count NEIGHBOURS --run-disabled --force
```
**remember to populate the variables DATASET_NAME, ALGORITHM_NAME and NEIGHBOURS**, like the following:
```
python3 run.py --dataset vgg16-features-au_air --local --runs 1 --algorithm pynndescent --count 10 --run-disabled --force
```

### How to plot
For plotting:
```
python3 plot.py --dataset DATASET_NAME --count 10 --recompute -x METRIC
```
**remember to populate the variables DATASET_NAME and METRIC**, like the following:
```
python3 plot.py --dataset vgg16-features --count 10 --recompute -x muse-hash-recall
```