### Definitive results
This directory contain the definitive results to support the research contribution in the scalability of the MuseHash method.

All experiments have been performed in the POWER-CTE cluster of MareNostrum VI (Barcelona Supercomputing Center). The hardware specification can be found in this public [guide](https://www.bsc.es/user-support/power.php). The results were run with the bash script named as *mn_launcher.sh*.

#### Folder structure
The folder structure of the experiments follows the next pattern:
- type of experiment
  - dataset name
    - hash code size
      - number of parallel processes
        - 5 distinct experiments with the same parameters (to extract error margins)

#### Available types of experiments
The definitive results contain the following types of experiments:
- *data_parallelism*: checking the scalability of the bruteforce sklearn algorithm when adding more processes for doing the single query computation (via the variable named as *n_jobs*)
- *query_parallelism*: checking the scalability of the bruteforce sklearn algorithm when adding more processes to handle the full set of input queries
- *query_parallelism_cuda*: checking the scalability of the bruteforce cuml algorithm when adding more processes to handle the full set of input queries (using GPUs)

#### Available charts
The definitive results contain the following list of charts:
- *comparison_qps*: these charts are created for all datasets and types of experiments. They show the scalability of the dataset when using multiple parallel processes and hash code sizes. The axis *y* represents the queries per second of each sample. These charts are created with the python script named as *generate_qps_charts*.
- *comparison_speedup*: these charts are created for all datasets and types of experiments. They show the scalability of the dataset when using multiple parallel processes and hash code sizes. The axis *y* represents the speedup of each sample with regard to the sample that uses only one single process (baseline). Notice that in the case of *query_parallelism_cuda*, the baseline is taken from the *query_parallelism* experiments. These charts are created with the python script named as *generate_qps_charts*.