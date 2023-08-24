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

The folder in the last level contains two files: *log_ID.err* and *log_ID.out* (the ID corresponds to an internal identifier of the POWER-CTE cluster). The first file contains the standard error stream, whereas the second file contains the standard output stream. 

The raw metric employed in all the scalability experiments is the number of queries per second. This metric is extracted from the files *log_ID.out* with the python script named as *extracted_results.py*. The script creates, for every group of 5 experiments with same characteristics, a csv file with the queries per second of all of them (the resulting file is named as *extracted_results.csv*). As well, this python script consolidates the extracted results for every hash code size computing the average result and margin of error. The consolidated results are saved in a new file named as *results.csv*. All of these files can be found in the same structure of folders previously represented. 

#### Available types of experiments
The definitive results contain the following types of experiments:
- *data_parallelism*: checking the scalability of the bruteforce sklearn algorithm when adding more processes for doing the single query computation (via the variable named as *n_jobs*)
- *query_parallelism*: checking the scalability of the bruteforce sklearn algorithm when adding more processes to handle the full set of input queries
- *query_parallelism_cuda*: checking the scalability of the bruteforce cuml algorithm when adding more processes to handle the full set of input queries (using GPUs)

#### Available charts
The definitive results contain the following list of charts:
- *comparison_qps*: these charts are created for all datasets and types of experiments. They show the scalability of the dataset when using multiple parallel processes and hash code sizes. The axis *y* represents the queries per second of each sample. These charts are created with the python script named as *generate_qps_charts.py*.
- *comparison_speedup*: these charts are created for all datasets and types of experiments. They show the scalability of the dataset when using multiple parallel processes and hash code sizes. The axis *y* represents the speedup of each sample with regard to the sample that uses only one single process (baseline). Notice that in the case of *query_parallelism_cuda*, the baseline is taken from the *query_parallelism* experiments. These charts are created with the python script named as *generate_speedup_charts.py*.