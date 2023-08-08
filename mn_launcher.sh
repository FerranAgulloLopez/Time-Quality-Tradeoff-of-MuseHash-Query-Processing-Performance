#!/bin/bash

# variables initialization
output_directory_path="../output"
number_experiments=1
algorithms=(bruteforce)
datasets=(fake-large)  # all values -> (fake-small fake-medium fake-large)
temporary_launcher_path="./temporary_launcher_file"

# helper function declaration
function run_experiment {

    # create output directory for the specific job
    mkdir -p $output_directory_path/$1
    
    # create temporary launcher file
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=$1"
        echo "#SBATCH --qos=bsc_cs"
        echo "#SBATCH -D ./"
        echo "#SBATCH --ntasks=1"
        echo "#SBATCH --exclusive"
        echo "#SBATCH --output=$output_directory_path/$1/log_%j.out"
        echo "#SBATCH --error=$output_directory_path/$1/log_%j.err"
        echo "#SBATCH --time=00:45:00"
        echo "module purge; module load cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 gcc/8.3.0 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML"
        echo $2
    } > "$temporary_launcher_path"

    # run job
    sbatch "$temporary_launcher_path"

    # remove temporary launcher file
    rm ${temporary_launcher_path}
}

# test data parallelism
list_number_workers=(1 2 4 8)
for algorithm in "${algorithms[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for number_workers in "${list_number_workers[@]}"
        do 
            for (( i=1; i<=$number_experiments; i++ ))
            do
                file_name="data_parallelism_${algorithm}_${dataset}_${number_workers}_count_$i"
                launch_command="python3 run_algorithm.py --dataset ${dataset} --runs 1 --count 10 --algorithm ${algorithm} --module ann_benchmarks.algorithms.${algorithm} --constructor ${algorithm} --batch \"[\\\"euclidean\\\", ${number_workers}, 1]\""

                run_experiment $file_name "$launch_command"
            done
        done
    done
done

# test query parallelism
list_number_workers=(1 2 4 8)
for algorithm in "${algorithms[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for number_workers in "${list_number_workers[@]}"
        do 
            for (( i=1; i<=$number_experiments; i++ ))
            do
                file_name="query_parallelism_${algorithm}_${dataset}_${number_workers}_count_$i"
                launch_command="python3 run_algorithm.py --dataset ${dataset} --runs 1 --count 10 --algorithm ${algorithm} --module ann_benchmarks.algorithms.${algorithm} --constructor ${algorithm} --batch \"[\\\"euclidean\\\", 1, ${number_workers}]\""

                run_experiment $file_name "$launch_command"
            done
        done
    done
done
