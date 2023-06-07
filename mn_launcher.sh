#!/bin/sh

# variables initialization
output_directory_path="../output"
number_experiments=5
algorithms=(bruteforce)
number_threads=(8 16 32 64 128 160)
datasets=(fake-small fake-medium fake-large)
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
        echo "#SBATCH --output=$output_directory_path/$1/log_%j.out"
        echo "#SBATCH --error=$output_directory_path/$1/log_%j.err"
        echo "#SBATCH --cpus-per-task=$3"
        echo "#SBATCH --time=00:15:00"
        echo "module purge; module load cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 gcc/8.3.0 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML"
        echo $2
    } > "$temporary_launcher_path"

    # run job
    sbatch "$temporary_launcher_path"

    # remove temporary launcher file
    rm ${temporary_launcher_path}
}

for algorithm in "${algorithms[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for threads in "${number_threads[@]}"
        do
            for (( i=1; i<=$number_experiments; i++ ))
            do
                file_name="dataset_${algorithm}_${dataset}_threads_${threads}_count_$i"
                launch_command="python3 run.py --dataset ${dataset} --local --runs 1 --count 10 --algorithm ${algorithm} --run-disabled --force"

                run_experiment $file_name "$launch_command" ${threads}
            done
        done
    done
done
