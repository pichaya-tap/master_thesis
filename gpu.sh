#!/bin/bash
#SBATCH --ntasks=14  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 10-00:00:00   # time in d-hh:mm:ss
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --gres=gpu
#SBATCH --mem=100000MB 

hostname
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo
echo "HOSTNAME = $HOSTNAME"
echo "Running on ${SLURM_NPROCS} processor(s) on ${SLURM_NNODES} node(s)."
echo "Running ${SLURM_GPUS_PER_NODE} GPUs."
date
echo
convertsecs() {
    ((h=${1}/3600))
    ((m=(${1}%3600)/60))
    ((s=${1}%60))
    printf "%02dh%02dm%02ds\n" $h $m $s
}

time_start=$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PATH="/home/tappay01/anaconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate gpu

python main_retrain.py
#python test_model.py


echo
echo

time_diff=$(($(date +%s) - time_start))
echo "Total time elapsed: $(convertsecs $time_diff) (on ${NSLOTS} cores)" | tee -a ${log_file}
echo "Done." | tee -a ${log_file}