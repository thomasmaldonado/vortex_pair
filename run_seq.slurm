#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --time=6:00:00         # total run time limit (HH:MM:SS)
##SBATCH --constraint=gpu80
##SBATCH --partition=mig
##SBATCH --gres=gpu:1 
NK=$(python params.py k)
MIN_K=0
MAX_K=$((NK-1))
#SBATCH --array=$MIN_K-$MAX_K
K_IDX=$SLURM_ARRAY_TASK_ID
echo $K_IDX

mkdir -p data
module purge
module load anaconda3/2024.2
conda activate jax

#export PATH=$PATH:/usr/local/cuda-12.3/bin

bash run.sh $K_IDX
#python plotter.py $SLURM_ARRAY_TASK_ID
