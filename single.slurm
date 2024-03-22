#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:60:00         # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.2
conda activate jax-gpu
export PATH=$PATH:/usr/local/cuda-12.3/bin # some mysterious line of code that the della staff told me to include

K_IDX=$SLURM_ARRAY_TASK_ID # (kappa index, with kappa taken from params.py)
mkdir -p data/$K_IDX

# read single separation from input
A_IDX=$1
OUTPUT=$2
INPUT=$3

python -u jsolver.py $K_IDX $A_IDX $OUTPUT $INPUT
python plotter.py $OUTPUT