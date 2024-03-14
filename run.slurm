#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --time=00:05:00         # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 

#SBATCH --array=0-0 # note that this must read 0-MAX_K with MAX_K given by params.py (MAX_K=NUM_K-1)

K_IDX=$1 # (kappa index, with kappa taken from params.py)
A_IDX=$SLURM_ARRAY_TASK_ID # (a index, with a taken from params.py)
NL=1 # left winding number
NR=1 # right winding number
NU=60 # number of u points (resolution)
NV=60 # number of v points (resolution)
OUTPUT=$K_IDX/$A_IDX # output file

mkdir -p data/$K_IDX

module purge
module load anaconda3/2024.2
conda activate jax-gpu

export PATH=$PATH:/usr/local/cuda-12.3/bin # some mysterious line of code that the della staff told me to include

# solve and plot
python -u jsolver.py $K_IDX $A_IDX $NL $NR $NU $NV $OUTPUT
python plotter.py $OUTPUT
