#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00         # total run time limit (HH:MM:SS)
#SBATCH --array=0-39
#SBATCH --gres=gpu:1 

#echo $SLURM_ARRAY_TASK_ID

K_IDX=$1
A_IDX = $SLURM_ARRAY_TASK_ID
NL=1
NR=1
NU=40
NV=40
OUTPUT=$K_IDX/$A_IDX

mkdir -p data/$K_IDX

module purge
module load anaconda3/2023.9
conda activate jax

python -u jsolver.py $K_IDX $A_IDX $NL $NR $NU $NV $OUTPUT

#python plotter.py $SLURM_ARRAY_TASK_ID
