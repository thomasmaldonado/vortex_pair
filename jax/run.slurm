#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00         # total run time limit (HH:MM:SS)
##SBATCH --array=0-99

#echo $SLURM_ARRAY_TASK_ID
mkdir -p data
module purge
module load anaconda3/2023.9
conda activate jax
python jsolver.py 50 10 1 100 100 50
#python jsolver.py $SLURM_ARRAY_TASK_ID 10 1 40 40 $SLURM_ARRAY_TASK_ID
#python plotter.py $SLURM_ARRAY_TASK_ID
