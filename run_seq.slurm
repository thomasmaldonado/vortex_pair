#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:62:00         # total run time limit (HH:MM:SS)
#SBATCH --array=0-40
##SBATCH --constraint=gpu80
##SBATCH --partition=mig
#SBATCH --gres=gpu:1 

#echo $SLURM_ARRAY_TASK_ID
mkdir -p data
module purge
module load anaconda3/2023.9
conda activate jax-gpu
bash run.sh
#python plotter.py $SLURM_ARRAY_TASK_ID
