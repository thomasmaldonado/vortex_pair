#!/bin/bash

module purge
module load anaconda3/2024.2

# check for conda environment
env_present=$(conda info --envs | grep -c "jax-gpu")
if [ $env_present == 0 ]
then
	conda create --name jax-gpu jax "jaxlib==0.4.23=cuda118*" sympy matplotlib -c conda-forge
fi

# call slurm script for a range of kappas specified in params.py
conda activate jax-gpu
NK=$(python params.py k)
MIN_K=0
MAX_K=$((NK-1))
sbatch -a $MIN_K-$MAX_K run.slurm