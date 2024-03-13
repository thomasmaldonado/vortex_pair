#!/bin/bash

### KAPPA SWEEP (FOR REMOTE USE) ###

module purge
module load anaconda3/2024.2

# check for conda environment
env_present=$(conda info --envs | grep -c "jax-gpu")
if [ $env_present == 0 ]
then
	conda create --name jax-gpu jax "jaxlib==0.4.23=cuda118*" matplotlib ipykernel -c conda-forge
fi

mkdir -p data

# read number of kappas to be swept from params.py (for now, ignore this, since it's just set to 1)
NK=$(python params.py k)
MIN_K=0
MAX_K=$((NK-1))

# sweep kappas and call slurm script for each kappa
for K_IDX in $(seq $MIN_K $MAX_K)
do	
	mkdir -p data/$K_IDX
	sbatch run.slurm $K_IDX
done