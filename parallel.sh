#!/bin/bash

### SEQUENTIAL SEPARATION SWEEP (FOR LOCAL USE) ###

mkdir -p data

module purge
module load anaconda3/2024.2
conda activate jax

NK=$(python params.py k)
MIN_K=0
MAX_K=$((NK-1))

for K_IDX in $(seq $MIN_K $MAX_K)
do	
	mkdir -p data/$K_IDX
	sbatch parallel.slurm $K_IDX
done