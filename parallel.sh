#!/bin/bash

### SEQUENTIAL SEPARATION SWEEP (FOR LOCAL USE) ###

mkdir -p data

NK=$(python params.py k)
MIN_K=0
MAX_K=$((NK-1))

for K_IDX in $(seq $((MAX_K-1)) -1 $MIN_K)
do	
	mkdir -p data/$K_IDX
	sbatch parallel.slurm $K_IDX
done
