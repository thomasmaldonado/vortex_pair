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

K_IDX=0
A_IDX=27
OUTPUT=0/27
INPUT=0/26

sbatch -a $K_IDX-$K_IDX single.slurm $A_IDX $OUTPUT $INPUT