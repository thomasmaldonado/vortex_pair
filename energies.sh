#!/bin/bash
for n in {0..99}; 
do
	sbatch energies.slurm $n
done
