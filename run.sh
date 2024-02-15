#!/bin/bash
#conda activate
#python solver.py 0.5 0.7 1 20 20 8 epsilon
#python plotter.py epsilon

for n in {0..9}; 
do
    echo $n
    #sbatch run.slurm $a/10 $n
done
