#!/bin/bash
#conda activate
#python solver.py 0.5 0.7 1 20 20 8 epsilon
#python plotter.py epsilon

for n in {0..99}; 
do
    path='data/'
    path+=$n
    path+='.npy'
    if [ -f $path ]; then
        echo $path 'exists'
    else
       	sbatch run_rec.slurm $n
    fi
done
