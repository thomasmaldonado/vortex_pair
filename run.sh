#!/bin/bash

### SEQUENTIAL SEPARATION SWEEP (FOR LOCAL USE) ###

mkdir -p data
for n in {0..99}; 
do
	python solver.py $n 10 1 10 10 $n
	python plotter.py $n
done