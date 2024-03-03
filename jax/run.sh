#!/bin/bash

### SEQUENTIAL SEPARATION SWEEP (FOR LOCAL USE) ###

mkdir -p data
for n in {0..49}; 
do
	python jsolver.py $n 50 1 40 40 $n
	python plotter.py $n
done