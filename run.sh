#!/bin/bash

### SEQUENTIAL SEPARATION SWEEP (FOR LOCAL USE) ###

mkdir -p data

NA=$(python params.py a)
MIN_A=0
MAX_A=$((NA-1))

A_IDX=$MAX_A
K_IDX=6
NL=1
NR=1
NU=20
NV=20

OUTPUT=$A_IDX
python jsolver.py $K_IDX $A_IDX $NL $NR $NU $NV $OUTPUT $INPUT
INPUT=$OUTPUT

for A_IDX in $(seq $((MAX_A-1)) -1 $MIN_A)
do	
	echo $A_IDX
	OUTPUT=$A_IDX
	python -u jsolver.py $K_IDX $A_IDX $NL $NR $NU $NV $OUTPUT $INPUT
	INPUT=$OUTPUT
done
python plot_energy.py