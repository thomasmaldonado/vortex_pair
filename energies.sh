#!/bin/bash
A_IDX=99
K=0.7
N=1
NU=20
NV=20

python solver.py $A_IDX $K $N $NU $NV $A_IDX
python plotter.py $A_IDX
python energies.py $A_IDX
for A_IDX in {98..0}; 
do
	#A_IDX_PLUS_1=$(($A_IDX+1))
    #python solver_init.py $A_IDX $K $N $NU $NV $A_IDX_PLUS_1 $A_IDX
    python solver.py $A_IDX $K $N $NU $NV $A_IDX
	python plotter.py $A_IDX
	python energies.py $A_IDX
done
python plot_energy.py
