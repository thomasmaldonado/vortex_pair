#!/bin/bash

for n in {98..0}; 
do
    np1=$(($n+1))
    python solver_init.py $n 0.7 1 20 20 $np1 $n
done
