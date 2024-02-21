#!/bin/bash

for n in {98..0}; 
do
    np1=$(($n+1))
    python solver_init.py $n 20 1 30 30 $np1 $n
done

