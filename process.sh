#!/bin/bash

### POST-PROCESSING (ENERGY PLOT + GIFS) ###

python plot_energy.py

NK=$(python params.py K)
MIN_K=0
MAX_K=$((NK-1))

for K_IDX in $(seq $MIN_K $MAX_K)
do	
    echo $K_IDX
    bash animate.sh V $K_IDX
    bash animate.sh Fu $K_IDX
    bash animate.sh Fv $K_IDX
    bash animate.sh C $K_IDX
    bash animate.sh EED $K_IDX
    bash animate.sh MED $K_IDX
    bash animate.sh HED $K_IDX
    bash animate.sh V_cart $K_IDX
    bash animate.sh Fu_cart $K_IDX
    bash animate.sh Fv_cart $K_IDX
    bash animate.sh C_cart $K_IDX
    bash animate.sh EED_cart $K_IDX
    bash animate.sh MED_cart $K_IDX
    bash animate.sh HED_cart $K_IDX
    bash animate.sh Ji $K_IDX
done


