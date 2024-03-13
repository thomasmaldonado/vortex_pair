#!/bin/bash

### ANIMATION ###

# convert sequence of pngs of the form data/n_suffix.png into gif
suffix=$1
K_IDX=$2

NA=$(python params.py a)
MIN_A=0
MAX_A=$((NA-1))

cd data/$K_IDX
sequence="$(seq -s "_${suffix}.png " 0 $MAX_A)"_$suffix.png
sequence_reversed="$(seq -s "_${suffix}.png " $MAX_A -1 0)"_$suffix.png
convert -delay 2 -loop 0 $sequence $sequence_reversed "${suffix}.gif"