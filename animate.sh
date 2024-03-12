#!/bin/bash

### ANIMATION ###

# convert sequence of pngs of the form data/n_suffix.png into gif
suffix=$1
K_IDX=$2
cd data/$K_IDX
sequence="$(seq -s "_${suffix}.png " 0 19)"_$suffix.png
sequence_reversed="$(seq -s "_${suffix}.png " 19 -1 0)"_$suffix.png
convert -delay 2 -loop 0 $sequence $sequence_reversed "${suffix}.gif"