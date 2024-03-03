#!/bin/bash

### ANIMATION ###

# convert sequence of pngs of the form data/n_suffix.png into gif
cd data
suffix=$1
sequence="$(seq -s "_${suffix}.png " 0 31)"
sequence_reversed="$(seq -s "_${suffix}.png " 31 0)"
convert -delay 2 -loop 0 $sequence $sequence_reversed "${suffix}.gif"
