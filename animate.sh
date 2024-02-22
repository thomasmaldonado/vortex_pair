#!/bin/bash
cd data
suffix=$1
sequence="$(seq -s "_${suffix}.png " 0 99)"
sequence_reversed="$(seq -s "_${suffix}.png " 99 0)"
convert -delay 2 -loop 0 $sequence $sequence_reversed "${suffix}.gif"


