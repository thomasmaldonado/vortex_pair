#!/bin/bash
cd data
suffix=$1
sequence="$(seq -s "_${suffix}.png " 0 99)"
convert -delay 2 -loop 0 $sequence "${suffix}.gif"


