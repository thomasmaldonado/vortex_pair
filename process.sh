#!/bin/bash

### POST-PROCESSING (ENERGY PLOT + GIFS) ###

python plot_energy.py
bash animate.sh V
bash animate.sh Fu
bash animate.sh Fv
bash animate.sh C
bash animate.sh EED
bash animate.sh MED
bash animate.sh HED
bash animate.sh V_cart
bash animate.sh Fu_cart
bash animate.sh Fv_cart
bash animate.sh C_cart
bash animate.sh EED_cart
bash animate.sh MED_cart
bash animate.sh HED_cart