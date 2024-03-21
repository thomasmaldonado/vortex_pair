#!/bin/bash

### PARALLIZED SEPARATION SWEEP (FOR REMOTE USE) ###

#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:60:00         # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.2
conda activate jax-gpu
export PATH=$PATH:/usr/local/cuda-12.3/bin # some mysterious line of code that the della staff told me to include

K_IDX=$SLURM_ARRAY_TASK_ID # (kappa index, with kappa taken from params.py)
mkdir -p data/$K_IDX

NL=1 # left winding number
NR=1 # right winding number
NU=85 # number of u points (resolution)
NV=85 # number of v points (resolution)

# read number of separations to be swept from params.py
NA=$(python params.py a)
MIN_A=0
MAX_A=$((NA-1))

# solve and plot for each separation
for A_IDX in $(seq $MIN_A $MAX_A)
do	
    OUTPUT=$K_IDX/$A_IDX # output file
    python -u jsolver.py $K_IDX $A_IDX $NL $NR $NU $NV $OUTPUT $INPUT
    python plotter.py $OUTPUT
    INPUT=$OUTPUT
done

# process data
bash animate.sh V $K_IDX
bash animate.sh Fu $K_IDX
bash animate.sh Fv $K_IDX
bash animate.sh C $K_IDX
bash animate.sh EED_cart $K_IDX
bash animate.sh MED_cart $K_IDX
bash animate.sh HED_cart $K_IDX
bash animate.sh J0_cart $K_IDX
bash animate.sh Ji $K_IDX
python plot_energy.py
