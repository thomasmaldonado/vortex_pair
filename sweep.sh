module purge
module load anaconda3/2024.2

# check for conda environment
env_present=$(conda info --envs | grep -c "jax-gpu")
if [ $env_present == 0 ]
then
	conda create --name jax-gpu jax "jaxlib==0.4.23=cuda118*" sympy matplotlib -c conda-forge
fi

conda activate jax-gpu
NK=$(python params.py k)
MAX_K=$((NK-1))

sbatch -J $MAX_K sweep.slurm