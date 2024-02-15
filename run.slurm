#!/bin/bash
#SBATCH --job-name=pair   # create a short name for your job
#SBATCH --nodes=1            # node count
#SBATCH --ntasks=1           # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
###SBATCH --mem=90G
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
###SBATCH -p vlong
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=maldonado@princeton.edu

python -u solver.py $1 0.7 1 40 40 4 $1
