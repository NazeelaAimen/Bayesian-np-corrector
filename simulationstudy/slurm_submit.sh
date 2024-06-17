#!/bin/bash
#
#SBATCH --job-name=pspline_simulation_study
#SBATCH --output=logs/download_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5



ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 r/4.2.1
source /fred/oz303/naimen/bnpc/lisa_venv/bin/activate
python simulationstudy.py $SLURM_ARRAY_TASK_ID 





