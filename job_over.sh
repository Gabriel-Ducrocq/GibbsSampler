#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=10
#SBATCH --array=0-0

srun python test2.py
