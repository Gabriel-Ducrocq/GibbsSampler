#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=1440
#SBATCH --array=0-19

srun python main.py
