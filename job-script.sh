#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=30
#SBATCH --array=0-4

srun python main_polarization.py
