#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=1080
#SBATCH --array=0-9

srun python main_polarization.py
