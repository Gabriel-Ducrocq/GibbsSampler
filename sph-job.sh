#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --time=180
#SBATCH -q bigmem
#SBATCH --array=0-0

srun python sph_computing.py 256 512
