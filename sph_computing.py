import pyshtools
import numpy as np
import multiprocessing as mp
import healpy as hp
import argparse
import os
import time
import gc
import numba as nb
from numba import prange, njit

scratch_path = os.environ['SCRATCH']
#slurm_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
#slurm_task_id = 0
parser = argparse.ArgumentParser(description="Compute the spherical harmonics in real form")
parser.add_argument("NSIDE", help="NSIDE", type=int)
parser.add_argument("LMAX", help="LMAX", type=int)


#@njit()
def get_lm(i, lmax):
    m = int(np.ceil(((2 * lmax + 1) - np.sqrt((2 * lmax + 1) ** 2 - 8 * (i - lmax))) / 2))
    l = i - m * (2 * lmax + 1 - m) // 2
    return (l,m)
  
#@njit()
def get_all_lm(start_index, stop_index, lmax):
    return [get_lm(i, lmax) for i in prange(start_index, stop_index)]


def compute_ylm(args):
    triu_indices = np.triu_indices
    theta, phi, LMAX = args
    res = pyshtools.expand.spharm(LMAX, theta, phi, kind="real", normalization="ortho", degrees=False, csphase=-1)
    triu_indices = np.triu_indices(res[0].shape[0])
    res1 = res[0].T[triu_indices]
    res2 = res[1].T[triu_indices]
    
    return (res1, res2)


arguments = parser.parse_args()                                                 
NSIDE = arguments.NSIDE                                              
LMAX = arguments.LMAX 
Npix = 12*NSIDE**2
Total_lm = int((LMAX +1)*(LMAX+2)/2)

#start_index = int(slurm_task_id*Total_lm//N_MACHINES)
#stop_index = int(min((slurm_task_id+1)*Total_lm//N_MACHINES, Total_lm))
theta, phi = hp.pixelfunc.pix2ang(NSIDE, [i for i in range(Npix)])
print("Create arguments")
arguments = [(theta[i], phi[i], LMAX) for i, _ in enumerate(theta)]


pool = mp.Pool(mp.cpu_count())

print("Trying to create the matrix:")
print("Done !")
print("Number of CPUs:", mp.cpu_count())
start = time.time()
results = pool.map(compute_ylm, arguments)
end = time.time()
print("Total time:", end-start)

Re = np.array([res[0] for res in results]).T
print(Re.shape)
np.save(scratch_path + "/data/spherical_harm/sph_all_real"+str(slurm_task_id)+".npy", Re)
del Re
gc.collect()

Im = np.array([res[1] for res in results]).T
print(Im.shape)
np.save(scratch_path + "/data/spherical_harm/sph_all_imag_"+str(slurm_task_id)+".npy", Im)




