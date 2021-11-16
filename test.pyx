import numpy as np

cdef extern from "/global/homes/g/gabrield/.conda/envs/test/Healpix_3.80/src/cxx/Healpix_cxx/alm_healpix_tools.h":
    cdef void plm_gen(int nsmax, int nlmax, int nmmax, int plm)

def plmgen(nside, lmax, mmax, pol = True):
    n_plm = nside*(mmax + 1)*(2*lmax - mmax + 2)
    print(pol)
    if pol:
        plm_array = np.empty((n_plm, 3))
    else:
        plm_array = np.empty((n_plm, 1))
        
    plm_gen(nside, lmax, mmax, plm_array)
    return plm_array
    