from classy import Class
import config
import matplotlib.pyplot as plt
import numpy as np
import camb
import time
import utils
import healpy as hp


start = time.time()
utils.generate_cls(config.COSMO_PARAMS_MEAN_PRIOR, pol=True)
end = time.time()
print("Duration Cls:", end-start)

start = time.time()
hp.map2alm([np.zeros(config.Npix),np.zeros(config.Npix), np.zeros(config.Npix)], lmax=config.L_MAX_SCALARS, pol=True, iter=0)
end = time.time()
print("Duration map2alm:", end - start)
