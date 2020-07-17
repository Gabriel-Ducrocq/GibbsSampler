import healpy as hp
import numpy as np
import config
import matplotlib.pyplot as plt
import config
import utils
from CenteredGibbs import PolarizedCenteredConstrainedRealization

pix_map = [np.zeros(config.Npix)*(4*np.pi)/config.Npix, np.zeros(config.Npix), np.zeros(config.Npix)]
noise_I = 1
noise_Q = 2

def generate_var_cl(cls_):
    var_cl_full = np.concatenate([cls_,
                                  np.array(
                                      [cl for m in range(1, config.L_MAX_SCALARS + 1) for cl in cls_[m:] for _ in range(2)])])
    return var_cl_full

bl_gauss = np.ones(config.L_MAX_SCALARS+1)
bl_map = generate_var_cl(bl_gauss)



sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_I, noise_Q, bl_map, config.L_MAX_SCALARS, config.Npix, 0.1, isotropic=True)

all_dls = np.zeros((config.L_MAX_SCALARS+1, 3, 3))
for i in range(2, config.L_MAX_SCALARS+1):
    all_dls[i, :, :] = np.diag([1, 1, 1])

h_s = []
for i in range(1000):
    s, _, _ = sampler.sample(all_dls.copy())
    h_s.append(s)






