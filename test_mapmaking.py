
import sys
import qcinv

import imp, os, sys, getopt

import numpy  as np
import healpy as hp
import config
import utils
import matplotlib.pyplot as plt



d = np.load("data/skymap_isotropic.npy", allow_pickle=True)
d = d.item()
for k in d.keys():
    print(k)

d = d["d_"]
print(len(d))
print(config.Npix)
hp.mollview(d)
plt.show()


def generate_dataset(polarization=True, mask_path = None):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.beam_fwhm, new=True)
    d = map_true
    if polarization:
        d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        return theta_, cls_, map_true,  d
    else:
        d += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        if mask_path is None:
            return theta_, cls_, map_true,  d
        else:
            mask = hp.ud_grade(hp.read_map(mask_path, 0), config.NSIDE)
            return theta_, cls_, map_true, d*mask





mask=hp.ud_grade(hp.read_map(config.mask_path), config.NSIDE)
np.mean(mask)


theta_, cls_, map_true,  d = generate_dataset(False, config.mask_path)

tr_cg   = qcinv.cd_solve.tr_cg

lmax=config.L_MAX_SCALARS
nside=config.NSIDE

noise = np.ones(config.Npix)*config.noise_covar_temp
inv_noise = 1/noise
if config.mask_path is not None:
    inv_noise *= mask
#chain_descr=[	  [  2,    ["split(dense,     32, diag_cl)"],    128,     32,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
#                  [  1,    ["split(stage(2), 128, diag_cl)"],    256,    128,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
#                  [  0,    ["split(stage(1), 256, diag_cl)"],    512,    256,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()] ]
chain_descr  =  [ [  0,    ["diag_cl"],    lmax,   nside,  np.inf,  1.0e-6,   qcinv.cd_solve.tr_cg,    qcinv.cd_solve.cache_mem()] ]


class cl(object):
    pass
s_cls = cl
s_cls.cltt = cls_
s_cls.lmax = lmax
beam=config.bl_gauss

n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv(inv_noise, config.bl_gauss)
chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, chain_descr, s_cls, n_inv_filt,
                                        debug_log_prefix=None)

# construct the chain.
# note: just-in-time instantiation here is useful e.g.
#       if n_inv_filt and chain are defined in a parameter
#       file and don't necessarily need to be loaded.

soltn  = np.zeros(int(qcinv.util_alm.lmax2nlm(lmax)), dtype=np.complex)

print('RUNNING SOLVE')
chain.solve( soltn, d )

soltnm=hp.alm2map([soltn.elm*0,hp.almxfl(soltn.elm,s_cls.clee*beam),hp.almxfl(soltn.blm,s_cls.clbb*beam)],512)




