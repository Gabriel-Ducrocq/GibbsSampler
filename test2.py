import healpy as hp
import numpy as np
import config
import matplotlib.pyplot as plt
import config
import utils


nside = 512
Npix = 12*nside**2
lmax = 2*nside

def generate_var_cl(cls_):
    var_cl_full = np.concatenate([cls_,
                                  np.array(
                                      [cl for m in range(1,lmax+ 1) for cl in cls_[m:] for _ in range(2)])])
    return var_cl_full

beam_fwhm = 0.35
fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=lmax)
bl_map = generate_var_cl(bl_gauss)
var_noise_temp = 40**2


theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale=config.COSMO_PARAMS_SIGMA_PRIOR)
cls_ = utils.generate_cls(theta_, False)
map_true = hp.synfast(cls_, nside=nside, lmax=lmax, fwhm=beam_fwhm, new=True)
d = map_true
d += np.random.normal(scale=np.sqrt(var_noise_temp))
pix_map = d

#d = np.load("test_polarization.npy", allow_pickle=True)
#d = d.item()
#pix_map = d["pix_map"]


Sigma = (Npix/(4*np.pi*var_noise_temp))*bl_map**2

print(Sigma.shape)
mu = Sigma*bl_map*utils.adjoint_synthesis_hp(pix_map)/var_noise_temp
all_weights = []
power = np.array([(2*i-1)/2 for i in range(lmax+1)])
for i in range(1000):
    if i % 10 == 0:
        print(i)

    map = np.random.normal(size=(len(Sigma)))*np.sqrt(Sigma) + mu
    alms = utils.real_to_complex(map)
    sigmas_all_l = hp.alm2cl(alms, lmax=lmax)
    log_w = np.sum(-np.log(sigmas_all_l)*power)
    all_weights.append(log_w)


all_weights = np.array(all_weights)
max_w = np.max(all_weights)
ESS = np.sum(np.exp(all_weights - max_w))**2/np.sum(np.exp(all_weights-max_w)**2)

print(ESS)