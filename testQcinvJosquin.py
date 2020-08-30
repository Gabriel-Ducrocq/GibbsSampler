import numpy as np
import healpy as hp
import qcinv
import utils
import matplotlib.pyplot as plt
from classy import Class
cosmo = Class()



## Setting params
LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None
COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]


nside=512
lmax=2*nside
npix = 12*nside**2

def generate_cls(theta, pol = False):
    params = {'output': OUTPUT_CLASS,
              "modes":"s,t",
              "r":0.001,
              'l_max_scalars': lmax,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(lmax)
    # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls_tt = cls["tt"]*2.7255e6**2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"]*2.7255e6**2
        cls_bb = cls["bb"]*2.7255e6**2
        cls_te = cls["te"]*2.7255e6**2
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt, cls_ee, cls_bb, cls_te


COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
noise_covar_temp = 0.1**2*np.ones(npix)
inv_noise = (1/noise_covar_temp)

beam_fwhm = 0.35
fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=lmax)




### Generating skymap

theta_ = COSMO_PARAMS_MEAN_PRIOR
cls_ = generate_cls(theta_, False)
map_true = hp.synfast(cls_, nside=nside, lmax=lmax, fwhm=beam_fwhm, new=True)
d = map_true
d += np.random.normal(scale=np.sqrt(noise_covar_temp))



#### Setting solver's params
class cl(object):
    pass

s_cls = cl
s_cls.cltt = cls_

eps = 1.0e-10
n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv(inv_noise, bl_gauss)
chain_descr = [
    [0, ["diag_cl"], lmax, nside, np.inf, eps, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]

chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, chain_descr, s_cls, n_inv_filt,
                                        debug_log_prefix=('log_'))

#soltn_complex = np.zeros(int(qcinv.util_alm.lmax2nlm(lmax)), dtype=np.complex)
soltn_complex = np.zeros(3, dtype=np.complex)
#### Solving
chain.solve(soltn_complex,d)
weiner_map_pcg = soltn_complex



##### Computing the exact weiner map:


## comouting b part of the system
b_weiner = hp.almxfl(hp.map2alm(inv_noise * d, lmax=lmax)*(npix/(4*np.pi)), bl_gauss)

inv_cls = np.array([1/cl if cl != 0 else 0 for cl in cls_])

Sigma = 1 / (inv_cls + inv_noise[0] * (npix / (4 * np.pi)) * bl_gauss ** 2)

##### Solving:
weiner_map_diag = hp.almxfl(b_weiner, Sigma)


##Graphics


pix_map_pcg = hp.alm2map(weiner_map_pcg, lmax=lmax, nside=nside)
pix_map_diag = hp.alm2map(weiner_map_diag, lmax=lmax, nside=nside)


rel_error_pix = np.abs((pix_map_pcg-pix_map_diag)/pix_map_diag)
print("Maximum relative error:", np.max(rel_error_pix))
hp.mollview(rel_error_pix)
plt.show()


plt.boxplot(rel_error_pix, showfliers=True)
plt.show()
plt.boxplot(rel_error_pix, showfliers=False)
plt.show()
fwd_op = qcinv.opfilt_tt.fwd_op(s_cls, n_inv_filt)
sol2 = fwd_op.calc(weiner_map_pcg)

residual = sol2 - b_weiner
print("External residual:", np.max(residual))
criterion = np.sum(hp.alm2cl(residual, lmax=lmax)*(2*np.arange(0, lmax+1)+1))
base_criterion = np.sum(hp.alm2cl(b_weiner, lmax=lmax)*(2*np.arange(0, lmax+1)+1))
print("criterion:", criterion)
print("base criterion:", base_criterion)
print("Criterion filled:", criterion < base_criterion*eps**2)

#error_ext = hp.almxfl(weiner_map_pcg, 1/Sigma)
#error_ext = hp.almxfl(weiner_map_diag, 1/Sigma)
rel_error_pix = np.abs((sol2 - b_weiner)/b_weiner)
print(np.max(rel_error_pix))
plt.boxplot(rel_error_pix, showfliers=True)
plt.show()
plt.boxplot(rel_error_pix, showfliers=False)
plt.show()