import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import inv, spsolve
import time
#from linear_algebra import product_cls_inverse, compute_cholesky, compute_inverse_matrices
import config
import utils
import healpy as hp
from CenteredGibbs import PolarizedCenteredConstrainedRealization, PolarizedCenteredClsSampler
from NonCenteredGibbs import PolarizedNonCenteredConstrainedRealization
import matplotlib.pyplot as plt
import main

theta_, cls_, map_true, d = main.generate_dataset(False, config.mask_path)

noise_up = 1.1*config.noise_covar_temp
noise_low = 0.9*config.noise_covar_temp
#noise = np.random.uniform(noise_low, noise_up, size = config.Npix)
noise = np.ones(config.Npix)*config.noise_covar_temp
inv_noise = (1/noise)
if config.mask_path is not None:
    mask = hp.ud_grade(hp.read_map(config.mask_path), config.NSIDE)
    inv_noise *= mask

theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
cls_TT, cls_EE, cls_BB, cls_TE = utils.generate_cls(theta_, True)



d = np.random.normal(size = config.Npix)
alms = hp.map2alm(d, lmax=config.L_MAX_SCALARS)
alms2 = config.w**2*(1/config.w)*hp.map2alm(hp.alm2map((1/config.w)*hp.map2alm(hp.alm2map(alms, nside=config.NSIDE, lmax=config.L_MAX_SCALARS), lmax=config.L_MAX_SCALARS), nside=config.NSIDE,lmax=config.L_MAX_SCALARS), lmax=config.L_MAX_SCALARS)
print(np.max(np.abs((alms-alms2)/alms)))

d = hp.synfast(cls_TT, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
d2 = hp.alm2map(hp.map2alm(d, lmax=config.L_MAX_SCALARS, iter= 10), nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
print("MAP")
print(np.abs((d-d2)/d))


snr = cls_TT * (config.bl_gauss ** 2) / (config.noise_covar_temp * 4 * np.pi / config.Npix)
plt.plot(snr)
plt.axhline(y=1)
plt.title("TT")
plt.show()
plt.close()

snr = cls_EE * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
plt.plot(snr)
plt.axhline(y=1)
plt.title("EE")
plt.show()
plt.close()

snr = cls_EE * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
plt.plot(snr)
plt.axhline(y=1)
plt.title("BB")
plt.show()
plt.close()

def pCN_step(s_old, var_cls, d, inv_noise, beta = 0.0001):
    s_prop = np.random.normal(size=len(var_cls))*np.sqrt(var_cls)
    s_new = np.sqrt(1-beta**2)*s_old + beta*s_prop
    s_new_pix = hp.alm2map(utils.real_to_complex(config.bl_map*s_new), lmax=config.L_MAX_SCALARS, nside=config.NSIDE)

    log_ratio = -(1/2)*np.sum((d - s_new_pix)**2*inv_noise) + (1/2)*np.sum((d - hp.alm2map(utils.real_to_complex(s_old),
                                                        lmax=config.L_MAX_SCALARS, nside=config.NSIDE))**2*inv_noise)
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0


def pCN_likelihood(s_old, inv_var_cls, d_alms, noise, inv_noise, beta = 0.9):
    sigma = 1/((1/config.noise_covar_temp)*(1/config.w)*config.bl_map**2)
    s_prop = np.sqrt(sigma)*np.random.normal(size = len(var_cls))
    s_new = np.sqrt(1-beta**2)*s_old + beta*s_prop
    mean = (1/config.bl_map)*d_alms

    log_ratio = -(1/2)*np.sum((s_new-mean)**2*inv_var_cls) + (1/2)*np.sum((s_old-mean)**2*inv_var_cls)
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0


def pCN_likelihood_sph(s_old, inv_var_cls, d_alms, noise, inv_noise, beta = 0.9):
    s_prop = config.w*(1/config.bl_map)*utils.adjoint_synthesis_hp(np.sqrt(noise)*np.random.normal(size=config.Npix))
    s_new = np.sqrt(1-beta**2)*s_old + beta*s_prop

    log_ratio = -(1/2)*np.sum((s_new+d_alms)**2*inv_var_cls) + (1/2)*np.sum((s_old+d_alms)**2*inv_var_cls)
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0


scale = np.array([(l*(l+1)/(2*np.pi)) for l in range(config.L_MAX_SCALARS+1)])
dls = cls_TT*scale
var_cls = utils.generate_var_cl(dls)
inv_var_cls = np.zeros(len(var_cls))
inv_var_cls[var_cls !=0] = 1/var_cls[var_cls != 0]
s_old = np.zeros(len(var_cls))
s_old = np.random.normal(0, 0.1, size = len(var_cls))
all_accept = 0
h_s = []
#d = np.zeros(config.Npix)
l_interest = 200000
d_alms = utils.complex_to_real(hp.synalm(cls_TT, lmax=config.L_MAX_SCALARS))
d = hp.alm2map(utils.real_to_complex(d_alms), lmax=config.L_MAX_SCALARS, nside=config.NSIDE)
#d_alms = np.zeros(len(d_alms))
#s_old = (1/(config.bl_map**2*(1/config.noise_covar_temp*config.w) + var_cls))*config.bl_map*utils.adjoint_synthesis_hp(inv_noise*d)
mean_post = (1/(config.bl_map**2*(1/config.noise_covar_temp)*(1/config.w) + inv_var_cls))*config.bl_map*utils.adjoint_synthesis_hp(inv_noise*d)
s_old = (1/config.bl_map)[l_interest]*d_alms[l_interest] - mean_post
N = 1001
h_s.append(s_old[l_interest])
#beta = np.array([np.exp(-l/100000000) for l in range(len(inv_var_cls))])(1 - np.sqrt(1-beta**2))*sigma*config.bl_map*hp.map2alm(inv_noise*d)
for i in range(1, N):
    print(i)
    s_old, accept = pCN_likelihood(s_old, inv_var_cls, d_alms, noise, inv_noise)
    #s_old, accept = pCN_step(s_old, var_cls, d, inv_noise, beta = 0.9)
    h_s.append((1/config.bl_map)[l_interest]*d_alms[l_interest] - s_old[l_interest])
    all_accept += accept
    print(all_accept/i)


print(all_accept/(N-1))
print("Mean:")
print(mean_post[l_interest])
overall_sigma = 1/((1/config.noise_covar_temp)*(1/config.w)*config.bl_map**2 + inv_var_cls)
true_l = np.random.normal(size = (1000))*np.sqrt(overall_sigma[l_interest]) + np.ones(1000)*mean_post[l_interest]
h_s = np.array(h_s)
plt.plot(h_s, alpha = 0.5)
plt.plot(true_l, alpha = 0.5)
plt.axhline(mean_post[l_interest], alpha = 0.5)
plt.show()

plt.hist(h_s[50:], alpha = 0.5, label="CN", density=True, bins= 25)
plt.hist(true_l, alpha = 0.5, label="True", density=True, bins = 25)
plt.show()
