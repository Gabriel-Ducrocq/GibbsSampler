from CenteredGibbs import CenteredGibbs
import config
import numpy as np
import utils
import healpy as hp
from scipy.stats import invgamma
import matplotlib.pyplot as plt

def generate_dataset(polarization=False):
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
        return theta_, cls_, map_true,  d


theta_, cls_, s_true, pix_map = generate_dataset(polarization=False)
noise_temp = np.ones(config.Npix) * config.noise_covar_temp
centered_gibbs = CenteredGibbs(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                               config.Npix, n_iter=10000)

alms = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
alms_real = utils.complex_to_real(alms)
cl_hat = hp.alm2cl(alms, lmax=config.L_MAX_SCALARS)
alpha = np.array([(2*l-1)/2 if l !=0 else 0 for l in range(config.L_MAX_SCALARS+1)])
beta = np.array([(2*l+1)*l*(l+1)*observed_cl/(4*np.pi) for l, observed_cl in enumerate(cl_hat)])
alpha[0] = 1


print("BINS")
print(centered_gibbs.cls_sampler.bins)
h_true = []
h_gibbs = []
for i in range(100000):
    if i%100==0:
        print(i)

    h_true.append(invgamma.rvs(a=alpha, scale=beta))
    h_gibbs.append(centered_gibbs.cls_sampler.sample(alms_real))


l_interest = 2
plt.hist(np.array(h_true)[:, l_interest], density=True, alpha = 0.5, label="True", bins = 300)
plt.hist(np.array(h_gibbs)[:, l_interest], density=True, alpha = 0.5, label="Gibbs", bins = 300)
plt.legend(loc="upper right")
plt.show()
