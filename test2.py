import healpy as hp
import numpy as np
import config
import matplotlib.pyplot as plt
import config
import utils
from CenteredGibbs import PolarizedCenteredConstrainedRealization, PolarizedCenteredClsSampler
from scipy.stats import norm, invgamma, invwishart
from scipy.stats import t as student
import scipy
import time
from scipy.special import gammaln, multigammaln
import mpmath
import main


inv_noise = 1/config.noise_covar_temp
mu = inv_noise + 1e-5
bl_gauss = config.bl_gauss
bl_map = config.bl_map
nside = config.NSIDE
lmax = config.L_MAX_SCALARS
alpha = 1


theta_, cls_, s_true, pix_map  = main.generate_dataset(False, None)
var_cls_true = utils.generate_var_cl(cls_)


def sample_gibbs_change_variable(var_cls, old_s):
    old_s = utils.real_to_complex(old_s)
    var_v = (mu - inv_noise)*alpha**2
    mean_v = alpha*var_v * hp.alm2map(hp.almxfl(old_s, bl_gauss), nside=nside, lmax=lmax)
    v = np.random.normal(size=len(mean_v)) * np.sqrt(var_v) + mean_v

    inv_var_cls = np.zeros(len(var_cls))
    inv_var_cls[np.where(var_cls != 0)] = 1 / var_cls[np.where(var_cls != 0)]
    var_s = 1 / ((mu / config.w) * bl_map ** 2 + inv_var_cls)
    mean_s = var_s * utils.complex_to_real(
        hp.almxfl(hp.map2alm((v/alpha + inv_noise * pix_map), lmax=lmax) * (1 / config.w), bl_gauss))
    s_new = np.random.normal(size=len(mean_s)) * np.sqrt(var_s) + mean_s
    return s_new, v


def run_gibbs(var_cls, l_interest):
    h_s = []
    h_v = []
    s = np.zeros(len(var_cls))
    for i in range(10000):
        if i % 10 == 0:
            print(i)

        s, v = sample_gibbs_change_variable(var_cls, s)
        h_s.append(s)
        h_v.append(v)


    print("Alpha end", alpha)
    return np.array(h_s), np.array(h_v)


h_s, h_v = run_gibbs(var_cls_true, 30)
#alpha = 1
#h_s2, h_v2 = run_gibbs(var_cls_true, 30)

Sigma = np.cov(h_s.T, h_v.T)
#Sigma2 = np.cov(h_s2.T, h_v2.T)

#diff_sigma = np.abs((Sigma-Sigma2)/Sigma2)
#plt.imshow(diff_sigma)
#plt.show()

plt.imshow(Sigma)
plt.show()
#plt.imshow(Sigma2)
#plt.show()
print(np.var(h_s[:, 20]))
#print(np.var(h_s2[:, 20]))

plt.plot(h_s[:, 20], alpha = 0.5)
#plt.plot(h_s2[:, 20], alpha = 0.5)
plt.show()


