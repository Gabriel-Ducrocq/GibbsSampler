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


map = [np.random.normal(size = config.Npix), np.random.normal(size = config.Npix), np.random.normal(size = config.Npix)]
alms = hp.map2alm(map)
pow_spec_TT, pow_spec_EE, _, pow_spec_TE, _, _ = hp.alm2cl(alms, lmax=config.L_MAX_SCALARS)

l_interest = 4

scale_mat = np.zeros((2, 2))
scale_mat[0, 0] = pow_spec_TT[l_interest]
scale_mat[1, 1] = pow_spec_EE[l_interest]
scale_mat[1, 0] = scale_mat[0, 1] = pow_spec_TE[l_interest]
scale_mat *= (2*l_interest+1)

bins = np.array([l for l in range(2, config.L_MAX_SCALARS + 1)])
bins = {"TT": bins, "EE": bins, "TE": bins, "BB": bins}


cls_sampler = PolarizedCenteredClsSampler(map, config.L_MAX_SCALARS, bins, config.bl_map, config.noise_covar_temp)


h_cond = []
h_successes = []
start = time.time()

print("ALMS")
alm_TT = utils.complex_to_real(alms[0, :])
alm_EE = utils.complex_to_real(alms[1, :])
alm_BB = utils.complex_to_real(alms[2, :])
alms = np.vstack([alm_TT, alm_EE, alm_BB]).T
print(alms.shape)

"""
for i in range(10000):
    if i % 10 == 0:
        print("Numerical inversion, iteration",i)

    pow_spec_sampled = cls_sampler.sample_bin(alms.copy(), l_interest)
    h_cond.append(pow_spec_sampled)

end = time.time()
print("Time numerical inversion:", end-start)

h_direct = []
for i in range(100000):
    mat_sample = invwishart.rvs(df=2*l_interest-2, scale=scale_mat)
    h_direct.append(mat_sample[0, 0])


d = {"h_cond":np.array(h_cond), "h_direct":np.array(h_direct), "h_successes":np.array(h_successes)}
np.save("numeric_inverse_test_"+str(l_interest)+".npy", d, allow_pickle=True)
"""
d = np.load("numeric_inverse_test.npy", allow_pickle=True)
d = d.item()
h_cond = d["h_cond"]
h_direct = d["h_direct"]
h_successes = d["h_successes"]


plt.hist(h_cond, label="Cond", alpha=0.5, density=True, bins = 50)
plt.hist(h_direct[:], label="Direct", alpha = 0.5, density=True, bins = 100)
plt.legend(loc="upper right")
plt.savefig(str(l_interest)+".png")
