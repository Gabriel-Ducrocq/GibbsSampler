import numpy as np
import matplotlib.pyplot as plt

"""
dim = 2
alpha = 10
noise = 1
beta = 0.5
var_latent = alpha*np.ones(dim)
var_noise = noise*np.ones(dim)
d = np.random.normal(size = dim)*np.sqrt(var_latent+var_noise)

Sigma = 1/(1/var_noise + 1/var_latent)
Mu = (Sigma/var_noise)*d
def pCN(x_old):
    x_prop = np.random.normal(size = dim)*np.sqrt(var_latent)
    x_new = np.sqrt(1-beta**2)*x_old + beta*x_prop

    log_ratio = -(1/2)*np.sum((d-x_new)**2/var_noise) + (1/2)*np.sum((d-x_old)**2/var_noise)
    if np.log(np.random.uniform()) < log_ratio:
        return x_new, 1

    return x_old, 0


def pCN_likelihood(y_old):
    y_prop = np.random.normal(size = dim)*np.sqrt(var_noise)
    y_new = np.sqrt(1-beta**2)*x_old + beta*y_prop

    log_ratio = -(1/2)*np.sum((d-y_new)**2/var_latent) + (1/2)*np.sum((d-y_old)**2/var_latent)
    if np.log(np.random.uniform()) < log_ratio:
        return y_new, 1

    return y_old, 0

x_old = 100*np.ones(dim)
y_old = 100*np.ones(dim)
all_accepts = 0
all_accepts_y = 0
x_h = []
y_h = []
N = 1000000
for i in range(N):
    x_old, accept = pCN(x_old)
    y_old, accept_y = pCN(y_old)
    all_accepts += accept
    all_accepts_y += accept_y
    x_h.append(x_old)
    y_h.append(y_old)

print(all_accepts/N)
print(all_accepts_y/N)
x_h = np.array(x_h)
y_h = np.array(y_h)


print(Mu)
print(np.mean(x_h, axis = 0))
print(np.mean(y_h, axis = 0))

plt.plot(x_h[:, 0], alpha = 0.5)
plt.plot(y_h[:, 0], alpha = 0.5)
plt.axhline(Mu[0])
plt.show()
"""



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

noise = np.ones(config.Npix)*config.noise_covar_temp
inv_noise = (1/noise)
theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
cls_TT, cls_EE, cls_BB, cls_TE = utils.generate_cls(theta_, True)

scale = np.array([(l*(l+1)/(2*np.pi)) for l in range(config.L_MAX_SCALARS+1)])
dls = cls_TT*scale
var_cls = utils.generate_var_cl(dls)
inv_var_cls = np.zeros(len(var_cls))
inv_var_cls[var_cls !=0] = 1/var_cls[var_cls != 0]

d_alms = utils.complex_to_real(hp.synalm(cls_TT, lmax=config.L_MAX_SCALARS)) + np.random.normal(size=(config.L_MAX_SCALARS+1)**2)
d_pCN = d_alms/config.bl_map
sigma_proposal = 1/((1/config.noise_covar_temp)*config.bl_map**2*(1/config.w))

def pCN_likelihood(s_old, beta = 0.1):
    s_prop = np.random.normal(size = len(var_cls))*np.sqrt(sigma_proposal)
    s_new = np.sqrt(1-beta**2)*s_old + beta * s_prop

    log_ratio = -(1/2)*np.sum((d_pCN - s_new)**2*inv_var_cls) + (1/2)*np.sum((d_pCN - s_old)**2*inv_var_cls)
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0



posterior_sigma = 1/((1/config.noise_covar_temp)*(1/config.w)*config.bl_map**2 + inv_var_cls)
posterior_mean = posterior_sigma*d_pCN

s_old = 0*np.ones(len(var_cls))
all_accept = 0
h_s = []
N = 10000
l_interest = 10000
beta = np.zeros(len(var_cls))
beta[:l_interest+1] = 0.8
for i in range(N):
    s_old, accept = pCN_likelihood(s_old, beta)
    all_accept += accept
    h_s.append(s_old[l_interest])


print(all_accept/N)
h_s = np.array(h_s)

plt.plot(h_s)
plt.axhline(posterior_mean[l_interest])
plt.show()

