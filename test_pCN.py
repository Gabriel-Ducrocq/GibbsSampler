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


low_noise = 0.9*config.noise_covar_temp
up_noise = 1.1*config.noise_covar_temp
#noise = np.ones(config.Npix)*config.noise_covar_temp
noise = np.random.uniform(low_noise, up_noise, size = config.Npix)
inv_noise = (1/noise)
theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
cls_TT, cls_EE, cls_BB, cls_TE = utils.generate_cls(theta_, True)

scale = np.array([(l*(l+1)/(2*np.pi)) for l in range(config.L_MAX_SCALARS+1)])
dls = cls_TT*scale
var_cls = utils.generate_var_cl(dls)
inv_var_cls = np.zeros(len(var_cls))
inv_var_cls[var_cls !=0] = 1/var_cls[var_cls != 0]

d_alms = utils.complex_to_real(hp.synalm(cls_TT, lmax=config.L_MAX_SCALARS)) + np.random.normal(size=(config.L_MAX_SCALARS+1)**2)
d = hp.alm2map(utils.real_to_complex(d_alms), nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
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


def pCN_independence(s_old):
    s_new = - sigma_proposal*inv_var_cls*(s_old - d_pCN) + np.random.normal(size=len(s_old))*np.sqrt(sigma_proposal)
    rho_u_v = np.sum((s_old - d_pCN)**2*inv_var_cls) + np.sum(s_new*inv_var_cls*(s_old-d_pCN))

    rho_v_u = np.sum((s_new - d_pCN)**2*inv_var_cls) + np.sum(s_old*inv_var_cls*(s_new-d_pCN))

    log_ratio = rho_u_v - rho_v_u
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0




def pCN_MALA_sph(s_old, delta = 1.8):
    s_new = ((2-delta)/(2+delta))*s_old - 2*(delta/(2+delta))* config.w**2*(1/config.bl_map)*utils.adjoint_synthesis_hp(noise*hp.alm2map(
        utils.real_to_complex((1/config.bl_map)*inv_var_cls*(s_old - d_pCN)), nside=config.NSIDE, lmax=config.L_MAX_SCALARS))\
            + (np.sqrt(8*delta)/(2+delta))*config.w*(1/config.bl_map)*utils.adjoint_synthesis_hp(np.sqrt(noise)*np.random.normal(size=config.Npix))

    rho_u_v = (1/2)*np.sum((s_old - d_pCN)**2*inv_var_cls) + (1/2)*np.sum((s_new - s_old)*inv_var_cls*(s_old - d_pCN)) \
    + (delta/4)*np.sum((s_old + s_new)*inv_var_cls*(s_old - d_pCN)) + (delta/4)*np.sum((config.w *np.sqrt(noise)* hp.alm2map(utils.real_to_complex((1/config.bl_map)*
                                                inv_var_cls*(s_old-d_pCN)), nside=config.NSIDE, lmax=config.L_MAX_SCALARS))**2)

    rho_v_u = (1/2)*np.sum((s_new - d_pCN)**2*inv_var_cls) + (1/2)*np.sum((s_old - s_new)*inv_var_cls*(s_new - d_pCN)) \
    + (delta/4)*np.sum((s_new + s_old)*inv_var_cls*(s_new - d_pCN)) + (delta/4)*np.sum((config.w*np.sqrt(noise)* hp.alm2map(utils.real_to_complex((1/config.bl_map)*
                                                inv_var_cls*(s_new-d_pCN)), nside=config.NSIDE, lmax=config.L_MAX_SCALARS))**2)

    log_ratio = rho_u_v - rho_v_u
    print("Log ratio:", log_ratio)
    if np.log(np.random.uniform()) < log_ratio:
        return s_new, 1

    return s_old, 0




posterior_sigma = 1/((1/config.noise_covar_temp)*(1/config.w)*config.bl_map**2 + inv_var_cls)
posterior_mean = posterior_sigma*config.bl_map*utils.adjoint_synthesis_hp(inv_noise*d)

s_old = 10*np.ones(len(var_cls))
all_accept = 0
h_s = []
N = 1000
l_interest = 10000
h_s.append(s_old[l_interest])
for i in range(N):
    #s_old, accept = pCN_likelihood(s_old)
    s_old, accept = pCN_MALA_sph(s_old)
    all_accept += accept
    h_s.append(d_pCN[l_interest] - s_old[l_interest])


true_sample = np.random.normal(size = 50000)*np.sqrt(posterior_sigma[l_interest]) + posterior_mean[l_interest]
print(all_accept/N)
h_s = np.array(h_s)

plt.plot(h_s, alpha = 0.5)
plt.plot(true_sample, alpha = 0.5)
plt.axhline(posterior_mean[l_interest])
plt.show()


plt.hist(h_s[100:], alpha = 0.5, label="MALA_PCN", bins = 10, density=True)
plt.hist(true_sample, alpha = 0.5, label="True", bins = 10, density=True)
plt.legend(loc="upper right")
plt.show()

