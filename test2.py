import healpy as hp
import numpy as np
import config
import matplotlib.pyplot as plt
import config
import utils
from CenteredGibbs import PolarizedCenteredConstrainedRealization
from scipy.stats import norm

alm = np.ones((config.L_MAX_SCALARS+1)**2)
alm = utils.real_to_complex(alm)
alm_first = [alm,alm,alm]

pix_map = hp.alm2map(alm_first, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)*4*np.pi/(config.Npix)
noise_I = config.Npix/(4*np.pi)
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
    all_dls[i, :, :] = np.diag([1, 1, 1])*i*(i+1)/(2*np.pi)*2

h_s = []
for i in range(10000):
    s, _, _ = sampler.sample(all_dls.copy())
    h_s.append(s)


mu = np.pi/18
x = np.linspace(-4, 4, 100000)
l_interest = 2
y = norm.pdf(x, loc = mu, scale=np.sqrt(2/3))

h_s = np.array(h_s)
plt.hist(h_s[:, l_interest, 0], density=True, bins = 30)
plt.plot(x, y)
plt.show()

print((2*np.pi*l_interest)/(12*l_interest+np.pi))
print(np.var(h_s[:, l_interest, 0]))
print(mu)
print(np.mean(h_s[:, l_interest, 0]))

