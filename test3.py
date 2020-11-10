from CenteredGibbs import CenteredGibbs
import config
import numpy as np
import utils
import healpy as hp
import main
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



theta_, cls_, s_true, pix_map = main.generate_dataset(polarization=False, mask_path=config.mask_path)
noise_temp = np.ones(config.Npix) * config.noise_covar_temp
centered_gibbs = CenteredGibbs(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                               config.Npix, n_iter=10000, mask_path=config.mask_path)


hp.mollview(pix_map)
plt.show()

rescale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
dls = cls_ * rescale
var_cls = utils.generate_var_cl(dls)
inv_var_cls = np.zeros(len(var_cls))
inv_var_cls[var_cls != 0] = 1/var_cls[var_cls!=0]
s_old = np.random.uniform(size = config.Npix)
s_old = utils.complex_to_real(hp.map2alm(s_old, lmax=config.L_MAX_SCALARS))

s_old2 = s_old.copy()
h_s = []
h_s2 = []
h_s.append(s_old)
for i in range(1000):
    print(i)
    s_old, _ = centered_gibbs.constrained_sampler.sample_gibbs_pix(var_cls, s_old)
    #s_old2, _ = centered_gibbs.constrained_sampler.sample_gibbs_change_variable(var_cls, s_old2)
    h_s.append(s_old)
    #h_s2.append(s_old2)



sigma = 1/(config.bl_map**2/(config.w*config.noise_covar_temp) + inv_var_cls)
mean = sigma*utils.adjoint_synthesis_hp((1/config.noise_covar_temp)*pix_map)*config.bl_map

l_interest = 2
true = mean[l_interest] + np.random.normal(size = 10000)*np.sqrt(sigma[l_interest])
h_s = np.array(h_s)
#h_s2 = np.array(h_s2)


plt.plot(h_s[:, l_interest], label="Pix", alpha = 0.5)
#plt.plot(h_s2[:, l_interest], label="Sph", alpha = 0.5)
plt.plot(true, label="True", alpha = 0.5)
plt.legend(loc = "upper right")
plt.show()


plt.hist(h_s[10:, l_interest], bins = 20, label="Pix", alpha = 0.5, density=True)
plt.hist(true, bins = 20, label="True", alpha = 0.5, density=True)
plt.legend(loc = "upper right")
plt.show()


plot_acf(h_s[100:, l_interest], lags = 100)
plt.show()

