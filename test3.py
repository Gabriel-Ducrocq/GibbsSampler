from CenteredGibbs import CenteredGibbs
import config
import numpy as np
import utils
import healpy as hp
import main
from scipy.stats import invgamma
import matplotlib.pyplot as plt



theta_, cls_, s_true, pix_map = main.generate_dataset(polarization=False, mask_path=config.mask_path)
noise_temp = np.ones(config.Npix) * config.noise_covar_temp
centered_gibbs = CenteredGibbs(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                               config.Npix, n_iter=10000)


hp.mollview(pix_map)
plt.show()

rescale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
dls = cls_ * rescale
var_cls = utils.generate_var_cl(dls)
s_old = np.random.uniform(size = config.Npix)
s_old = utils.complex_to_real(hp.map2alm(s_old, lmax=config.L_MAX_SCALARS))


h_s = []
h_s.append(s_old)
for i in range(10000):
    print(i)
    s_old, _ = centered_gibbs.constrained_sampler.sample_gibbs_pix(var_cls, s_old)
    h_s.append(s_old)



h_s = np.array(h_s)


plt.plot(h_s[:, 10])
plt.show()

