import numpy as np
from CenteredGibbs import CenteredConstrainedRealization
import config
from main import generate_dataset
import utils
import matplotlib.pyplot as plt



noise = config.noise_covar_temp*np.ones(config.Npix)
theta_, cls_, s_true, pix_map = generate_dataset(polarization=False, mask_path=config.mask_path)


cr = CenteredConstrainedRealization(pix_map, noise, config.bl_map, config.beam_fwhm, config.L_MAX_SCALARS
                                    , config.Npix, mask_path=None, isotropic=True)




dls = np.array([l*(l+1)/(2*np.pi)*cl for l, cl in enumerate(cls_)])
var_cls = utils.generate_var_cl(dls)


h_true = []
h_pcg = []
for i in range(10000):
    if i % 10 == 0:
        print(i)

    samp_true, _ = cr.sample_no_mask(cls_, var_cls)
    samp_pcg, _ = cr.sample_mask(cls_, var_cls, np.zeros((config.L_MAX_SCALARS+1)**2))
    h_true.append(samp_true)
    h_pcg.append(samp_pcg)


h_pcg = np.array(h_pcg)
h_true = np.array(h_true)
l_interest = 16600
plt.hist(h_pcg[:, l_interest], density = True, alpha = 0.5, label = "PCG", bins = 50)
plt.hist(h_true[:, l_interest], density = True, alpha = 0.5, label = "True", bins = 50)
plt.legend(loc="upper right")
plt.show()




