import numpy as np
import config
from CenteredGibbs import PolarizedCenteredConstrainedRealization
import main_polarization
import matplotlib.pyplot as plt
import healpy as hp
import utils

noise_temp = np.ones(config.Npix) * config.noise_covar_temp
noise_pol = np.ones(config.Npix) * config.noise_covar_pol
theta_, cls_, s_true, pix_map  = main_polarization.generate_dataset(True, None)
rescaling = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
all_dls = {"EE": cls_[1]*rescaling, "BB":cls_[2]*rescaling}


centered = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol, config.bl_map,
                                                   config.L_MAX_SCALARS, config.Npix, config.beam_fwhm, mask_path=None)

h_no_mask = {"EE":[], "BB":[]}
h_mask = {"EE":[], "BB":[]}
h_rj = {"EE":[], "BB": []}
d_Q = np.random.normal(size = config.Npix)
d_U = np.random.normal(size = config.Npix)
_, s_EE, s_BB = hp.map2alm([np.zeros(len(d_Q)) ,d_Q, d_U],lmax=config.L_MAX_SCALARS, pol=True)
s_old = {"EE":utils.complex_to_real(s_EE), "BB":utils.complex_to_real(s_BB)}
acceptions_rj = []
for i in range(1000):
    print("Iteration: ", i)
    alms, _ = centered.sample_no_mask(all_dls)
    h_no_mask["EE"].append(alms["EE"])
    h_no_mask["BB"].append(alms["BB"])

    alms_mask, _ = centered.sample_mask(all_dls)
    h_mask["EE"].append(alms_mask["EE"])
    h_mask["BB"].append(alms_mask["BB"])
    print("Now, reversble jump:")
    s_old, accept = centered.sample_mask_rj(all_dls, s_old)
    h_rj["EE"].append(s_old["EE"])
    h_rj["BB"].append(s_old["BB"])
    acceptions_rj.append(accept)

print("Acceptance rate RJPO: ", np.mean(acceptions_rj))

h_no_mask["EE"] = np.array(h_no_mask["EE"])
h_no_mask["BB"] = np.array(h_no_mask["BB"])

h_mask["EE"] = np.array(h_mask["EE"])
h_mask["BB"] = np.array(h_mask["BB"])

h_rj["EE"] = np.array(h_rj["EE"])
h_rj["BB"] = np.array(h_rj["BB"])
l_interests = [2, 5, 9, 13, 20, 24]
for pol in ["EE", "BB"]:
    for l in l_interests:
        plt.plot(h_no_mask[pol][:, l], label="No Mask", alpha = 0.5)
        plt.plot(h_mask[pol][:, l], label="Mask", alpha=0.5)
        plt.plot(h_rj[pol][:, l], label="RJ", alpha=0.5)
        plt.legend(loc = "upper right")
        plt.title(pol + " for l=" + str(l))
        plt.show()

        plt.hist(h_no_mask[pol][:, l], label="No Mask", alpha = 0.5, bins = 50, density=True)
        #plt.hist(h_mask[pol][:, l], label="Mask", alpha = 0.5, bins = 50, density=True)
        plt.hist(h_rj[pol][10:, l], label="RJ", alpha=0.5, bins=50, density=True)
        plt.legend(loc = "upper right")
        plt.title(pol + " for l=" + str(l))
        plt.show()



