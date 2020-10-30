import numpy as np
from CenteredGibbs import PolarizedCenteredClsSampler, PolarizedCenteredConstrainedRealization
import config
from main_polarization import generate_dataset
import healpy as hp
import utils
import matplotlib.pyplot as plt
from NonCenteredGibbs import PolarizedNonCenteredConstrainedRealization

theta_, cls_, s_true, pix_map = generate_dataset(polarization=True, mask_path=config.mask_path)


lmax = config.L_MAX_SCALARS
nside = config.NSIDE
bins = config.bins
bl_map = config.bl_map
noise = config.noise_covar_pol

cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, bins, bl_map, noise)
map_sampler_nc = PolarizedNonCenteredConstrainedRealization(pix_map, config.noise_covar_temp*np.ones(config.Npix),
                                                            config.noise_covar_pol*np.ones(config.Npix),config.bl_map,
                                                            lmax, config.Npix, config.beam_fwhm)
map_sampler = PolarizedCenteredConstrainedRealization(pix_map, config.noise_covar_temp*np.ones(config.Npix),
                                                      config.noise_covar_pol*np.ones(config.Npix),config.bl_map,
                                                            lmax, config.Npix, config.beam_fwhm)




rescale = np.array([l*(l+1)/(2*np.pi) for l in range(lmax+1)])
all_dls = {"EE":rescale*cls_[1], "BB":rescale*cls_[2]}
var_cls_EE = utils.generate_var_cl(all_dls["EE"])
var_cls_BB = utils.generate_var_cl(all_dls["BB"])



h_nc = {"EE":[], "BB":[]}
h_cent = {"EE":[], "BB":[]}
for i in range(100000):
    nc, _ = map_sampler_nc.sample(all_dls)
    cent, _ = map_sampler.sample(all_dls)
    h_cent["EE"].append(cent["EE"])
    h_cent["BB"].append(cent["BB"])

    h_nc["EE"].append(np.sqrt(var_cls_EE)*nc["EE"])
    h_nc["BB"].append(np.sqrt(var_cls_BB)*nc["BB"])



h_nc["EE"] = np.array(h_nc["EE"])
h_nc["BB"] = np.array(h_nc["BB"])

h_cent["EE"] = np.array(h_cent["EE"])
h_cent["BB"] = np.array(h_cent["BB"])
for l in [2, 10, 50, 140, 210, 280]:
    for pol in ["EE", "BB"]:
        plt.hist(h_cent[pol][:, l], alpha = 0.5, density=True, label="Cenered", bins = 50)
        plt.hist(h_nc[pol][:, l], alpha=0.5, density=True, label="NonCentered", bins = 50)
        plt.legend(loc="upper right")
        plt.title(pol + " for l="+str(l))
        plt.show()



"""
d = {"EE":pix_map_E, "BB":pix_map_B}
h_dls = {"EE":[], "BB":[]}
for k in range(100000):
    if k % 1000:
        print(k)

    sampled_dls = cls_sampler.sample(d.copy())
    h_dls["EE"].append(sampled_dls["EE"])
    h_dls["BB"].append(sampled_dls["BB"])


h_dls["EE"] = np.array(h_dls["EE"])
h_dls["BB"] = np.array(h_dls["BB"])
for pol in ["EE", "BB"]:
    for l in range(2, lmax+1):
        y, xs, norm = utils.trace_likelihood_pol_binned_bis(h_dls, d.copy(), l, np.max(h_dls[pol][:, l]), pol)

        print(norm)
        plt.hist(h_dls[pol][:, l], density=True, alpha = 0.5, bins = 50)
        if norm < 1e-9:
            plt.plot(xs, y)
        else:
            plt.plot(xs, y/norm)

        plt.title(pol + " with l="+str(l))
        plt.show()
"""