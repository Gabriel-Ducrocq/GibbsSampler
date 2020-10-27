import numpy as np
from CenteredGibbs import PolarizedCenteredClsSampler
import config
from main import generate_dataset
import healpy as hp
import utils
import matplotlib.pyplot as plt

theta_, cls_, s_true, pix_map = generate_dataset(polarization=True, mask_path=config.mask_path)


lmax = config.L_MAX_SCALARS
nside = config.NSIDE
bins = config.bins
bl_map = config.bl_map
_, pix_map_E, pix_map_B = hp.synalm(cls_, lmax=lmax, new = True)
pix_map_E = bl_map * utils.complex_to_real(pix_map_E)
pix_map_B = bl_map * utils.complex_to_real(pix_map_B)
noise = config.noise_covar_pol

cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, bins, bl_map, noise)
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
