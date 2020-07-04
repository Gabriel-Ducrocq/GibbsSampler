import utils
import config
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.integrate
import json
import scipy
import time
import pickle
import healpy as hp
from CenteredGibbs import CenteredGibbs
from NonCenteredGibbs import NonCenteredGibbs
from PNCP import PNCPGibbs
from ASIS import ASIS


def generate_dataset(planck=False):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_)
    var_cl_full = utils.generate_var_cl(cls_)

    alms_true = np.random.normal(scale=np.sqrt(var_cl_full)) * config.bl_map
    s_true = utils.synthesis_hp(alms_true)
    d = s_true + np.random.normal(scale=np.sqrt(config.var_noise))
    return theta_, cls_, s_true, alms_true, d



if __name__ == "__main__":
    np.random.seed()
    cls_init = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
    cls_init = np.concatenate([np.zeros(2), cls_init])
    #cls_init_binned = utils.generate_init_values(cls_init)
    #scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    #cls_init_binned = np.ones(len(cls_init_binned))*3000
    #cls_init_binned = np.random.normal(loc=cls_init_binned, scale=np.sqrt(10))
    #cls_init_binned[:2] = 0

    theta_, cls_, s_true, alm_, pix_map = generate_dataset()

    snr = cls_ * (config.bl_gauss ** 2) / (config.noise_covar * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.show()


    d = np.load("test.npy", allow_pickle=True)
    d = d.item()
    h_cls_nc = d["h_cls_nc"]
    h_cls_centered = d["h_cls_centered"]
    h_cls_pncp = d["h_cls_pncp"]
    pix_map = d["pix_map"]
    cls_ = d["cls_"]


    non_centered_gibbs = NonCenteredGibbs(pix_map, config.noise_covar, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, proposal_variances=config.proposal_variances_nc, n_iter=100000)

    centered_gibbs = CenteredGibbs(pix_map, config.noise_covar, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, n_iter=100000)

    pncp_sampler = PNCPGibbs(pix_map, config.noise_covar, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                             config.Npix, config.proposal_variances_pncp, config.l_cut, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 100000, n_iter_metropolis=1)

    asis_sampler = ASIS(pix_map, config.noise_covar, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                            config.Npix, proposal_variances=config.proposal_variances_asis, n_iter=100000)



    #h_cls_nc, _ = non_centered_gibbs.run(cls_init)
    #h_cls_centered, _ = centered_gibbs.run(cls_init)
    #h_cls_pncp, _, _ = pncp_sampler.run(cls_init)
    #h_asis, _, _ = asis_sampler.run(cls_init)

    #d = {"h_cls_nc":h_cls_nc, "h_cls_centered":h_cls_centered, "h_cls_pncp":h_cls_pncp, "h_cls_asis":h_asis ,
    #     "pix_map":pix_map, "cls_":cls_}
    #np.save("test.npy", d, allow_pickle=True)

    d = np.load("test.npy", allow_pickle=True)
    d = d.item()
    h_cls_centered = d["h_cls_centered"]
    h_cls_nc = d["h_cls_nc"]
    h_cls_pncp = d["h_cls_pncp"]
    h_cls_asis = d["h_cls_asis"]

    l_interest = 14
    plt.plot(h_cls_centered[:, l_interest], label="Centered", alpha=0.5)
    plt.plot(h_cls_nc[:, l_interest],label="NonCentered", alpha=0.5)
    plt.plot(h_cls_pncp[:, l_interest], label="PNCP", alpha=0.5)
    plt.plot(h_cls_asis[:, l_interest], label="ASIS", alpha=0.5)
    plt.legend(loc="upper right")
    plt.show()

    n_bins = 100
    plt.hist(h_cls_centered[:, l_interest], label="Centered", bins = n_bins, density=True, alpha=0.5)
    plt.hist(h_cls_nc[:, l_interest], label="NonCentered", bins = n_bins, density=True, alpha=0.5)
    plt.hist(h_cls_pncp[:, l_interest], label="PNCP", alpha=0.5, bins = n_bins, density=True)
    plt.hist(h_cls_asis[:, l_interest], label="ASIS", alpha=0.5, bins=n_bins, density=True)
    plt.legend(loc="upper right")
    plt.show()
