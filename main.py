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


def generate_dataset(polarization=True):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.beam_fwhm, new=True)
    d = map_true
    d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp))
    d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
    d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
    return theta_, cls_, map_true,  d



if __name__ == "__main__":
    np.random.seed()
    cls_init = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
    cls_init = np.concatenate([np.zeros(2), cls_init])
    #cls_init_binned = utils.generate_init_values(cls_init)
    #scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    #cls_init_binned = np.ones(len(cls_init_binned))*3000
    #cls_init_binned = np.random.normal(loc=cls_init_binned, scale=np.sqrt(10))
    #cls_init_binned[:2] = 0

    theta_, cls_, s_true, pix_map = generate_dataset()

    range_l = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    plt.plot(cls_[0]*range_l)
    plt.show()

    snr = cls_[0] * (config.bl_gauss ** 2) / (config.noise_covar_temp * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.title("TT")
    plt.show()

    snr = cls_[1] * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.title("EE")
    plt.show()

    snr = cls_[2] * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.title("BB")
    plt.show()

    snr = cls_[3] * (config.bl_gauss ** 2) / (config.noise_covar_temp * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.title("TE")
    plt.show()


    non_centered_gibbs = NonCenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol ,config.beam_fwhm,
                                          config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, proposal_variances=config.proposal_variances_nc, n_iter=100000)

    centered_gibbs = CenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, n_iter=100000)

    pncp_sampler = PNCPGibbs(pix_map, config.noise_covar_temp, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                             config.Npix, config.proposal_variances_pncp, config.l_cut, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 100000, n_iter_metropolis=1)

    asis_sampler = ASIS(pix_map, config.noise_covar_temp, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                            config.Npix, proposal_variances=config.proposal_variances_asis, n_iter=100000)


    polarized_centered = CenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, n_iter=10000, polarization=True)


    polarized_non_centered_gibbs = NonCenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol,
                                                    config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                                   config.Npix, proposal_variances=config.proposal_variances_nc_polarized, n_iter=100000, polarization=True)


    #h_cls_nc, _ = non_centered_gibbs.run(cls_init)
    #h_cls_centered, _ = centered_gibbs.run(cls_init)
    #h_cls_pncp, _, _ = pncp_sampler.run(cls_init)
    #h_asis, _, _ = asis_sampler.run(cls_init)

    init_cls = np.zeros((config.L_MAX_SCALARS+1, 3, 3))
    init_cls[:, 0, 0] = cls_[0]
    init_cls[:, 1, 1] = cls_[1]
    init_cls[:, 2, 2] = cls_[2]
    init_cls[:, 1, 0] = init_cls[:, 0, 1] = cls_[3]
    #init_cls[:, 0, 1] = cls_[3]
    for i in range(config.L_MAX_SCALARS+1):
        init_cls[i, :, :] *= i*(i+1)/(2*np.pi)

    #h_cls_pol, _ = polarized_centered.run(init_cls)

    #h_cls_pol, _ = polarized_non_centered_gibbs.run(init_cls)

    #d = {"h_cls_non_centered":h_cls_pol, "pix_map":pix_map, "cls_":cls_}
    #np.save("test_polarization_centered.npy", d, allow_pickle=True)


    d = np.load("test_polarization_centered.npy", allow_pickle=True)
    d = d.item()
    h_cls = d["h_cls_non_centered"]
    pix_map = d["pix_map"]


    pix_map_alm = hp.map2alm(pix_map, lmax=config.L_MAX_SCALARS)
    map_alm_B = pix_map_alm[2]
    all_pow_spec = hp.alm2cl(map_alm_B, lmax=config.L_MAX_SCALARS)

    l_interest = 4
    i = 0
    j = 1

    alpha = (2*l_interest-1)/2
    beta = ((2*l_interest+1)/2)*((l_interest*(l_interest+1))/(2*np.pi))*all_pow_spec[l_interest]*(1/config.bl_gauss[l_interest]**2)
    yy = []
    xx = []
    for x in np.linspace(0, 0.002, 1000):
        xx.append(x)
        y = scipy.stats.invgamma.pdf(x, a=alpha, scale=beta, loc = - (config.noise_covar_pol*4*np.pi/config.Npix)\
                                                                   *(l_interest*(l_interest+1)/(2*np.pi))*(1/config.bl_gauss[l_interest]**2))
        yy.append(y)

    plt.plot(h_cls[:, l_interest, i, j])
    plt.axhline(y=init_cls[l_interest, i, j])
    plt.show()

    plt.hist(h_cls[:, l_interest, i, j]*2*np.pi/(l_interest*(l_interest+1)), density=True, bins = 100)
    plt.plot(xx, yy)
    plt.show()

    print(yy)

    """
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
    """