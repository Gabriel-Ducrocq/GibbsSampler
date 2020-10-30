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
from scipy.stats import invwishart
from ASIS import ASIS
from default_gibbs import default_gibbs
from CenteredGibbs import CenteredGibbs


def generate_dataset(polarization=True, mask_path = None):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.beam_fwhm, new=True)
    d = map_true
    if polarization:
        d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        return theta_, cls_, map_true,  {"Q":d[1], "U":d[2]}
    else:
        d += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        if mask_path is None:
            return theta_, cls_, map_true,  d
        else:
            mask = hp.ud_grade(hp.read_map(mask_path, 0), config.NSIDE)
            return theta_, cls_, map_true, d*mask


def compute_marginal_TT(x_EE, x_TE, x_TT, l, scale_mat, cl_EE, cl_TE):
    param_mat = np.array([[x_TT, x_TE], [x_TE, x_EE]])
    if x_TT <= x_TE**2/x_EE:
        return 0
    else:
        return invwishart.pdf(param_mat, df=2*l-2, scale=scale_mat)




if __name__ == "__main__":
    np.random.seed()

    theta_, cls_, s_true, pix_map = generate_dataset(polarization=True, mask_path=config.mask_path)

    #data_path = "data/skymap_isotropic.npy"
    #d = np.load(data_path, allow_pickle=True)
    #d = d.item()
    #pix_map = d["d_"]
    #cls_ = d["cls_"]

    #d = {"pix_map":pix_map, "theta":theta_, "cls_":cls_, "s_true":s_true, "beam_fwhm":config.beam_fwhm,
    #     "mask_path":config.mask_path, "noise_rms":np.sqrt(config.noise_covar_temp), "nside":config.NSIDE,
    #     "LMAX":config.L_MAX_SCALARS}

    #np.save(config.scratch_path + "/data/non_isotropic_runs/skymap/skymap.npy", d, allow_pickle=True)
    #print(pix_map)
    #hp.mollview(pix_map)
    #plt.show()

    #data_path = config.scratch_path + "/data/non_isotropic_runs/skymap/skymap.npy"
    #data_path = "data/skymap.npy"
    #d = np.load(data_path, allow_pickle=True)
    #d = d.item()
    #pix_map = d["pix_map"]

    #range_l = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    #plt.plot(cls_[0]*range_l)
    #plt.show()

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

    noise_temp = np.ones(config.Npix) * config.noise_covar_temp
    noise_pol = np.ones(config.Npix) * config.noise_covar_pol

    centered_gibbs = CenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 100000)

    non_centered_gibbs = NonCenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 1000,
                                          proposal_variances=config.proposal_variances_nc_polarized, metropolis_blocks=config.blocks)


    l_interest =3

    np.random.seed()
    if config.preliminary_run:
        cls_init_E = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
        cls_init_B = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])

        cls_init_E_binned = np.concatenate([np.zeros(2), cls_init_E])
        #cls_init_E_binned = utils.generate_init_values(cls_init_E)

        cls_init_B_binned = np.concatenate([np.zeros(2), cls_init_B])
        #cls_init_B_binned = utils.generate_init_values(cls_init_B)

        scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])

        cls_init_E_binned = np.ones(len(cls_init_E_binned)) * 3000
        cls_init_B_binned = np.ones(len(cls_init_B_binned)) * 3000
        cls_init_binned_E = np.random.normal(loc=cls_init_E_binned, scale=np.sqrt(10))
        cls_init_binned_B = np.random.normal(loc=cls_init_B_binned, scale=np.sqrt(10))
        cls_init_E_binned[:2] = 0
        cls_init_B_binned[:2] = 0
        starting_point = {"EE":cls_init_E_binned, "BB":cls_init_B_binned}
        #starting_point = config.starting_point
    else:
        starting_point = config.starting_point

    ###Checker que ça marche avec bruit différent de 100**2
    #h_old_centered, _ = default_gibbs(pix_map, cls_init)
    start = time.time()
    #start_cpu = time.clock()
    h_cls_noncentered, h_accept_cr_noncentered, _ = non_centered_gibbs.run(starting_point)
    #h_cls_centered, h_accept_cr_centered, _ = centered_gibbs.run(starting_point)
    #h_cls_asis, h_accept, h_accept_cr_asis, times_asis = asis_sampler.run(starting_point)
    #h_cls_asis_gibbs, h_accept, h_accept_cr_asis_gibbs,times_asis_gibbs = asis_sampler_gibbs.run(starting_point)
    end = time.time()
    #end_cpu = time.clock()
    #h_cls_nonCentered, _, times = non_centered_gibbs.run(cls_init)
    total_time = end - start
    #total_cpu_time = end_cpu - start_cpu
    print("Total time:", total_time)
    #print("Total Cpu time:",total_cpu_time)

    #save_path = config.scratch_path + \
    #            "/data/non_isotropic_runs/asis/run/asis_" + str(config.slurm_task_id) + ".npy"


    for _, pol in enumerate(["EE", "BB"]):
        for l in range(2, config.L_MAX_SCALARS+1):
            y, xs, norm = utils.trace_likelihood_pol_binned(h_cls_noncentered[pol], pix_map, l, maximum=np.max(h_cls_noncentered[pol][:, l]), pol=pol)
            plt.plot(h_cls_noncentered[pol][:, l])
            plt.show()


            plt.hist(h_cls_noncentered[pol][100:, l], density=True, alpha=0.5, label="Gibbs", bins=100)
            print("Norm:", norm)
            plt.plot(xs, y/norm)
            plt.title(pol + " with l="+str(l))
            plt.show()

    """
    save_path = "test_nside_512.npy"

    d = {"h_cls": np.array(h_cls_asis), "bins": config.bins, "metropolis_blocks": config.blocks,
         "h_accept": np.array(h_accept),
         "h_times_iteration": np.array(times_asis), "h_cpu_time": None}

    np.save(save_path, d, allow_pickle=True)

    l_interest = 600
    plt.plot(h_cls_asis[:, l_interest])
    plt.savefig("Trajectory " + str(l_interest))
    """


