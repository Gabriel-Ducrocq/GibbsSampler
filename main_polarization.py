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


def generate_cls(polarization = True):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    return theta_, cls_

def generate_dataset(cls_, polarization=True, mask_path = None):
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.fwhm_radians, new=True)
    d = map_true
    if polarization:
        d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        if mask_path is None:
            #hp.mollview(d[1])
            #plt.show()
            #hp.mollview(d[2])
            #plt.show()
            return map_true,  {"Q":d[1], "U":d[2]}
        else:
            mask = hp.ud_grade(hp.read_map(mask_path, 0), config.NSIDE)
            hp.mollview(mask)
            plt.show()
            hp.mollview(mask*d[1])
            plt.show()
            hp.mollview(mask*d[2])
            plt.show()
            print("Mask taken into account")
            return map_true, {"Q": d[1]*mask, "U": d[2]*mask}

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

    ####Be careful of the cls_TT and cls_TE
    theta_, cls_ = generate_cls()
    cls_[0] = np.zeros(len(cls_[0]))
    cls_[3] = np.zeros(len(cls_[0]))
    s_true, pix_map = generate_dataset(cls_, polarization=True, mask_path=config.mask_path)






    d = {"pix_map":pix_map, "params_":theta_, "skymap_true": s_true, "cls_":cls_, "fwhm_arcmin_beam":config.beam_fwhm,
         "noise_var_temp":config.noise_covar_temp, "noise_var_pol":config.noise_covar_pol, "mask_path":config.mask_path,
         "NSIDE":config.NSIDE, "lmax":config.L_MAX_SCALARS}

    np.save(config.scratch_path + "/data/polarization_runs/cut_sky/skymap/skymap.npy", d, allow_pickle=True)

    data_path = config.scratch_path + "/data/polarization_runs/full_sky/skymap/skymap.npy"
    d = np.load(data_path, allow_pickle=True)
    d = d.item()
    pix_map = d["pix_map"]
    """
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

    print("Variances pol FIRST")
    print(config.proposal_variances_nc_polarized)
    """
    d_sph_TT, d_sph_EE, d_sph_BB = hp.map2alm([np.zeros(len(pix_map["Q"])), pix_map["Q"], pix_map["U"]], lmax=config.L_MAX_SCALARS)
    pix_map = {"EE": utils.complex_to_real(d_sph_EE), "BB":utils.complex_to_real(d_sph_BB)}
    noise_temp = np.ones(config.Npix) * config.noise_covar_temp
    noise_pol = np.ones(config.Npix) * config.noise_covar_pol

    centered_gibbs = CenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 10000,
                                   rj_step=False)

    ### ALL SPH ACTIVATED
    non_centered_gibbs = NonCenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 10000,
                                          proposal_variances=config.proposal_variances_nc_polarized, metropolis_blocks=config.blocks,
                                          all_sph=True)


    ### ALL SPH ACTIVATED
    asis = ASIS(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 10000,
                                          proposal_variances=config.proposal_variances_nc_polarized, metropolis_blocks=config.blocks,
                                    rj_step = False, all_sph=True)


    l_interest =3

    np.random.seed()
    if config.preliminary_run:
        _, cls_EE, cls_BB, _ = utils.generate_cls(config.COSMO_PARAMS_PLANCK, pol=True)
        scale = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)])
        dls_EE = scale*cls_EE
        dls_BB = scale*cls_BB
        all_dls = {"EE":dls_EE, "BB":dls_BB}

        starting_point = {"EE":[], "BB":[]}
        for pol in ["EE", "BB"]:
            for i, l_start in enumerate(config.bins[pol][:-1]):
                l_end = config.bins[pol][i+1]
                starting_point[pol].append(np.mean(all_dls[pol][l_start:l_end]))

        starting_point["EE"] = np.array(starting_point["EE"])
        starting_point["BB"] = np.array(starting_point["BB"])

        #cls_init_E = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
        #cls_init_B = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])

        #cls_init_E_binned = np.concatenate([np.zeros(2), cls_init_E])
        #cls_init_E_binned = utils.generate_init_values(cls_init_E, pol="EE")
        #cls_init_E_binned = utils.compute_init_values(cls_init_E, pol="EE")

        #cls_init_B_binned = np.concatenate([np.zeros(2), cls_init_B])
        #cls_init_B_binned = utils.generate_init_values(cls_init_B, pol="BB")
        #cls_init_B_binned = utils.compute_init_values(cls_init_B, pol="BB")

        #scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])

        #cls_init_E_binned = np.ones(len(cls_init_E_binned)) * 3000
        #cls_init_B_binned = np.ones(len(cls_init_B_binned)) * 3000
        #cls_init_binned_E = np.random.normal(loc=cls_init_E_binned, scale=np.sqrt(10))
        #cls_init_binned_B = np.random.normal(loc=cls_init_B_binned, scale=np.sqrt(10))
        #cls_init_E_binned[:2] = 0
        #cls_init_B_binned[:2] = 0
        #starting_point = {"EE":cls_init_E_binned, "BB":cls_init_B_binned}
        #rescale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
        #starting_point = {"EE": cls_[1]*rescale, "BB": cls_[2]*rescale}
        #starting_point = config.starting_point
    else:
        starting_point = config.starting_point

    """
    start = time.time()
    start_cpu = time.clock()
    #h_cls_centered, h_accept_cr_centered, h_duration_cr, h_duration_cls_sampling = centered_gibbs.run(starting_point)
    h_cls_asis, h_accept_asis, _ = asis.run(starting_point)
    #h_cls_noncentered, h_accept_cr_noncentered, h_duration_cr, h_duration_cls_sampling = non_centered_gibbs.run(starting_point)
    #h_cls_asis, h_accept, h_accept_cr_asis, times_asis = asis_sampler.run(starting_point)
    #h_cls_asis_gibbs, h_accept, h_accept_cr_asis_gibbs,times_asis_gibbs = asis_sampler_gibbs.run(starting_point)
    end = time.time()
    end_cpu = time.clock()
    #h_cls_nonCentered, _, times = non_centered_gibbs.run(cls_init)
    total_time = end - start
    total_cpu_time = end_cpu - start_cpu
    print("Total time:", total_time)
    #print("Total Cpu time:",total_cpu_time)

    save_path = config.scratch_path + \
                "/data/polarization_runs/full_sky/asis/run_preliminary/asis_" + str(config.slurm_task_id) + ".npy"

    d = {"h_cls":h_cls_asis, "h_accept_cr":h_accept_asis, "h_duration_cls":None,
         "h_duration_cr":None, "bins_EE":config.bins["EE"], "bins_BB":config.bins["BB"],
         "blocks_EE":config.blocks["EE"],
         "blocks_BB":config.blocks["BB"], "proposal_variances_EE":config.proposal_variances_nc_polarized["EE"],
         "proposal_variances_BB":config.proposal_variances_nc_polarized["BB"], "total_cpu_time":total_cpu_time}

    np.save(save_path, d, allow_pickle=True)
    """
    """
    for _, pol in enumerate(["EE", "BB"]):
        #for l in range(2, config.L_MAX_SCALARS+1):
        for l in range(2, len(config.bins[pol][:-1])):
            y, xs, norm = utils.trace_likelihood_pol_binned(h_cls_noncentered[pol], pix_map, l,
                                                            maximum=np.max(h_cls_noncentered[pol][:, l]), pol=pol, all_sph=True)
            plt.plot(h_cls_noncentered[pol][:, l])
            plt.show()

            plt.hist(h_cls_noncentered[pol][100:, l], density=True, alpha=0.5, label="Gibbs NC", bins=300)
            #plt.hist(h_cls_centered[pol][100:, l], density=True, alpha=0.5, label="Gibbs Centered", bins=400)
            #plt.hist(h_cls_asis[pol][100:, l], density=True, alpha=0.5, label="ASIS", bins=400)
            print("Norm:", norm)
            plt.plot(xs, y/norm)
            plt.title(pol + " with l="+str(l))
            plt.legend(loc="upper right")
            plt.show()
    """

