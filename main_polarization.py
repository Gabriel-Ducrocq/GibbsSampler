import utils
import config
import numpy as np
import matplotlib.pyplot as plt
import time
import healpy as hp
from NonCenteredGibbs import NonCenteredGibbs
from ASIS import ASIS
from CenteredGibbs import CenteredGibbs


def generate_cls(polarization = True):
    """
    Function to generate cosmological parameters and the corresponding power spectrum.
    :param polarization: boolean. if True, we deal with "EE" and "BB" only. If False, with "TT" only.
    :return: arrays of floats, being cosmo_params, cls_tt, cls_ee, cls_bb and cls_te of polarization is True.
            Otherwise, cosmo_params, cls_tt.
    """
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR #+ np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    return theta_, cls_

def generate_dataset(cls_, polarization=True, mask_path = None):
    """
    Create observed sky map when we deal with "EE" and "BB" only.

    :param cls_: array of floats, power spectrum C_\ell. Size (3, L_max +1) if polarization, (L_max + 1,) otherwise.
    :param polarization: boolean. whether we are dealing with polarization or not.
    :param mask_path: string, path of the mas. If None, no mask is applied.
    :return: array of floats, size 6 of the parramters. Arrays of floats, size (L_max+1, ) if only "TT" or size (3, L_max + 1) if polarization.
            array of size (3, Npix) if polarization, otherwise size (Npix,). True skymap in pixel domain.
            array of size (3, Npix) if polarization, otherwise size (Npix,). True skymap in pixel domain.
    """
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.fwhm_radians, new=True) # Sample a sky map in pixel domain.
    d = map_true
    if polarization:
        d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp)) # add noise
        d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol)) # same
        d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol)) # same
        if mask_path is None:
            # If no mask is applied, output the Q and U maps.
            return map_true,  {"Q":d[1], "U":d[2]}
        else:
            #Otherwise apply the mask.
            mask = hp.ud_grade(hp.read_map(mask_path, 0), config.NSIDE)
            return map_true, {"Q": d[1]*mask, "U": d[2]*mask}

    else:
        #Same but with "TT" and intensity only.
        d += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        if mask_path is None:
            return theta_, cls_, map_true,  d
        else:
            mask = hp.ud_grade(hp.read_map(mask_path, 0), config.NSIDE)
            return theta_, cls_, map_true, d*mask


if __name__ == "__main__":
    np.random.seed()

    theta_, cls_ = generate_cls() # Sample cosmological parameters and power spectrum.
    cls_ = np.array([cls for cls in cls_])
    cls_[0] = np.zeros(len(cls_[0])) # Set the TT pow spec to 0
    cls_[3] = np.zeros(len(cls_[0])) # Same for TE
    s_true, pix_map = generate_dataset(cls_, polarization=True, mask_path=config.mask_path) # Generate an observed map.


    d = {"pix_map":pix_map, "params_":theta_, "skymap_true": s_true, "cls_":cls_, "fwhm_arcmin_beam":config.beam_fwhm,
         "noise_var_temp":config.noise_covar_temp, "noise_var_pol":config.noise_covar_pol, "mask_path":config.mask_path,
         "NSIDE":config.NSIDE, "lmax":config.L_MAX_SCALARS} #Save the map and its parameters.

    #np.save(config.scratch_path + "/data/polarization_runs/cut_sky/skymap_planck_mask/skymap.npy", d, allow_pickle=True) #Actual saving.
    #np.save(config.scratch_path + "/data/simon/cut-sky/skymap/skymap.npy", d, allow_pickle=True)
    np.save(config.scratch_path + "/data/polarization_runs/cut_sky/skymap_planck_mask/skymapTest.npy", d, allow_pickle=True)


    
    data_path = config.scratch_path + "/data/polarization_runs/cut_sky/skymap_planck_mask/skymapTest.npy" # Load the skymap.
    #data_path = config.scratch_path + "/data/simon/cut-sky/skymap/skymap.npy"
    d = np.load(data_path, allow_pickle=True) # Loading the skymap.
    d = d.item()
    pix_map = d["pix_map"] # Getting the map.

    #All the next line plot the SNR for "EE" and "BB".
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


    # Set the noise maps.
    noise_temp = np.ones(config.Npix) * config.noise_covar_temp
    noise_pol = np.ones(config.Npix) * config.noise_covar_pol

    centered_gibbs = CenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 100000,
                                   rj_step=False, gibbs_cr = True) # Create  centered Gibbs sampler with auxiliary variable step.

    ### ALL SPH ACTIVATED
    non_centered_gibbs = NonCenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 1000,
                                          proposal_variances=config.proposal_variances_nc_polarized, metropolis_blocks=config.blocks,
                                          all_sph=True) # Create a non centered Gibbs sampler with no mask and isotropic noise covariance matrix;


    asis = ASIS(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS, config.Npix,
                                    mask_path = config.mask_path, polarization = True, bins=config.bins, n_iter = 1000,
                                          proposal_variances=config.proposal_variances_nc_polarized, metropolis_blocks=config.blocks,
                                    rj_step = False, all_sph=False, gibbs_cr = True, n_gibbs = 20) # Create a ASIS sampler with auxiliary step.


    if config.preliminary_run:
        # If it is a preliminary run, we create a starting point D_\ell.
        _, _, cls_EE, cls_BB, cls_TE = generate_cls()
        scale = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)]) # factors l(l+1)/2\pi to go from C_\ell to D_\ell.
        dls_EE = scale*cls_EE
        dls_BB = scale*cls_BB
        all_dls = {"EE":dls_EE, "BB":dls_BB}

        starting_point = {"EE":[], "BB":[]}
        for pol in ["EE", "BB"]:
            for i, l_start in enumerate(config.bins[pol][:-1]):
                # We bin the power spectrum.
                l_end = config.bins[pol][i+1]
                starting_point[pol].append(np.mean(all_dls[pol][l_start:l_end]))

        starting_point["EE"] = np.array(starting_point["EE"])
        starting_point["BB"] = np.array(starting_point["BB"])
    else:
        # If it is not a preliminary run, we compute the starting point using the preliminary chains.
        starting_point = config.starting_point


    start = time.time()
    start_cpu = time.clock()
    #h_cls_centered, h_accept_cr, h_duration_cr, h_duration_cls_sampling = centered_gibbs.run(starting_point)
    h_cls_asis, h_accept_asis, h_accept_cr, h_it_duration, h_duration_cr, h_duration_centered, h_duration_nc = asis.run(starting_point) # Actual sampling.
    #h_cls_noncentered, h_accept_cr_noncentered, h_duration_cr, h_duration_cls_sampling = non_centered_gibbs.run(starting_point)
    end = time.time()
    end_cpu = time.clock()
    #h_cls_nonCentered, _, times = non_centered_gibbs.run(cls_init)
    total_time = end - start
    total_cpu_time = end_cpu - start_cpu
    print("Total time:", total_time)
    print("Total Cpu time:",total_cpu_time)

    save_path = config.scratch_path + \
                "/data/polarization_runs/cut_sky/planck_mask_runs/asis_gibbs_20_late/runs/asis_20_late" + str(config.slurm_task_id) + ".npy" # Save path

    d = {"h_cls":h_cls_asis, "h_accept_nc":h_accept_asis, "h_duration_cls_centered":None,
         "h_duration_cr":h_duration_cr, "bins_EE":config.bins["EE"], "bins_BB":config.bins["BB"],
         "blocks_EE":config.blocks["EE"], "h_duration_cls_non_centered":h_duration_nc, "h_duration_iteration":h_it_duration,
         "blocks_BB":config.blocks["BB"], "proposal_variances_EE":config.proposal_variances_nc_polarized["EE"],
         "proposal_variances_BB":config.proposal_variances_nc_polarized["BB"], "total_cpu_time":total_cpu_time,
         "pcg_accuracy": asis.constrained_sampler.pcg_accuracy, "h_accept_cr":h_accept_cr, "total_time":total_time,
         "rj_step": asis.rj_step, "gibbs_iterations":asis.constrained_sampler.n_gibbs,
         "gibbs_cr":asis.constrained_sampler.gibbs_cr
         } # All the information we save about the run.

    #np.save(save_path, d, allow_pickle=True) # Actual saving.


