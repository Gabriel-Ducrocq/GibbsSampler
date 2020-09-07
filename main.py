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


def generate_dataset(polarization=True, mask_path = None):
    theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR)
    cls_ = utils.generate_cls(theta_, polarization)
    map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.beam_fwhm, new=True)
    d = map_true
    if polarization:
        d[0] += np.random.normal(scale=np.sqrt(config.var_noise_temp))
        d[1] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        d[2] += np.random.normal(scale=np.sqrt(config.var_noise_pol))
        return theta_, cls_, map_true,  d
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
    cls_init = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
    cls_init = np.concatenate([np.zeros(2), cls_init])
    #cls_init_binned = utils.generate_init_values(cls_init)
    #scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    #cls_init_binned = np.ones(len(cls_init_binned))*3000
    #cls_init_binned = np.random.normal(loc=cls_init_binned, scale=np.sqrt(10))
    #cls_init_binned[:2] = 0

    theta_, cls_, s_true, pix_map = generate_dataset(polarization=False, mask_path=config.mask_path)

    #d = {"pix_map":pix_map, "theta":theta_, "cls_":cls_, "s_true":s_true, "beam_fwhm":config.beam_fwhm,
    #     "mask_path":config.mask_path, "noise_rms":np.sqrt(config.noise_covar_temp), "nside":config.NSIDE,
    #     "LMAX":config.L_MAX_SCALARS}

    #np.save(config.scratch_path + "/data/non_isotropic_runs/skymap/skymap.npy", d, allow_pickle=True)
    #print(pix_map)
    hp.mollview(pix_map)
    plt.show()

    #data_path = config.scratch_path + "/data/non_isotropic_runs/skymap/skymap.npy"
    #d = np.load(data_path, allow_pickle=True)
    #d = d.item()
    #pix_map = d["pix_map"]

    #range_l = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    #plt.plot(cls_[0]*range_l)
    #plt.show()

    snr = cls_ * (config.bl_gauss ** 2) / (config.noise_covar_temp * 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    plt.title("TT")
    plt.show()
    """
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
    """
    noise_temp = np.ones(config.Npix) * config.noise_covar_temp
    #non_centered_gibbs = NonCenteredGibbs(pix_map, noise_temp, None ,config.beam_fwhm,
    #                                      config.NSIDE, config.L_MAX_SCALARS,
    #                               config.Npix, proposal_variances=config.proposal_variances_asis, n_iter=10000,
    #                                      bins = config.bins, mask_path = config.mask_path)

    #centered_gibbs = CenteredGibbs(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
    #                               config.Npix, n_iter=100000, bins = config.bins,
    #                               mask_path = config.mask_path)

    asis_sampler = ASIS(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                            config.Npix, proposal_variances=config.proposal_variances_nc, n_iter=1000, bins = config.bins,
                        mask_path = config.mask_path, gibbs_cr=False, metropolis_blocks=config.blocks)

    asis_sampler_gibbs = ASIS(pix_map, noise_temp, None, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                            config.Npix, proposal_variances=config.proposal_variances_nc, n_iter=1000, bins = config.bins,
                        mask_path = config.mask_path, gibbs_cr=True, metropolis_blocks=config.blocks)
    """
    dls_ = np.array([cl*l*(l+1)/(2*np.pi) for l, cl in enumerate(cls_)])
    var_cls_ = utils.generate_var_cl(dls_)
    inv_var_cls_ = np.zeros(len(var_cls_))
    np.reciprocal(var_cls_, out=inv_var_cls_, where=config.mask_inversion)
    h_centered_1 = []
    for i in range(10000):
        if i % 100 == 0:
            print("centered 1:", i)

        s1, var_cls_full, binned_dls = centered_gibbs.run(dls_[:])
        #inv_var_cls_full = np.zeros(len(var_cls_full))
        #inv_var_cls_full[np.where(var_cls_full!= 0)] = 1/var_cls_full[np.where(var_cls_full!=0)]
        #s_nonCentered = np.sqrt(inv_var_cls_full)*s1
        #binned_dls_new, _, _ = non_centered_gibbs.cls_sampler.sample(s_nonCentered, binned_dls, var_cls_full)
        h_centered_1.append(s1)

    h_asis = []
    h_asis_centered = []
    print("True CLS")
    print(cls_)
    for i in range(10000):
        if i % 100 == 0:
            print("asis:", i)

        skymap = asis_sampler.run(dls_[:])
        h_asis.append(skymap)


    #print(np.mean(np.array(h_centered_1)[:, 15]))
    #print(np.mean(np.array(h_asis)[:, 15]))
    plt.hist(np.array(h_centered_1)[:, 15], label="centered1", density=True, alpha=0.5, bins = 100)
    plt.hist(np.array(h_asis)[:, 15], label="asis", density=True, alpha=0.5, bins=100)
    plt.legend(loc="upper right")
    plt.show()
    """

    #polarized_centered = CenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
    #                               config.Npix, n_iter=10000, polarization=True)


    #polarized_non_centered_gibbs = NonCenteredGibbs(pix_map, config.noise_covar_temp, config.noise_covar_pol,
    #                                                config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
    #                               config.Npix, proposal_variances=config.proposal_variances_nc_polarized, n_iter=10000, polarization=True)


    #h_cls_nc, _ = non_centered_gibbs.run(cls_init)
    l_interest =3

    np.random.seed()
    cls_init = np.array([1e3 / (l ** 2) for l in range(2, config.L_MAX_SCALARS + 1)])
    cls_init = np.concatenate([np.zeros(2), cls_init])
    cls_init_binned = utils.generate_init_values(cls_init)
    scale = np.array([l*(l+1)/(2*np.pi) for l in range(config.L_MAX_SCALARS+1)])
    cls_init_binned = np.ones(len(cls_init_binned))*3000
    cls_init_binned = np.random.normal(loc=cls_init_binned, scale=np.sqrt(10))
    cls_init_binned[:2] = 0
    ###Checker que ça marche avec bruit différent de 100**2
    #h_old_centered, _ = default_gibbs(pix_map, cls_init)
    cls_init = cls_init[:5]
    start = time.time()
    #start_cpu = time.clock()
    #h_cls_centered, h_accept_cr_centered, _ = centered_gibbs.run(cls_init_binned)
    #h_cls_asis, h_accept, h_accept_cr_asis, times_asis = asis_sampler.run(cls_init_binned)
    h_cls_asis_gibbs, h_accept, h_accept_cr_asis_gibbs,times_asis_gibbs = asis_sampler_gibbs.run(cls_init_binned)
    end = time.time()
    #end_cpu = time.clock()
    #h_cls_nonCentered, _, times = non_centered_gibbs.run(cls_init)
    total_time = end - start
    #total_cpu_time = end_cpu - start_cpu
    print("Total time:", total_time)
    #print("Total Cpu time:",total_cpu_time)

    #save_path = config.scratch_path + \
    #            "/data/non_isotropic_runs/asis/preliminary_run/asis_" + str(config.slurm_task_id) + ".npy"

    d = {"h_cls":h_cls_asis_gibbs, "bins":config.bins, "metropolis_blocks":config.blocks, "h_accept":h_accept,
         "h_times_iteration":times_asis_gibbs, "total_time":total_time}#,"total_cpu_time":total_cpu_time ,"data_path":data_path}

    #np.save(save_path, d, allow_pickle=True)
    np.save("test_gibbs_non_change_variable.npy", d, allow_pickle=True)
    #d = np.load("test_pcg.npy", allow_pickle = True)
    #d = d.item()
    #h_cls_asis = d["h_cls_non_centered"]
    #pix_map = d["pix_map"]
    #d = {"h_cls":h_cls_asis_gibbs, "pix_map":pix_map, "cls_":cls_}
    #np.save("test_gibbs.npy", d, allow_pickle=True)
    #d2 = {"h_cls_asis":h_cls_asis, "pix_map":pix_map, "cls_":cls_}
    #np.save("test_pcg.npy", d2, allow_pickle=True)
    #print("Time per iteration ASIS:", np.median(times_asis))
    #print("Time per iteration ASIS GIBBS:", np.median(times_asis_gibbs))
    #print(np.sum((h_cls_asis[1:, :] - h_cls_asis[:-1, :])**2, axis = 1).shape)
    #print(np.sum((h_cls_asis_gibbs[1:, :] - h_cls_asis_gibbs[:-1, :]) ** 2, axis=1).shape)
    #esjd_asis = np.median(np.sum((h_cls_asis[1:, :] - h_cls_asis[:-1, :])**2, axis = 1))
    #esjd_asis_gibbs = np.median(np.sum((h_cls_asis_gibbs[1:, :] - h_cls_asis_gibbs[:-1, :]) ** 2, axis=1))
    #esjd_asis_per_sec = esjd_asis/np.median(times_asis)
    #esjd_asis_gibbs_per_sec = esjd_asis_gibbs / np.median(times_asis_gibbs)
    #print("Mean ESJD ASIS:", esjd_asis)
    #print("Mean ESJD ASIS, GIBBS:", esjd_asis_gibbs)
    #print("ESJD per sec ASIS:", esjd_asis_per_sec)
    #print("ESJD per sec ASIS GIBBS", esjd_asis_gibbs_per_sec)
    #print("Acceptance rate cr centered:", np.mean(h_accept_cr_centered))
    #print("Acceptance rate cr asis:", np.mean(h_accept_cr_asis))
    #print("Iteration time:", np.mean(times))
    #plt.plot(h_cls_asis[:, l_interest], alpha=0.5, label="ASIS")
    #plt.plot(h_cls_asis_gibbs[:, l_interest], alpha=0.5, label="ASIS GIBBS")
    #plt.legend(loc="upper right")
    #plt.show()

    #d = np.load("test_gibbs.npy", allow_pickle=True)
    #d2 = np.load("test_pcg.npy", allow_pickle=True)
    #d = d.item()
    #d2 = d2.item()
    #h_cls_asis = d["h_cls"]
    #h_cls_gibbs = d2["h_cls_asis"]
    #pix_map = d["pix_map"]
    #cls_true = d["cls_"]

    #print("SHAPE")
    #print(h_cls_asis.shape)
    #esjd_rjpo = np.mean((h_cls_asis[1:, :] - h_cls_asis[:-1, :])**2, axis = 0)
    #esjd_gibbs = np.mean((h_cls_gibbs[1:, :] - h_cls_gibbs[:-1, :]) ** 2, axis = 0)
    #print("ESJDS")
    #print("esjd rjpo:",esjd_rjpo)
    #print("esjd gibbs:", esjd_gibbs)
    #print(esjd_rjpo/esjd_gibbs)
    #plt.plot(esjd_rjpo/esjd_gibbs)
    #plt.show()

    #for l_interest in range(2, config.L_MAX_SCALARS+1):
    #    utils.plot_autocorr_multiple([h_cls_asis[None, :, :], h_cls_gibbs[None, :, :]], ["asis", "asis gibbs"], l_interest, 100, cls_true)

    for l_interest in range(2, config.L_MAX_SCALARS+1):
        yy, xs, norm = utils.trace_likelihood_binned(h_cls_asis_gibbs[:, l_interest] ,pix_map, l_interest, np.max(h_cls_asis_gibbs[:, l_interest]))

        print("NORM:", norm)
        #plt.hist(h_cls_asis[:, l_interest], density=True, alpha=0.5, bins = 250, label="ASIS")
        #plt.hist(h_cls_asis[:, l_interest], density=True, alpha=0.5, bins=200, label="ASIS")
        plt.hist(h_cls_asis_gibbs[:, l_interest], density=True, alpha=0.5, bins=200, label="ASIS GIBBS")
        #plt.hist(h_cls_asis_gibbs[:, l_interest], density=True, alpha=0.5, bins=200, label="ASIS GIBBS")
        #plt.hist(h_cls_nonCentered[:, l_interest], density=True, alpha=0.5, bins=100, label="Non Centered")
        plt.legend(loc="upper right")
        plt.title("l interest =" + str(l_interest))
        if norm > 0:
            plt.plot(xs, yy/norm)
        else:
            plt.plot(xs, yy)#/norm)

        plt.show()
        #plt.plot(h_cls_asis[:, l_interest], alpha = 0.5, label="ASIS")
        plt.plot(h_cls_asis_gibbs[:, l_interest], alpha = 0.5, label="ASIS Gibbs")
        plt.legend(loc="upper right")
        plt.show()
        #plt.plot("test_pcg"+str(l_interest)+".png")
        #plt.close()




    #h_cls_pol, _ = polarized_non_centered_gibbs.run(init_cls)




    #pix_map_alm = hp.map2alm(pix_map, lmax=config.L_MAX_SCALARS)
    #map_alm_B = pix_map_alm[2]
    #map_alm_T = pix_map_alm[0]
    #all_pow_spec_B = hp.alm2cl(map_alm_B, lmax=config.L_MAX_SCALARS)
    #all_pow_spec_T = hp.alm2cl(map_alm_T, lmax=config.L_MAX_SCALARS)
    #all_pow_spec_TT, all_pow_spec_EE, all_pow_spec_BB, all_pow_spec_TE, _, _ = hp.alm2cl(pix_map_alm, lmax=config.L_MAX_SCALARS)

    #l_interest = 7
    #i = 0
    #j = 0

    #alpha = (2*l_interest-1)/2
    #beta = ((2*l_interest+1)/2)*((l_interest*(l_interest+1))/(2*np.pi))*all_pow_spec_B[l_interest]*(1/config.bl_gauss[l_interest]**2)
    #loc = - (config.noise_covar_pol*4*np.pi/config.Npix)*(l_interest*(l_interest+1)/(2*np.pi))*(1/config.bl_gauss[l_interest]**2)
    #yy = []
    #xx = []
    #opposite_norm = scipy.stats.invgamma.cdf(0, a = alpha, loc=loc, scale=beta)
    #norm = 1 - opposite_norm
    #for x in np.linspace(0, 0.5, 10000):
    #    xx.append(x)
    #    y = scipy.stats.invgamma.pdf(x, a=alpha, scale=beta, loc = - (config.noise_covar_pol*4*np.pi/config.Npix)\
    #                                                               *(l_interest*(l_interest+1)/(2*np.pi))*(1/config.bl_gauss[l_interest]**2))
    #    yy.append(y/norm)



    #alpha = (2*l_interest-3)/2
    #beta = ((2*l_interest+1)/2)*((l_interest*(l_interest+1))/(2*np.pi))*all_pow_spec_TE[l_interest]*(1/config.bl_gauss[l_interest]**2)
    #loc = - (config.noise_covar_pol*4*np.pi/config.Npix)*(l_interest*(l_interest+1)/(2*np.pi))*(1/config.bl_gauss[l_interest]**2)
    #loc = 0
    #yy = []
    #xx = []
    #opposite_norm = scipy.stats.invgamma.cdf(0, a = alpha, loc=loc, scale=beta)
    #norm = 1 - opposite_norm
    #for x in np.linspace(-3, 16, 10000):
    #    xx.append(x)
    #    y = scipy.stats.invgamma.pdf(x, a=alpha, scale=beta, loc = loc)
    #    #y = scipy.stats.invwishart.pdf(x, df = 2*l_interest-3, scale = np.array([[beta]]))
    #    yy.append(y)

    """
    scale_mat = np.zeros((2, 2))
    scale_mat[0, 0] = all_pow_spec_TT[l_interest]
    scale_mat[1, 1] = all_pow_spec_EE[l_interest]
    scale_mat[1, 0] = all_pow_spec_TE[l_interest]
    scale_mat[0, 1] = all_pow_spec_TE[l_interest]
    scale_mat *= (2*l_interest+1)
    h_true = []
    for m in range(100000):
        if m % 1000 == 0:
            print("True sampling iteration:", i)

        sample = scipy.stats.invwishart.rvs(df=2*l_interest-2, scale = scale_mat)
        if sample[0, 0] > config.noise_covar_temp*config.w and sample[1, 1] > config.noise_covar_pol*config.w and (sample[0, 0] - config.noise_covar_temp*config.w)*(sample[1, 1]-config.noise_covar_pol*config.w) - sample[1, 0]**2 > 0:
            h_true.append((sample[i, j]-config.w*config.noise_covar_temp)*(l_interest*(l_interest+1)/(2*np.pi))*(1/config.bl_map[l_interest]**2))

    """
    """
    alpha = (2*l_interest-3)/2
    beta = ((2*l_interest+1)/2)*all_pow_spec_T[l_interest]/config.bl_map[0]**2
    yy = []
    xx = []
    for x in np.linspace(0, 1000, 10000):
        xx.append(x)
        y = scipy.stats.invgamma.pdf(x, a=alpha, scale=beta)
        #y = (y - config.noise_covar_temp*config.w)*l_interest*(l_interest+1)/(2*np.pi)
        yy.append(y)
    """
    """
    print("Len h_true")
    print(len(h_true))
    print(len(h_cls))
    alms = hp.map2alm(pix_map, lmax=config.L_MAX_SCALARS)
    cls_hat_TT, cls_hat_EE, cls_hat_BB,  cls_hat_TE, _, _ = hp.alm2cl(alms, lmax=config.L_MAX_SCALARS)
    plt.plot(h_cls[:, l_interest, i, j])
    plt.axhline(y=init_cls[l_interest, i, j])
    plt.axhline(y=cls_hat_TT[l_interest]*l_interest*(l_interest+1)/(2*np.pi), color="red")
    plt.show()

    plt.hist(h_cls[:, l_interest, i, j], density=True, bins = 70, alpha=0.5, label="Gibbs")
    plt.hist(h_true, density=True, bins = 100, label="True", alpha=0.5)
    #plt.plot(xx, yy)
    plt.legend(loc="upper right")
    plt.axvline(x=init_cls[l_interest, i, j])
    plt.axvline(x=cls_hat_TT[l_interest]*l_interest*(l_interest+1)/(2*np.pi), color="red")
    plt.show()

    print(yy)
    """
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
