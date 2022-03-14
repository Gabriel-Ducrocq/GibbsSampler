import numpy as np
import config
#from classy import Class
import scipy
from scipy.sparse import linalg
from scipy import stats
import matplotlib.pyplot as plt
import time
import healpy as hp
from classy import Class
from variance_expension import generate_var_cl_cython, synthesis_hp as synthesis_cython
from statsmodels.tsa.stattools import acovf
from numba import njit, prange
import qcinv

cosmo = Class()

def generate_theta():
    return np.random.normal(config.COSMO_PARAMS_MEAN, config.COSMO_PARAMS_SIGMA)


def generate_cls(theta, pol = False):
    params = {'output': config.OUTPUT_CLASS,
              "modes":"s,t",
              "r":0.001,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': config.LENSING}
    d = {name:val for name, val in zip(config.COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls_tt = cls["tt"]*2.7255e6**2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"]*2.7255e6**2
        cls_bb = cls["bb"]*2.7255e6**2
        cls_te = cls["te"]*2.7255e6**2
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt, cls_ee, cls_bb, cls_te


def likelihood(x, sigma, l):
    return np.exp(-(((2*l+1)/2)*sigma)/(x+config.noise_covar))/(x+config.noise_covar)**((2*l+1)/2)


def likelihood_normalized(x, sigma, l, norm_before):
    return np.exp(-(((2*l+1)/2)*sigma)/(x+config.noise_covar))/((x+config.noise_covar)**((2*l+1)/2)*norm_before)



def trace_likelihood_beam(h_cl, d, l):
    d_sph = analysis_hp(d)
    d_sph = real_to_complex(d_sph)
    print("d_sph:")
    print(d_sph)
    power_spec = hp.alm2cl(d_sph, lmax=config.L_MAX_SCALARS)
    observed_cl = power_spec[l]/config.bl_gauss[l]**2
    norm, err = scipy.integrate.quad(stats.invgamma.pdf, a=0, b=np.inf, args=((2*l-1)/2,
                -(4*np.pi/config.Npix)*config.noise_covar_temp/config.bl_gauss[l]**2,(2*l+1)*observed_cl/2))

    y = []
    xs = np.linspace(0, np.max(h_cl),10000)
    print("MAX RANGE")
    print(np.max(h_cl))
    for x in xs:
        res = stats.invgamma.pdf(x, a=(2*l-1)/2,
                    loc=-(4*np.pi/config.Npix)*config.noise_covar_temp/config.bl_gauss[l]**2, scale=(2*l+1)*observed_cl/2)
        y.append(res)

    return np.array(y), xs, norm


def cut_cls_high(cls_high, l, next_l):
    return cls_high[l-config.l_cut:next_l-config.l_cut]


def compute_observed_spectrum(d):
    observed_cls = []
    for l in range(2, config.L_MAX_SCALARS+1):
        piece_d = d[l**2-4:(l+1)**2-4]
        observed_cl = (np.abs(piece_d[0]) ** 2 + 2 * np.sum(piece_d[1:] ** 2)) / (2 * l + 1)
        observed_cls.append(observed_cl)

    return observed_cls


def plot_histogram(chains_rescale, chains_PNCP, chains_ASIS, chain_direct, d, l_interest, cls):
    rescale = chains_rescale[:, :, l_interest-2].flatten()
    pncp = chains_PNCP[:, :, l_interest-2].flatten()
    asis = chains_ASIS[:, :, l_interest-2].flatten()
    direct = chain_direct[:, :, l_interest-2].flatten()

    n_bins = 50
    d_cut = d[l_interest**2 - 4: (l_interest+1)**2 - 4]
    norm, y, xs = trace_likelihood(asis, d_cut, l_interest)
    plt.plot(xs, y/norm)
    plt.hist(rescale, label="Rescale", density=True, alpha=0.5, bins=n_bins)
    plt.hist(pncp, label="PNCP", density=True, alpha=0.5, bins=n_bins)
    plt.hist(asis, label="ASIS", density=True, alpha=0.5, bins=n_bins)
    plt.hist(direct, label="Direct", density=True, alpha=0.5, bins=n_bins)
    plt.legend(loc="upper right")
    plt.title("Trace plot SNR " + str(cls[l_interest-2]/config.noise_covar))
    plt.savefig("histogram"+"l="+str(l_interest)+"_thinned_"+str(10)+".png")
    plt.close()



def trace_plot(chains_rescale, chains_PNCP, chains_ASIS, chains_direct, chains_exchange, l_interest, n_chain, cls):
    plt.plot(chains_rescale[n_chain, :, l_interest-2], label="Rescale", alpha=0.5)
    plt.plot(chains_PNCP[n_chain, :, l_interest-2], label="PNCP", alpha=0.5)
    plt.plot(chains_ASIS[n_chain, :, l_interest-2], label="ASIS", alpha=0.5)
    plt.plot(chains_direct[n_chain, :, l_interest-2], label="Direct", alpha=0.5)
    plt.plot(chains_exchange[n_chain, :, l_interest - 2], label="Exchange", alpha=0.5)
    plt.legend(loc="upper right")
    plt.title("Example run SNR:" + str(cls[l_interest-2]/config.noise_covar))
    plt.savefig("trace_plots_"+"l="+str(l_interest)+"_thinned_10"+".png")
    plt.close()


def compute_autocov(chain, nlag):
    autocov = acovf(chain, demean=False, fft=True, nlag=nlag)
    return autocov


def compute_autocorr_multiple(chains, l_interest, nlag):
    chains = chains[:, :, l_interest]
    av = np.mean(chains)
    chains = chains - av
    var_chain = np.var(chains)
    total_autocov = 0
    for i, chain in enumerate(chains):
        total_autocov += compute_autocov(chain, nlag)

    total_autocov /= chains.shape[0]

    return total_autocov/var_chain


def plot_autocorr_multiple(list_chains_algo, name_chains_algo, l_interest, nlag, cls):
    indexes = [i for i in range(nlag+1)]
    indexes_asis_block = [i for i in range(21)]
    n_algo = len(list_chains_algo)
    if n_algo%2 == 0:
        n_rows = n_algo/2
        n_cols = 2
    else:
        n_rows = np.ceil(n_algo/2)
        n_cols = 2

    for i,name in enumerate(name_chains_algo):
        plt.subplot(n_rows, n_cols, i+1)
        chains = list_chains_algo[i]
        if name == "ASIS_block":
            autocorr = compute_autocorr_multiple(chains, l_interest, 20)
            plt.bar(indexes_asis_block, autocorr)
        else:
            autocorr = compute_autocorr_multiple(chains, l_interest, nlag)
            print(name + "autocorr length:", np.sum(autocorr))
            plt.bar(indexes, autocorr)

        plt.title(name)

    plt.suptitle("Autocorrelations "+ "l="+str(l_interest) + " and " + "SNR=" + str(cls[l_interest]/config.noise_covar)+" thinned 10")
    plt.show()
    plt.savefig("autocorrelations_"+"l="+str(l_interest)+"_thinned_10.png")
    plt.close()



def plot_steps_vs_variance(list_chains, name_chains_algo, variances):
    n_algo = len(list_chains)
    if n_algo%2 == 0:
        n_rows = n_algo/2
        n_cols = 2
    else:
        n_rows = np.ceil(n_algo/2)
        n_cols = 2

    for i,name in enumerate(name_chains_algo):
        plt.subplot(n_rows, n_cols, i + 1)
        chains = list_chains[i]
        steps = (chains[:, 1:, :] - chains[:, :-1, :])**2
        mean_steps = np.mean(steps, axis=(0, 1))
        plt.plot(variances, mean_steps, label=name)
        plt.title(name)

    plt.legend(loc="upper right")
    plt.savefig("sizes_steps.png")
    plt.close()


def plot_steps_vs_variance_high_l(list_chains, name_chains_algo, variances):
    n_algo = len(list_chains)
    if n_algo%2 == 0:
        n_rows = n_algo/2
        n_cols = 2
    else:
        n_rows = np.ceil(n_algo/2)
        n_cols = 2

    for i,name in enumerate(name_chains_algo):
        plt.subplot(n_rows, n_cols, i + 1)
        chains = list_chains[i]
        steps = (chains[:, 1:, :] - chains[:, :-1, :])**2
        mean_steps = np.mean(steps, axis=(0, 1))
        plt.scatter(variances[config.l_cut-2:], np.log(mean_steps[config.l_cut-2:]), s=10)
        plt.ylim(-0.005, 0.005)
        plt.title(name)

    plt.savefig("high_l_sizes_steps.png")
    plt.close()


def complex_to_real_libsharp(alms):
    temp = [[np.sqrt(2)*np.real(alm_h), -np.sqrt(2)*np.imag(alm_h)] if i > config.L_MAX_SCALARS else
            [np.real(alm_h)] for i, alm_h in enumerate(alms)]
    return np.array([a for alm in temp for a in alm])


def real_to_complex(alms):
    m_0 = alms[:config.L_MAX_SCALARS+1] + np.zeros(config.L_MAX_SCALARS+1)*1j
    m_pos = alms[config.L_MAX_SCALARS+1:]
    m_pos = (m_pos[::2] + 1j*m_pos[1::2])/np.sqrt(2)
    return np.concatenate([m_0, m_pos])


def complex_to_real(alms):
    m_0 = alms[:config.L_MAX_SCALARS+1].real
    m_pos = alms[config.L_MAX_SCALARS + 1:]
    m_pos_real = m_pos.real*np.sqrt(2)
    m_pos_img = m_pos.imag*np.sqrt(2)
    m_pos = np.array([a for parts in zip(m_pos_real, m_pos_img) for a in parts])
    return np.concatenate([m_0, m_pos])



def real_to_real_libsharp(alms):
    #The alms are assumed to be organized as follows: first the L + 1 alms corresponding to m = 0
    #Then the real parts of alm for m > 0 and then the imaginary parts
    m_0 = alms[:config.L_MAX_SCALARS+1]
    m_pos = alms[config.L_MAX_SCALARS+1:]
    m_pos[::2] *= np.sqrt(2)
    m_pos[1::2] *= -np.sqrt(2)
    alm_libsharp = np.concatenate([m_0, m_pos])
    return np.array(alm_libsharp)


def real_libsharp_to_real(alms_libsharp):
    m_0 = alms_libsharp[:config.L_MAX_SCALARS+1]
    m_pos = alms_libsharp[config.L_MAX_SCALARS+1:]
    m_pos[::2] /= np.sqrt(2)
    m_pos[1::2] /= -np.sqrt(2)
    alms = np.concatenate([m_0, m_pos])
    return alms




def synthesis_hp2(alms):
    alms = alms*config.rescaling_alm2map
    alms = real_to_complex(alms)
    s = hp.alm2map(alms, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
    return s


def synthesis_hp(alms):
    synth = synthesis_cython(alms, config.NSIDE)
    return np.asarray(synth)

def set_zero_monopole_dipole_contribution(alms):
    alms[[0, 1, config.L_MAX_SCALARS+1, config.L_MAX_SCALARS+1 + int(config.L_MAX_SCALARS*(config.L_MAX_SCALARS+1)/2)]] = 0
    return alms


def remove_monopole_dipole_contributions(alms):
    #Setting a00, a10 and Re(a11) and Imag(a11) to zero
    alms[[0, 1, config.L_MAX_SCALARS+1, config.L_MAX_SCALARS+2]] = 0.0
    return alms

def adjoint_synthesis_hp(map, bl_map=None):
    if len(map) == 3:
        alms = hp.map2alm(map, lmax=config.L_MAX_SCALARS, iter=3, pol=True)
        alms_T, alms_E, alms_B = alms
        alms_T = complex_to_real(alms_T)*config.rescaling_map2alm
        alms_E = complex_to_real(alms_E)*config.rescaling_map2alm
        alms_B = complex_to_real(alms_B)*config.rescaling_map2alm
        if bl_map is not None:
            alms_T *= bl_map
            alms_E *= bl_map
            alms_B *= bl_map

        return alms_T, alms_E, alms_B

    else:
        #I am not removing monopole and dipole anymore !
        alms = hp.map2alm(map, lmax=config.L_MAX_SCALARS, iter=3)
        #alms = remove_monopole_dipole_contributions(complex_to_real(alms))
        alms = complex_to_real(alms)
        alms *= config.rescaling_map2alm
        if bl_map is not None:
            alms *= bl_map

        return alms

def analysis_hp(map):
    return remove_monopole_dipole_contributions(
        complex_to_real(hp.map2alm(map, lmax=config.L_MAX_SCALARS)))

def generate_var_cl(cls_):
    var_cl_full = generate_var_cl_cython(cls_)
    return np.asarray(var_cl_full)

def trace_normal(inv_var_cls, h_s, l, d):
    h_min = np.min(h_s[:, l])
    h_max = np.max(h_s[:, l])
    loc = adjoint_synthesis((1 / config.var_noise) * d)[l]/(inv_var_cls[l] + config.Npix/(4*np.pi*config.noise_covar))
    scale = 1/np.sqrt(inv_var_cls[l]+ (config.Npix/(4*np.pi*config.noise_covar)))

    xs = np.arange(h_min, h_max, 0.01)
    ys = []
    for x in xs:
        y = stats.norm.pdf(x, loc=loc, scale=scale)
        ys.append(y)

    return np.array(ys), xs


def trace_normal_to_test(h_s, l, inv_var_cls):
    h_min = np.min(h_s[:, l])
    h_max = np.max(h_s[:, l])

    print("MIN")
    print(h_min)
    print("MAX")
    print(h_max)
    scale = np.sqrt(config.Npix/(4*np.pi*config.noise_covar) + inv_var_cls[l])
    #scale = np.sqrt(config.Npix/(4*np.pi))

    xs = np.arange(h_min, h_max, 0.01)
    ys = []
    for x in xs:
        y = stats.norm.pdf(x, loc=0, scale=scale)
        ys.append(y)

    return np.array(ys), xs



def compute_SNR(cls_):
    #plt.plot(cls_/(config.noise_covar*4*np.pi/(config.Npix*(config.bl_gauss**2))), label="Cls")
    snr = cls_*(config.bl_gauss**2)/(config.noise_covar*4*np.pi/config.Npix)
    plt.plot(snr)
    plt.axhline(y=1)
    #plt.plot(nls_, label="Nls")

    plt.show()



def set_var_prop(d):
    d_sph = analysis_hp(d)
    d_sph = real_to_complex(d_sph)
    power_spec = hp.alm2cl(d_sph, lmax=config.L_MAX_SCALARS)
    observed_cl = power_spec[2:]/ config.bl_gauss[2:] ** 2
    alphas = np.array([(2*l-1)/2 for l in range(2, config.L_MAX_SCALARS+1)])
    scale_beta = np.array([(2*l+1)/2 for l in range(2, config.L_MAX_SCALARS+1)])
    gauss_mean = (scale_beta*observed_cl)/(alphas-1) -(4*np.pi/config.Npix)*config.noise_covar/config.bl_gauss[2:]**2
    gauss_stdd = np.sqrt(np.abs(((scale_beta*observed_cl)**2) / ((alphas - 1) ** 2 * (alphas - 2))))
    alpha_vars = -gauss_mean/gauss_stdd
    Zs = 1 - scipy.stats.norm.cdf(alpha_vars)
    posterior_vars = gauss_stdd**2 * (1 + (alpha_vars*scipy.stats.norm.pdf(alpha_vars)/Zs) - (scipy.stats.norm.pdf(alpha_vars)/Zs)**2)
    scale = np.array([1 if l < 540 else 0.1 for l in range(2, config.L_MAX_SCALARS+1)])
    rescale_post_var = posterior_vars
    rescale_post_var[:30] = gauss_stdd[:30]**2
    #config.proposal_variances_nc = posterior_vars*scale
    config.proposal_variances_nc = posterior_vars * 0.5
    config.proposal_variances_nc = config.proposal_variances_nc[:-1]
    config.proposal_variances_asis = posterior_vars*scale
    config.proposal_variances_pncp = posterior_vars * scale
    config.proposal_variances_rescale = rescale_post_var*0.0000001
    config.proposal_variances_exchange = posterior_vars*0.5



def unfold_bins(binned_cls_, bins):
    n_per_bin = bins[1:] - bins[:-1]
    return np.repeat(binned_cls_, n_per_bin)


def compute_binned_lik(x, alphas, betas, locs):
    return np.prod(stats.invgamma.pdf(x, a=alphas,
                    loc=locs, scale = betas))


def compute_init_values(cls, pol = None):
    vals = []
    if pol is None:
        for i, l_start in enumerate(config.bins[:-1]):
            l_end = config.bins[i+1]
            vals.append(np.mean(cls[l_start:l_end]))

        print("LEN VALS")
        print(len(vals))
        return np.array(vals)
    else:
        for i, l_start in enumerate(config.bins[pol][:-1]):
            l_end = config.bins[pol][i + 1]
            vals.append(np.mean(cls[l_start:l_end]))

        print("LEN VALS")
        print(len(vals))
        return np.array(vals)


def generate_init_values(cls):
    cls_binned = compute_init_values(cls)
    var = cls_binned[2:]/10000
    clip_low = -cls_binned[2:] / np.sqrt(var)
    print(var)
    print("\n")
    print(cls_binned[2:])
    cls_init = scipy.stats.truncnorm.rvs(a=clip_low, b=np.inf, loc=cls_binned[2:], scale=np.sqrt(var))
    return np.concatenate([np.zeros(2), cls_init])




def compute_binned_likelihood(x, alphas, betas, locs, denoms):
    return np.prod(stats.invgamma.pdf(x, a=alphas,
                                      loc=locs, scale=betas) / denoms)


def trace_likelihood_binned(h_cl, d, l, maximum, dl=True):
    l_start, l_end = config.bins[l], config.bins[l + 1]
    d_sph = analysis_hp(d)
    d_sph = real_to_complex(d_sph)
    scale_dl = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)])
    if not dl:
        scale_dl = np.ones(len(scale_dl))

    power_spec = hp.anafast(d, lmax=config.L_MAX_SCALARS)
    observed_cls = power_spec * scale_dl / config.bl_gauss ** 2

    scale_betas = [(2 * l + 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    alphas = [(2 * l - 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    betas = observed_cls * scale_betas
    locs = -((4 * np.pi / config.Npix) * config.noise_covar_temp / config.bl_gauss ** 2) * scale_dl
    denoms_factors = [(config.bl_gauss[l] ** 2) for l in range(config.L_MAX_SCALARS + 1)]
    if not dl:
        denoms_factors = np.ones(len(denoms_factors))

    norm, err = scipy.integrate.quad(compute_binned_likelihood, a=0, b=np.inf,
                                     args=(alphas[l_start:l_end], betas[l_start:l_end],
                                           locs[l_start:l_end], denoms_factors[l_start:l_end]))

    y = []
    maxi = np.max(h_cl)
    maxi = maximum
    steps = maxi / 100000
    xs = np.arange(0, maxi, steps)
    # xs = np.arange(0, 5 ,0.01)
    print("MAX RANGE")
    print(np.max(h_cl))
    for x in xs:
        res = compute_binned_likelihood(x, alphas[l_start:l_end], betas[l_start:l_end], locs[l_start:l_end],
                                        denoms_factors[l_start:l_end])
        y.append(res)

    return np.array(y), xs, norm



def trace_likelihood_pol_binned_bis(h_cl, d_all, l, maximum, pol = "EE", dl=True):
    l_start, l_end = config.bins[pol][l], config.bins[pol][l + 1]
    scale_dl = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)])
    if not dl:
        scale_dl = np.ones(len(scale_dl))

    #_, power_spec_EE, power_spec_BB, _, _, _ = hp.anafast([np.zeros(config.Npix), d_all["Q"], d_all["U"]], lmax=config.L_MAX_SCALARS, pol=True)
    if pol == "EE":
        #power_spec = power_spec_EE
        power_spec = hp.alm2cl(real_to_complex(d_all["EE"]), lmax=config.L_MAX_SCALARS)
    else:
        #power_spec = power_spec_BB
        power_spec = hp.alm2cl(real_to_complex(d_all["BB"]), lmax=config.L_MAX_SCALARS)

    observed_cls = power_spec * scale_dl / config.bl_gauss ** 2

    scale_betas = [(2 * l + 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    alphas = [(2 * l - 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    betas = observed_cls * scale_betas
    locs = -((4 * np.pi / config.Npix) * config.noise_covar_pol / config.bl_gauss ** 2) * scale_dl
    denoms_factors = [(config.bl_gauss[l] ** 2) for l in range(config.L_MAX_SCALARS + 1)]
    if not dl:
        denoms_factors = np.ones(len(denoms_factors))

    norm, err = scipy.integrate.quad(compute_binned_likelihood, a=0, b=np.inf,
                                     args=(alphas[l_start:l_end], betas[l_start:l_end],
                                           locs[l_start:l_end], denoms_factors[l_start:l_end]))

    y = []
    maxi = np.max(h_cl)
    maxi = maximum
    steps = maxi / 10000
    print("MAXI:", maxi)
    xs = np.arange(0, maxi, steps)
    # xs = np.arange(0, 5 ,0.01)
    print("MAX RANGE")
    print(np.max(h_cl))
    for x in xs:
        res = compute_binned_likelihood(x, alphas[l_start:l_end], betas[l_start:l_end], locs[l_start:l_end],
                                        denoms_factors[l_start:l_end])
        y.append(res)

    return np.array(y), xs, norm


def trace_likelihood_pol_binned(h_cl, d_all, l, maximum, pol = "EE", dl=True, all_sph = False):
    l_start, l_end = config.bins[pol][l], config.bins[pol][l + 1]
    scale_dl = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)])
    if not dl:
        scale_dl = np.ones(len(scale_dl))

    if not all_sph:
        _, power_spec_EE, power_spec_BB, _, _, _ = hp.anafast([np.zeros(config.Npix), d_all["Q"], d_all["U"]], lmax=config.L_MAX_SCALARS, pol=True)
    else:
        _, power_spec_EE, power_spec_BB, _, _, _ = hp.alm2cl([real_to_complex(np.zeros(len(d_all["EE"]))),
                                                              real_to_complex(d_all["EE"]), real_to_complex(d_all["BB"])],
                                                             lmax=config.L_MAX_SCALARS)

    if pol == "EE":
        power_spec = power_spec_EE
    else:
        power_spec = power_spec_BB

    observed_cls = power_spec * scale_dl / config.bl_gauss ** 2

    scale_betas = [(2 * l + 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    alphas = [(2 * l - 1) / 2 for l in range(config.L_MAX_SCALARS + 1)]
    betas = observed_cls * scale_betas
    locs = -((4 * np.pi / config.Npix) * config.noise_covar_pol / config.bl_gauss ** 2) * scale_dl
    denoms_factors = [(config.bl_gauss[l] ** 2) for l in range(config.L_MAX_SCALARS + 1)]
    if not dl:
        denoms_factors = np.ones(len(denoms_factors))

    norm, err = scipy.integrate.quad(compute_binned_likelihood, a=0, b=np.inf,
                                     args=(alphas[l_start:l_end], betas[l_start:l_end],
                                           locs[l_start:l_end], denoms_factors[l_start:l_end]))

    y = []
    #maxi = np.max(h_cl[:, l])/10000
    maxi = np.max(h_cl[:, l])
    #maxi = maximum
    steps = maxi / 10000
    print("MAXI:", maxi)
    xs = np.arange(0, maxi, steps)
    # xs = np.arange(0, 5 ,0.01)
    print("MAX RANGE")
    print(np.max(h_cl))
    for x in xs:
        res = compute_binned_likelihood(x, alphas[l_start:l_end], betas[l_start:l_end], locs[l_start:l_end],
                                        denoms_factors[l_start:l_end])
        y.append(res)

    return np.array(y), xs, norm



def compute_log_nc(x, pow_spec, l, s_nonCentered, v):
    dls= np.zeros(config.L_MAX_SCALARS+1)
    dls[l] = x
    var_cls = generate_var_cl(dls)
    mean_term = np.sum(np.sqrt(var_cls)*s_nonCentered*v)


    #return np.exp(-((2 * l + 1) / 2)*((2*np.pi)/(l*(l+1))) * (pow_spec[l] * config.bl_gauss[l] ** 2 / (config.noise_covar_pol * config.w)) * x +
    #              np.sqrt(2*np.pi/(l*(l+1)))*np.sqrt(x)*mean_term)
    return np.exp(-((2 * l + 1) / 2)*((2*np.pi)/(l*(l+1))) * (pow_spec[l] * config.bl_gauss[l] ** 2 / (config.noise_covar_pol * config.w)) * x +
                  mean_term)

def conditionnal_nc(s_nonCentered, l, h, pol, pix_map):
    _, alm_E, alm_B = adjoint_synthesis_hp([np.zeros(len(pix_map["Q"])), pix_map["Q"]/config.noise_covar_pol,pix_map["U"]/config.noise_covar_pol],
               bl_map = config.bl_map)

    if pol == "EE":
        v = alm_E
    else:
        v = alm_B

    #indexes = np.array([hp.Alm.getidx(config.L_MAX_SCALARS,l ,m) for m in range(l+1)])
    s = s_nonCentered[pol]
    #product = real_to_complex(s*v)
    #product = product[indexes]
    #mean_term = product[0].real + (1/np.sqrt(2))* np.sum(product[1:].real + product[1:].imag)

    max = np.max(h)
    x = np.arange(0, max, 0.001)
    pow_spec = hp.alm2cl(real_to_complex(s_nonCentered[pol]), lmax=config.L_MAX_SCALARS)


    norm, err = scipy.integrate.quad(compute_log_nc, a=0, b=np.inf,
                                     args=(pow_spec,l, s, v))
    #return np.exp(-((2*l+1)/2)*((2*np.pi)/(l*(l+1)))*(pow_spec[l]*config.bl_gauss[l]**2/(config.noise_covar_pol*config.w))*x
    #                + np.sqrt(2*np.pi/(l*(l+1)))*np.sqrt(x)*mean_term), x, norm

    res = np.array([compute_log_nc(xx, pow_spec, l, s, v) for xx in x])
    return res, x, norm