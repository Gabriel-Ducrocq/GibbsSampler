import config
import utils
import numpy as np
from scipy.stats import invgamma, truncnorm
import healpy as hp
import time



def sample_cls(alms, binning=True, dls=True):
    alms_complex = utils.real_to_complex(alms)
    observed_Cls = hp.alm2cl(alms_complex, lmax=config.L_MAX_SCALARS)
    alphas = np.array([(2 * l - 1) / 2 for l in range(config.L_MAX_SCALARS+1)])
    alphas[0] = 1
    if not dls:
        betas = np.array([(2 * l + 1) * (observed_Cl / 2) for l, observed_Cl in enumerate(observed_Cls)])
    else:
        betas = np.array([(2 * l + 1)*l*(l+1) * (observed_Cl / (4*np.pi)) for l, observed_Cl in enumerate(observed_Cls)])

    sampled_cls = betas * invgamma.rvs(a=alphas)
    #### Sampled C0 and C1 are equal to zero as expected thanks to the beta coefficient that are 0
    if binning:
        exponent = np.array([(2 * l + 1) / 2 for l in range(config.L_MAX_SCALARS + 1)])
        binned_betas = []
        binned_alphas = []
        for i, l in enumerate(config.bins[:-1]):
            somme_beta = np.sum(betas[l:config.bins[i+1]])
            somme_exponent = np.sum(exponent[l:config.bins[i+1]])
            alpha = somme_exponent - 1
            binned_alphas.append(alpha)
            binned_betas.append(somme_beta)

        binned_alphas[0] = 1
        sampled_cls = binned_betas * invgamma.rvs(a=binned_alphas)

    return sampled_cls



def default_gibbs(d, cls_init, isotropic=True, binning=True, dls=True):
    cls = cls_init
    h_cls = []
    h_time_seconds = []
    binned_cls = cls_init
    for i in range(10000):
        if i%1000 == 0:
            print("Default Gibbs")
            print(i)


        cls = utils.unfold_bins(binned_cls, config.bins)
        start_time = time.process_time()
        var_cls_full = utils.generate_var_cl(cls)
        skymap,time_to_solution, err = utils.generate_normal_ASIS_transform_def_diag(d, var_cls_full, isotropic)
        binned_cls = sample_cls(skymap, binning, dls=dls)
        end_time = time.process_time()
        h_cls.append(binned_cls)
        h_time_seconds.append(end_time - start_time)

    return np.array(h_cls), h_time_seconds