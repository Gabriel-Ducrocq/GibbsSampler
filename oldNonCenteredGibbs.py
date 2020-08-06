import numpy as np
from scipy.stats import truncnorm
import config
import utils
import healpy as hp
import time


def propose_cl(cls_old):
    clip_low = -cls_old[2:]/np.sqrt(config.proposal_variances_nc)
    return np.concatenate([np.zeros(2),
                          truncnorm.rvs(a=clip_low, b=np.inf, loc=cls_old[2:], scale=np.sqrt(config.proposal_variances_nc))])


def propose_cl_block(cls_old, l_start, l_end):
    clip_low = -cls_old[l_start:l_end]/np.sqrt(config.proposal_variances_nc[l_start-2:l_end-2])
    return truncnorm.rvs(a=clip_low, b=np.inf, loc=cls_old[l_start:l_end],
                         scale=np.sqrt(config.proposal_variances_nc[l_start-2:l_end-2]))


def compute_log_proposal(cl_old, cl_new):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
    clip_low = -cl_old[2:] / np.sqrt(config.proposal_variances_nc)
    return np.sum(truncnorm.logpdf(cl_new[2:], a=clip_low, b=np.inf, loc=cl_old[2:], scale=np.sqrt(config.proposal_variances_nc)))


def compute_log_lik(var_cls, s_nonCentered, d):
    return -(1/2)*np.sum(((d - utils.synthesis_hp(config.bl_map*np.sqrt(var_cls)*s_nonCentered))**2)*config.inv_var_noise)


def compute_log_MH_ratio2(cls_old, cls_new, var_cls_old, var_cls_new, s_nonCentered, d):
    part1 = compute_log_lik(var_cls_new, s_nonCentered, d) - compute_log_lik(var_cls_old, s_nonCentered, d)
    part2 = compute_log_proposal(cls_new, cls_old) - compute_log_proposal(cls_old, cls_new)
    return part1 + part2


def compute_log_MH_ratio(binned_cls_old, binned_cls_new, var_cls_old, var_cls_new, s_nonCentered, d, old_lik):
    new_lik = compute_log_lik(var_cls_new, s_nonCentered, d)
    part1 = new_lik - old_lik
    part2 = compute_log_proposal(binned_cls_new, binned_cls_old) - compute_log_proposal(binned_cls_old, binned_cls_new)
    return part1 + part2, new_lik


def metropolis(cls_old, var_cls_old, s_nonCentered, d):
    accept = 0
    cls_new = propose_cl(cls_old)
    var_cls_new = utils.generate_var_cl(cls_new)

    log_r = compute_log_MH_ratio(cls_old, cls_new, var_cls_old, var_cls_new, s_nonCentered, d)
    if np.log(np.random.uniform()) < log_r:
        cls_old = cls_new
        var_cls_old = var_cls_new
        accept = 1

    return cls_old, var_cls_old, accept


def metropolis_block(binned_cls_old, var_cls_old, s_nonCentered, d):
    accept = []
    old_lik = compute_log_lik(var_cls_old, s_nonCentered, d)
    for i, l_start in enumerate(config.metropolis_blocks_gibbs_nc[:-1]):
        l_end = config.metropolis_blocks_gibbs_nc[i+1]

        for _ in range(config.N_metropolis):
            binned_cls_new_block = propose_cl_block(binned_cls_old, l_start, l_end)
            binned_cls_new = binned_cls_old.copy()
            binned_cls_new[l_start:l_end] = binned_cls_new_block
            cls_new = utils.unfold_bins(binned_cls_new, config.bins)
            var_cls_new = utils.generate_var_cl(cls_new)

            log_r, new_lik = compute_log_MH_ratio(binned_cls_old, binned_cls_new, var_cls_old, var_cls_new, s_nonCentered, d, old_lik)
            if np.log(np.random.uniform()) < log_r:
                binned_cls_old = binned_cls_new
                var_cls_old = var_cls_new
                old_lik = new_lik
                accept.append(1)
            else:
                accept.append(0)

    return binned_cls_old, var_cls_old, accept


def gibbs_nc(cl_init, d, block=True, isotropic=True, binning=True):
    h_time_seconds = []
    if not block:
        total_accept = 0
    else:
        total_accept = []

    binned_cls = cl_init
    if binning:
        print(config.bins)
        cls = utils.unfold_bins(cl_init, config.bins)
    else:
        cls = cl_init

    h_cl = []
    var_cls = utils.generate_var_cl(cls)
    for i in range(config.N_nc_gibbs):
        if i % 10000 == 0:
            print("Non centered gibbs")
            print(i)

        start_time = time.process_time()
        s_nonCentered, error = utils.generate_normal_NonCentered_diag(d, var_cls, isotropic)
        if block:
            binned_cls, var_cls, accept = metropolis_block(binned_cls, var_cls, s_nonCentered, d)
            total_accept.append(accept)
        else:
            cls, var_cls, accept = metropolis(cls, var_cls, s_nonCentered, d)
            total_accept += accept

        end_time = time.process_time()
        h_cl.append(binned_cls)
        #print("\n")
        h_time_seconds.append(end_time - start_time)

    acceptance_rate = 0
    if not block:
        print("Non centered acceptance rate:")
        acceptance_rate = total_accept / config.N_nc_gibbs
        print(acceptance_rate)
    else:
        total_accept = np.array(total_accept)
        print("Non centered acceptance rate:")
        print(np.mean(total_accept, axis=0))

    return np.array(h_cl), total_accept, np.array(h_time_seconds)
