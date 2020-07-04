from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import MHClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma
import time
import config
import utils
from scipy.stats import truncnorm


def compute_var_high_low(var_cls_full):
    var_cls_high = var_cls_full.copy()
    var_cls_low = var_cls_full.copy()
    var_cls_low[config.mask_non_centered] = 1
    var_cls_high[config.mask_centered] = 1

    inv_var_cls_low = np.zeros(len(var_cls_full))
    np.reciprocal(var_cls_low, where=config.mask_inversion, out=inv_var_cls_low)
    return var_cls_low, var_cls_high, inv_var_cls_low


class PNCPConstrainedRealization(ConstrainedRealization):


    def sample(self, var_cls):
        var_cls_low, var_cls_high, inv_var_cls_low = compute_var_high_low(var_cls)
        b_weiner = np.sqrt(var_cls_high) * self.bl_map * utils.adjoint_synthesis_hp((1 / self.noise) * self.pix_map)
        b_fluctuations = np.random.normal(loc=0, scale=1, size=len(var_cls))*np.sqrt(inv_var_cls_low) + \
                     np.sqrt(var_cls_high)*self.bl_map \
                     *utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)*np.sqrt(self.inv_noise))

        start = time.time()
        Sigma = 1/(inv_var_cls_low + (var_cls_high*self.Npix/(self.noise*4*np.pi))*self.bl_map**2)
        weiner = Sigma*b_weiner
        fluctuations = Sigma*b_fluctuations

        map = weiner + fluctuations
        map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0
        time_to_solution = time.time() - start
        error = 0

        return map, time_to_solution, error


class PNCPClsSampler(MHClsSampler):

    def __init__(self, pix_map, lmax, bins, bl_map, noise, metropolis_blocks, proposal_variances, l_cut, n_iter = 1):
        super().__init__(pix_map, lmax, bins, bl_map, noise, metropolis_blocks, proposal_variances, n_iter = n_iter)
        self.l_cut = l_cut
        if metropolis_blocks == None:
            self.metropolis_blocks = list(range(l_cut, len(self.bins)))
        else:
            self.metropolis_blocks = metropolis_blocks

    def sample_low_l(self, alms):
        alms_complex = utils.real_to_complex(alms)
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax)[:self.l_cut]
        alphas = np.array([(2 * l - 1)/2 for l in range(self.l_cut)])
        alphas[0] = 1
        betas = np.array([(2 * l + 1)*l*(l+1) * (observed_Cl /(4*np.pi)) for l, observed_Cl in enumerate(observed_Cls)])
        sampled_cls = betas * invgamma.rvs(a=alphas)
        return sampled_cls

    def compute_log_proposal(self, cl_old, cl_new):
        clip_low = -cl_old[self.l_cut:] / np.sqrt(self.proposal_variances[self.l_cut - 2:])
        return np.sum(truncnorm.logpdf(cl_new[self.l_cut:], a=clip_low, b=np.inf, loc=cl_old[self.l_cut:],
                                       scale=np.sqrt(self.proposal_variances[self.l_cut - 2:])))

    def sample_high_l(self, binned_cls_old, var_cls_old, s_nonCentered):
        accept = []
        old_lik = self.compute_log_likelihood(var_cls_old, s_nonCentered)
        for i, l_start in enumerate(self.metropolis_blocks[:-1]):
            l_end = self.metropolis_blocks[i + 1]

            for _ in range(self.n_iter):
                binned_cls_new_block = self.propose_cl(binned_cls_old, l_start, l_end)

                binned_cls_new = binned_cls_old.copy()
                binned_cls_new[l_start:l_end] = binned_cls_new_block
                cls_new = utils.unfold_bins(binned_cls_new, self.bins)

                var_cls_new = utils.generate_var_cl(cls_new)
                _, var_cls_high_full_new, _ = compute_var_high_low(var_cls_new)

                log_r, new_lik = self.compute_log_MH_ratio(binned_cls_old, binned_cls_new,
                                                            var_cls_high_full_new, s_nonCentered, old_lik)

                if np.log(np.random.uniform()) < log_r:
                    binned_cls_old = binned_cls_new
                    var_cls_old = var_cls_new
                    old_lik = new_lik
                    accept.append(1)
                else:
                    accept.append(0)

        return binned_cls_old, var_cls_old, accept


class PNCPGibbs(GibbsSampler):

    def __init__(self, pix_map, noise, beam, nside, lmax, Npix, proposal_variances, l_cut, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = PNCPConstrainedRealization(pix_map, noise, self.bl_map, lmax, Npix, isotropic=True)
        self.cls_sampler = PNCPClsSampler(pix_map, lmax, self.bins, self.bl_map, noise, metropolis_blocks,
                                                 proposal_variances, l_cut, n_iter = n_iter_metropolis)

        self.l_cut = l_cut
        if metropolis_blocks is None:
            self.metropolis_blocks = list(range(l_cut, len(self.bins)))
        else:
            self.metropolis_blocks = metropolis_blocks


    def run(self, cls_init):
        h_cls = []
        if self.l_cut not in self.metropolis_blocks:
            print("l_cut is not in blocks")
            return None

        h_cls.append(cls_init)
        cls = cls_init
        h_time_seconds = []
        all_acceptions = []
        unfolded_cls = utils.unfold_bins(cls_init, self.bins)
        var_cl_full = utils.generate_var_cl(unfolded_cls)
        _, var_cls_high_full, _ = compute_var_high_low(var_cl_full)
        for i in range(self.n_iter):
            if i %10000 == 0:
                print("PNCP Gibbs, iteratio:", i)

            start_time = time.process_time()
            s, time_to_solution, error = self.constrained_sampler.sample(var_cl_full)
            cls_low = self.cls_sampler.sample_low_l(s)
            cls = np.concatenate([cls_low, cls[self.l_cut:]])
            cls, var_cls_high_full, acceptions = self.cls_sampler.sample_high_l(cls, var_cls_high_full, s)
            unfolded_cls = utils.unfold_bins(cls, self.bins)
            var_cl_full = utils.generate_var_cl(unfolded_cls)

            h_cls.append(cls)
            all_acceptions.append(acceptions)

            end_time = time.process_time()
            h_time_seconds.append(end_time - start_time)

        print("Acceptions PNCP:")
        acceptions_by_l = np.mean(all_acceptions, axis=0)
        print(acceptions_by_l)

        return np.array(h_cls), np.array(acceptions_by_l), np.array(h_time_seconds)