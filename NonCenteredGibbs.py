from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import ClsSampler, MHClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma
import time
import config
import utils
from scipy.stats import truncnorm




class NonCenteredConstrainedRealization(ConstrainedRealization):

    def sample(self, var_cls):
        b_weiner = np.sqrt(var_cls) * self.bl_map * utils.adjoint_synthesis_hp(self.pix_map * self.inv_noise)
        b_fluctuations = np.random.normal(size=len(var_cls)) + \
                         np.sqrt(var_cls) * self.bl_map * \
                         utils.adjoint_synthesis_hp(np.random.normal(size=self.Npix) * np.sqrt(self.inv_noise))

        start = time.time()
        Sigma = 1/(1 + (var_cls / self.noise) * (self.Npix / (4 * np.pi)) * self.bl_map ** 2)
        weiner = Sigma* b_weiner
        flucs = Sigma * b_fluctuations
        map = weiner + flucs
        map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0
        error = 0
        time_to_solution = time.time() - start

        return map, time_to_solution, error


class NonCenteredClsSampler(MHClsSampler):

    def compute_log_proposal(self, cl_old, cl_new):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low = -cl_old[2:] / np.sqrt(self.proposal_variances)
        return np.sum(truncnorm.logpdf(cl_new[2:], a=clip_low, b=np.inf, loc=cl_old[2:],
                                   scale=np.sqrt(self.proposal_variances)))

    def sample(self, alm_map_non_centered, binned_cls_old, var_cls_old):
        """
        :param binned_cls_old: binned power spectrum, including monopole and dipole
        :param var_cls_old: variance associated to this power spectrum, including monopole and dipole
        :param alm_map_non_centered: non centered skymap expressed in harmonic domain
        :return: a new sampled power spectrum, using M-H algorithm

        Not that here l_start and l_end are not shifted by -2 because binned_cls_old contains ALL ell, including monopole
        and dipole
        """
        accept = []
        old_lik = self.compute_log_likelihood(var_cls_old, alm_map_non_centered)
        for i, l_start in enumerate(self.metropolis_blocks[:-1]):
            l_end = self.metropolis_blocks[i + 1]

            for _ in range(self.n_iter):
                binned_cls_new_block = self.propose_cl(binned_cls_old, l_start, l_end)
                binned_cls_new = binned_cls_old.copy()
                binned_cls_new[l_start:l_end] = binned_cls_new_block
                cls_new = utils.unfold_bins(binned_cls_new, self.bins)
                var_cls_new = utils.generate_var_cl(cls_new)
                log_r, new_lik = self.compute_log_MH_ratio(binned_cls_old, binned_cls_new, var_cls_new,
                                                  alm_map_non_centered, old_lik)

                if np.log(np.random.uniform()) < log_r:
                    binned_cls_old = binned_cls_new
                    var_cls_old = var_cls_new
                    old_lik = new_lik
                    accept.append(1)
                else:
                    accept.append(0)

        return binned_cls_old, var_cls_old, accept



class NonCenteredGibbs(GibbsSampler):
    def __init__(self, pix_map, noise, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = NonCenteredConstrainedRealization(pix_map, noise, self.bl_map, lmax, Npix, isotropic=True)
        self.cls_sampler = NonCenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise, metropolis_blocks,
                                                 proposal_variances, n_iter = n_iter_metropolis)

    def run(self, cls_init):
        h_accept = []
        h_cls = []
        h_time_seconds = []
        binned_cls = cls_init
        cls = utils.unfold_bins(binned_cls, self.bins)
        var_cls_full = utils.generate_var_cl(cls)
        h_cls.append(binned_cls)
        for i in range(self.n_iter):
            if i % 1000 == 0:
                print("Non centered Gibbs, iteration:", i)

            start_time = time.process_time()
            alm_map, time_to_solution, err = self.constrained_sampler.sample(var_cls_full)
            binned_cls, var_cls_full, accept = self.cls_sampler.sample(alm_map, binned_cls, var_cls_full)
            end_time = time.process_time()
            h_cls.append(binned_cls)
            h_time_seconds.append(end_time - start_time)
            h_accept.append(accept)

        print("Non centered gibbs acceptance rate:", np.mean(np.array(h_accept)))

        return np.array(h_cls), h_time_seconds
