import numpy as np
from scipy.stats import truncnorm
import utils


class ClsSampler():

    def __init__(self, pix_map, lmax, bins, bl_map, noise):
        self.lmax = lmax
        self.bins = bins
        self.pix_map = pix_map
        self.bl_map = bl_map
        self.noise = noise
        self.inv_noise = 1/noise

    def sample(self, alm_map):
        return None



class MHClsSampler(ClsSampler):
    def __init__(self, pix_map, lmax, bins, bl_map, noise, metropolis_blocks, proposal_variances, n_iter = 1):
        super().__init__(pix_map, lmax, bins, bl_map, noise)
        if metropolis_blocks == None:
            self.metropolis_blocks = list(range(2, len(self.bins)))
        else:
            self.metropolis_blocks = metropolis_blocks

        self.n_iter = n_iter
        self.proposal_variances = proposal_variances

    def propose_cl(self, cls_old, l_start, l_end):
        """

        :param cls_old: old cls sample
        :param l_start: starting index of the block
        :param l_end: ending index (not included) of the block
        :return: propose cls

        Note that every index is shifted by -2: the first l_start is 2 - since we are not samplint l=0,1 - and the
        proposal variance also starts at l = 2. But then we need to take the first element of this array, hence setting
        l_start - 2:l_end - 2
        """
        clip_low = -cls_old[l_start:l_end] / np.sqrt(self.proposal_variances[l_start - 2:l_end - 2])
        return truncnorm.rvs(a=clip_low, b=np.inf, loc=cls_old[l_start:l_end],
                             scale=np.sqrt(self.proposal_variances[l_start - 2:l_end - 2]))

    def compute_log_proposal(self, cl_old, cl_new):
        return None

    def compute_log_likelihood(self, var_cls, s_nonCentered):
        return -(1 / 2) * np.sum(
            ((self.pix_map - utils.synthesis_hp(self.bl_map * np.sqrt(var_cls) * s_nonCentered)) ** 2) * self.inv_noise)

    def compute_log_MH_ratio(self, binned_cls_old, binned_cls_new, var_cls_new, s_nonCentered, old_lik):
        new_lik = self.compute_log_likelihood(var_cls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = self.compute_log_proposal(binned_cls_new, binned_cls_old) - self.compute_log_proposal(binned_cls_old,
                                                                                            binned_cls_new)
        return part1 + part2, new_lik



