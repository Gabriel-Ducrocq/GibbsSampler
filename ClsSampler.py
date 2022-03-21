import numpy as np
from scipy.stats import truncnorm
import utils
import healpy as hp
import time


class ClsSampler():
    def __init__(self, pix_map, lmax, nside, bins, bl_map, noise, mask_path = None):
        """
        This class is the base class every conditional sampler of the power spectrum will inherit.
        :param pix_map: array size Npix, observed skymap called d in the paper.
        :param lmax: int, maximum \ell on which we perform inference
        :param nside: NSIDE used to generate the grid on the sphere
        :param bins: array of integers, each element is the starting index and ending index of a power spectrum bin.
        :param bl_map: array of floats, the b_\ells expension over all the spherical harmonics coefficients. This is the diagonal
                of the B matrix in the paper.
        :param noise: float, noise level, assumes the noise covariance matrix is proportional to the diagonal.
        :param mask_path: string, path to the skymask.
        """
        self.lmax = lmax
        self.bins = bins
        self.nside = nside
        self.pix_map = pix_map
        self.bl_map = bl_map
        self.noise = noise
        self.inv_noise = 1/noise
        if mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside) #read the mask and downgrades it to the right resolution.
            self.inv_noise *= self.mask # multiply the diagonal of the inverse noise covariance matrix by the mask.
                                        #Typically the mask is zero on some pixels and the N^-1 matrix is singular
                                        # Having 0 entries on the N^-1 matrix for some pixels assumes infinite noise on these pixels.
                                        # See the papers of Jewell and H-K Eriksen.

    def sample(self, alm_map):
        """

        :param alm_map: array of float, skymap expressed in spherical harmonics domain, m major.
        :return: None
        """
        return None



class MHClsSampler(ClsSampler):
    def __init__(self, pix_map, lmax, nside, bins, bl_map, noise, metropolis_blocks, proposal_variances, n_iter = 1, mask_path = None,
                 polarization=False):
        """
        This is the basic class for making non centered power spectrum sampling steps. All the other classes to make such a step
        will inherit from this one.

        :param pix_map: pix_map: array size Npix, observed skymap called d in the paper.
        :param lmax: int, maximum \ell on which we perform inference
        :param nside: NSIDE used to generate the grid on the sphere
        :param bins: array of integers, each element is the starting index and ending index of a power spectrum bin.
        :param bl_map: array of floats, the b_\ells expension over all the spherical harmonics coefficients. This is the diagonal
                of the B matrix in the paper.
        :param noise: float, noise level, assumes the noise covariance matrix is proportional to the diagonal.
        :param metropolis_blocks: array of integers, size (number of blocks+1,). Each element is the start and end of a block for the non cnetered power spectrum sampling.
        :param proposal_variances: array of floats, size (number of blocks, ). Proposal variances for the blocks.
        :param n_iter: integer. Number of iterations of the M-w-G to do.
        :param mask_path: string, path to the skymask.
        :param polarization: boolean, whether we are dealing with "TT" only or "EE" and "BB" only.
        """
        super().__init__(pix_map, lmax, nside, bins, bl_map, noise, mask_path)
        if metropolis_blocks is None:
            self.metropolis_blocks = list(range(2, len(self.bins)))
        else:
            self.metropolis_blocks = metropolis_blocks

        self.n_iter = n_iter
        self.proposal_variances = proposal_variances
        self.polarization = polarization
        self.dls_to_cls_array = np.array([2 * np.pi / (l * (l + 1)) if l != 0 else 0 for l in range(lmax + 1)])

    def dls_to_cls(self, dls_):
        return dls_ * self.dls_to_cls_array

    def propose_dl(self, dls_old):
        """
        :param dls_old: old dls sample or if polarization mode on, coefficients of the lower triang chol matrix
        :param l_start: starting index of the block
        :param l_end: ending index (not included) of the block
        :return: propose dls

        Note that every index is shifted by -2: the first l_start is 2 - since we are not samplint l=0,1 - and the
        proposal variance also starts at l = 2. But then we need to take the first element of this array, hence setting
        l_start - 2:l_end - 2
        """
        clip_low = -dls_old[2:] / np.sqrt(self.proposal_variances)
        return np.concatenate([np.zeros(2), truncnorm.rvs(a=clip_low, b=np.inf, loc=dls_old[2:],
                             scale=np.sqrt(self.proposal_variances))])

    def compute_log_proposal(self, cl_old, cl_new):
        """
        This method will be redefined in the inheriting objects.
        """
        return None

    def compute_log_likelihood(self, var_cls, s_nonCentered):
        """

        :param var_cls: array of floats, size (L_max +1,), diagonal of the diagonal matrix C, see paper.
        :param s_nonCentered: array of floats, size ((L_max +1)**2, ), non centered sky map in spherical harmonic basis, in real convention.
        :return: the log likelihood evaluated in s_nonCentered and var_cls
        """
        result = -(1 / 2) * np.sum(
            ((self.pix_map - utils.synthesis_hp(self.bl_map * np.sqrt(var_cls) * s_nonCentered)) ** 2) * self.inv_noise)

        return result

    def compute_log_MH_ratio(self, log_r_ratio, var_cls_new, s_nonCentered, old_lik):
        """
        computes the log MH ratio

        :param log_r_ratio: float, log ratio of the proposals.
        :param var_cls_new: array of floats, size (L_max +1,), diagonal of the diagonal matrix C in the proposed power spectrum, see paper.
        :param s_nonCentered: s_nonCentered: array of floats, size ((L_max +1)**2, ), non centered sky map in spherical harmonic basis, in real convention.
        :param old_lik: float. log likelihood evaluated at the previous power spectrum. Keeping it in memory avoids computing it two times.
        :return: float. log MH ratio.
        """
        new_lik = self.compute_log_likelihood(var_cls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = log_r_ratio
        return part1 + part2, new_lik



