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


    """
    def propose_dl(self, dls_old, l_start, l_end):

        :param dls_old: old dls sample or if polarization mode on, coefficients of the lower triang chol matrix
        :param l_start: starting index of the block
        :param l_end: ending index (not included) of the block
        :return: propose dls

        Note that every index is shifted by -2: the first l_start is 2 - since we are not samplint l=0,1 - and the
        proposal variance also starts at l = 2. But then we need to take the first element of this array, hence setting
        l_start - 2:l_end - 2

        if not self.polarization:
            clip_low = -dls_old[l_start:l_end] / np.sqrt(self.proposal_variances[l_start - 2:l_end - 2])
            return truncnorm.rvs(a=clip_low, b=np.inf, loc=dls_old[l_start:l_end],
                                 scale=np.sqrt(self.proposal_variances[l_start - 2:l_end - 2]))
        else:
            new_dls = np.zeros((l_end - l_start, 3, 3))
            ### Sampling cls_TT:
            clip_low_TT = -dls_old[l_start:l_end, 0, 0] / np.sqrt(self.proposal_variances["TT"][l_start - 2:l_end - 2])
            new_dls_TT = truncnorm.rvs(a=clip_low_TT, b=np.inf, loc=dls_old[l_start:l_end, 0, 0],
                                 scale=np.sqrt(self.proposal_variances["TT"][l_start - 2:l_end - 2]))

            ### Sampling cls_EE
            clip_low_EE = -dls_old[l_start:l_end, 1, 1] / np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2])
            new_dls_EE = truncnorm.rvs(a=clip_low_EE, b=np.inf, loc=dls_old[l_start:l_end, 1, 1],
                                 scale=np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2]))

            ### Sampling cls_BB
            clip_low_BB = -dls_old[l_start:l_end, 2, 2] / np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2])
            news_dls_BB = truncnorm.rvs(a=clip_low_BB, b=np.inf, loc=dls_old[l_start:l_end, 2, 2],
                                 scale=np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2]))

            ### Sampling cls_TE
            upp_bound = np.sqrt(new_dls_TT*new_dls_EE)
            low_bound = -np.sqrt(new_dls_TT*new_dls_EE)
            clip_high_TE = (upp_bound-dls_old[l_start:l_end, 1, 0])/np.sqrt(self.proposal_variances["TE"][l_start-2:l_end-2])
            clip_low_TE = (low_bound-dls_old[l_start:l_end, 1, 0])/np.sqrt(self.proposal_variances["TE"][l_start-2:l_end-2])
            new_dls_TE = truncnorm.rvs(a=clip_low_TE, b=clip_high_TE, loc=dls_old[l_start:l_end, 1, 0],
                                 scale=np.sqrt(self.proposal_variances["TE"][l_start-2:l_end-2]))

            new_dls[:, 0, 0] = new_dls_TT
            new_dls[:, 1, 1] = new_dls_EE
            new_dls[:, 2, 2] = news_dls_BB
            new_dls[:, 1, 0] = new_dls_TE
            new_dls[:, 0, 1] = new_dls_TE

            new_cholesky = new_dls.copy()
            for i in range(l_end-l_start):
                l = l_start + i
                new_cholesky[i, :, : ] = np.linalg.cholesky(new_dls[i, :, :]*(2*np.pi/(l*(l+1))))

            return new_dls, new_cholesky
        """


    def compute_log_proposal(self, cl_old, cl_new):
        return None

    def compute_log_likelihood(self, var_cls, s_nonCentered):
        start = time.time()
        result = -(1 / 2) * np.sum(
            ((self.pix_map - utils.synthesis_hp(self.bl_map * np.sqrt(var_cls) * s_nonCentered)) ** 2) * self.inv_noise)

        end = time.time()
        return result

    """
    def compute_log_MH_ratio(self, binned_cls_old, binned_cls_new, var_cls_new, s_nonCentered, old_lik):
        new_lik = self.compute_log_likelihood(var_cls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = self.compute_log_proposal(binned_cls_new, binned_cls_old) - self.compute_log_proposal(binned_cls_old,
                                                                                            binned_cls_new)
        return part1 + part2, new_lik
    """

    def compute_log_MH_ratio(self, log_r_ratio, var_cls_new, s_nonCentered, old_lik):
        new_lik = self.compute_log_likelihood(var_cls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = log_r_ratio
        return part1 + part2, new_lik



