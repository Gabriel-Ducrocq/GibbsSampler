from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import ClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma
import time
import config
from linear_algebra import compute_inverse_matrices, compute_matrix_product
from numba import njit, prange



class CenteredClsSampler(ClsSampler):


    def sample(self, alms):
        """
        :param alms: alm skymap
        :return: Sample each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        alms_complex = utils.real_to_complex(alms)
        observed_Cls = hp.alm2cl(alms_complex, lmax=config.L_MAX_SCALARS)
        exponent = np.array([(2 * l + 1) / 2 for l in range(config.L_MAX_SCALARS + 1)])
        binned_betas = []
        binned_alphas = []
        betas = np.array([(2 * l + 1) * l * (l + 1) * (observed_Cl / (4 * np.pi)) for l, observed_Cl in
                          enumerate(observed_Cls)])

        for i, l in enumerate(self.bins[:-1]):
            somme_beta = np.sum(betas[l:self.bins[i + 1]])
            somme_exponent = np.sum(exponent[l:self.bins[i + 1]])
            alpha = somme_exponent - 1
            binned_alphas.append(alpha)
            binned_betas.append(somme_beta)

        binned_alphas[0] = 1
        sampled_cls = binned_betas * invgamma.rvs(a=binned_alphas)

        return sampled_cls






class CenteredConstrainedRealization(ConstrainedRealization):

    def sample(self, var_cls):
        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)

        b_weiner = self.bl_map * utils.adjoint_synthesis_hp(self.inv_noise * self.pix_map)
        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise))

        start = time.time()
        Sigma = 1/(inv_var_cls + self.inv_noise * (config.Npix / (4 * np.pi)) * config.bl_map ** 2)
        weiner = Sigma * b_weiner
        flucs = Sigma * b_fluctuations
        map = weiner + flucs
        err = 0
        map[[0, 1, config.L_MAX_SCALARS + 1, config.L_MAX_SCALARS + 2]] = 0.0
        time_to_solution = time.time() - start

        return map, time_to_solution, err


complex_dim = int((config.L_MAX_SCALARS+1)*(config.L_MAX_SCALARS+2)/2)


@njit(fastmath=True, parallel=True)
def matrix_product(dls_, b):
    alms_shape = np.zeros((complex_dim, 3, 3))
    result = np.zeros(((config.L_MAX_SCALARS+1)**2, 3))

    for l in prange(config.L_MAX_SCALARS + 1):
        for m in range(l + 1):
            idx = m * (2 * config.L_MAX_SCALARS + 1 - m) // 2 + l
            if l == 0:
                alms_shape[idx, :, :] = dls_[l, :, :]
            else:
                alms_shape[idx, :, :] = dls_[l, :, :] * 2 * np.pi / (l * (l + 1))

    for i in prange(config.L_MAX_SCALARS + 1):
        result[i, 0] = alms_shape[i, 0, 0] * b[i, 0] + alms_shape[i, 0, 1] * b[i, 1]
        result[i, 1] = alms_shape[i, 1, 0] * b[i, 0] + alms_shape[i, 0, 1] * b[i, 1]
        result[i, 2] = alms_shape[i, 2, 2] * b[i, 2]

    for i in prange(config.L_MAX_SCALARS+1, complex_dim):
        result[2*i - (config.L_MAX_SCALARS+1), 0] = alms_shape[i, 0, 0]*b[2*i - (config.L_MAX_SCALARS+1), 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1), 1]
        result[2*i - (config.L_MAX_SCALARS+1), 1] = alms_shape[i, 1, 0]*b[2*i - (config.L_MAX_SCALARS+1), 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1), 1]
        result[2*i - (config.L_MAX_SCALARS+1), 2] = alms_shape[i, 2, 2]*b[2*i - (config.L_MAX_SCALARS+1), 2]

        result[2*i - (config.L_MAX_SCALARS+1) + 1, 0] = alms_shape[i, 0, 0]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 1]
        result[2*i - (config.L_MAX_SCALARS+1) + 1, 1] = alms_shape[i, 1, 0]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 1]
        result[2*i - (config.L_MAX_SCALARS+1) + 1, 2] = alms_shape[i, 2, 2]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 2]

    return result

@njit(fastmath=True, parallel=True)
def compute_inverse_and_cholesky(all_dls, pix_part_variance):
    inv_dls = np.zeros((len(all_dls), 3, 3))
    chol_dls = np.zeros((len(all_dls), 3, 3))

    for i in prange(2, len(all_dls)):
        inv_dls[i, :, :] = np.linalg.inv(all_dls[i]) + np.diag(pix_part_variance[i])
        chol_dls[i, :, :] = np.linalg.cholesky(inv_dls[i])

    return inv_dls, chol_dls




class PolarizedCenteredConstrainedRealization(ConstrainedRealization):
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, isotropic=True):
        super().__init__(pix_map, noise_temp, bl_map, lmax, Npix, isotropic=True)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.noise_alm = np.hstack(zip(self.noise_temp*np.ones((config.L_MAX_SCALARS+1)**2), self.noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)
                                       , self.noise_pol*np.ones((config.L_MAX_SCALARS+1)**2))).reshape(-1, 3)
        self.inv_noise_alm = 1/self.noise_alm
        self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([self.inv_noise_temp*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2], axis = 1)
        self.bl_fwhm = bl_fwhm
        self.complex_dim = int((config.L_MAX_SCALARS +1)*(config.L_MAX_SCALARS+2)/2)

    def sample(self, all_cls):
        start = time.time()
        inv_all_l, chol_all_l = compute_inverse_matrices(all_cls, config.L_MAX_SCALARS+1, self.pix_part_variance)

        b_weiner_unpacked = utils.adjoint_synthesis_hp([self.inv_noise_temp * self.pix_map[0],
                    self.inv_noise_pol * self.pix_map[1], self.inv_noise_pol * self.pix_map[2]], self.bl_fwhm)


        b_weiner = np.asfortranarray(np.stack(b_weiner_unpacked, axis = 1))
        b_fluctuations = np.random.normal(size=((config.L_MAX_SCALARS+1)**2, 3)).reshape((-1, 3), order="F")

        chol_all_l = np.asfortranarray(np.asarray(chol_all_l))

        flucs = compute_matrix_product(chol_all_l, b_fluctuations)
        mean = compute_matrix_product(inv_all_l, b_weiner)
        map = np.asarray(mean) + np.asarray(flucs)
        time_to_solution = time.time() - start
        err = 0
        print("Time to solution")
        print(time_to_solution)
        return map, time_to_solution, err

    def sample2(self, all_dls):
        start = time.time()
        inv_dls, chol_dls = compute_inverse_and_cholesky(all_dls, self.pix_part_variance)


        b_weiner_unpacked = utils.adjoint_synthesis_hp([self.inv_noise_temp * self.pix_map[0],
                    self.inv_noise_pol * self.pix_map[1], self.inv_noise_pol * self.pix_map[2]], self.bl_fwhm)


        b_weiner = np.stack(b_weiner_unpacked, axis = 1)
        b_fluctuations = np.random.normal(size=((config.L_MAX_SCALARS+1)**2, 3)).reshape((-1, 3), order="F")

        fluctuations = matrix_product(chol_dls, b_fluctuations)
        weiner_map = matrix_product(inv_dls, b_weiner)
        map = weiner_map + fluctuations
        time_to_solution = time.time() - start
        err = 0
        print("Time to solution")
        print(time_to_solution)
        return map, time_to_solution, err



class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise, beam, nside, lmax, Npix, polarization = False, bins=None, n_iter = 10000):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, lmax, Npix, isotropic=True)
        self.cls_sampler = CenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise)





