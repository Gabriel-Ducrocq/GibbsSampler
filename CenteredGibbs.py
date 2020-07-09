from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import ClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma
import time
import config
from linear_algebra import compute_inverse_matrices



class CenteredClsSampler(ClsSampler):


    def sample(self, alms):
        """
        :param alms: alm skymap
        :return: Sample each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        alms_complex = utils.real_to_complex(alms)
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax)
        exponent = np.array([(2 * l + 1) / 2 for l in range(self.lmax + 1)])
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
        map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0
        time_to_solution = time.time() - start

        return map, time_to_solution, err


class PolarizedCenteredConstrainedRealization(ConstrainedRealization):
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, isotropic=True):
        super().__init__(pix_map, noise_temp, bl_map, lmax, Npix, isotropic=True)
        self.noise_temp = np.ones(self.lmax)*noise_temp
        self.noise_pol = np.ones(self.lmax)*noise_pol
        self.noise_alm = np.hstack(zip(self.noise_temp, self.noise_pol, self.noise_pol))
        self.inv_noise_alm = 1/self.noise_alm
        self.pix_part_variance = (self.Npix/(4*np.pi))*self.bl_map**2*self.inv_noise
        self.noise_pix = np.concatenate([self.noise_temp, self.noise_pol, self.noise_pol])
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.bl_fwhm = bl_fwhm

    def sample(self, all_cls):
        start = time.time()
        inv_all_l, chol_all_l = compute_inverse_matrices(all_cls, self.lmax+1, self.pix_part_variance)

        b_weiner_unpacked = utils.adjoint_synthesis_hp([self.inv_noise_temp * self.pix_map[0],
                    self.inv_noise_pol * self.pix_map[1], self.inv_noise_pol * self.pix_map[1]], self.bl_fwhm)

        b_weiner = np.hstack(zip(b_weiner_unpacked))
        b_fluctuations = np.random.normal(size=3*(self.lmax+1)**2)

        time_to_solution = time.time() - start
        err = 0
        return map, time_to_solution, err



class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise, beam, nside, lmax, Npix, polarization = False, bins=None, n_iter = 10000):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, lmax, Npix, isotropic=True)
        self.cls_sampler = CenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise)





