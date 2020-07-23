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
from numba import prange
from CenteredGibbs import PolarizedCenteredConstrainedRealization




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


def compute_sigma_and_chol(all_chol_cls, pix_part_variance):
    sigma = np.zeros((len(all_chol_cls), 3, 3))
    sigma_chol = np.zeros((len(all_chol_cls), 3, 3))
    for l in prange(2, len(all_chol_cls)):
        block_sigma = np.linalg.inv(np.dot(np.dot(all_chol_cls[l, :, :].T, np.diag(pix_part_variance[l, :])), all_chol_cls[l, :, :])
                                       + np.diag([1, 1, 1]))

        sigma[l, :, :] = block_sigma
        sigma_chol[l, :, :] = np.linalg.cholesky(block_sigma)

    return sigma, sigma_chol


class PolarizedNonCenteredConstrainedRealization(ConstrainedRealization):
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, isotropic=True):
        super().__init__(pix_map, noise_temp, bl_map, lmax, Npix, isotropic=True)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([self.inv_noise_temp*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2], axis = 1)

        self.bl_fwhm = bl_fwhm
        self.pol_centered_constraint_realizer = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                            bl_map, lmax, Npix, bl_fwhm, isotropic=True)

    def sample(self, all_dls, all_chol_cls):
        #print("VERIF CHOLS AND DLS")
        #for i in range(2, len(all_dls)):
        #    print(all_chol_cls[i, :, :])
        #    print(np.linalg.cholesky(all_dls[i, :, :]*(2*np.pi/(i*(i+1)))))
        #    print("\n")

        #print("Done")


        all_inv_chol_cls = np.zeros((len(all_chol_cls), 3, 3))
        for i in range(2, config.L_MAX_SCALARS+1):
            all_inv_chol_cls[i, :, :] = np.linalg.inv(all_chol_cls[i, :, :])

        centered_skymap, time_to_solution, err = self.pol_centered_constraint_realizer.sample(all_dls)
        non_centered_skymap = utils.matrix_product(all_inv_chol_cls, centered_skymap)
        return non_centered_skymap, time_to_solution, err


"""
    def sample(self, all_chol_dls):
        start = time.time()
        rescaling = [0 if l == 0 else 2*np.pi/(l*(l+1)) for l in range(self.lmax+1)]
        all_chol_cls = all_chol_dls
        for i in range(self.lmax+1):
            all_chol_cls[i, :, :] *= np.sqrt(rescaling[i])

        variance, chol_variance = compute_sigma_and_chol(all_chol_cls, self.pix_part_variance)


        b_weiner_unpacked_temp = utils.adjoint_synthesis_hp([self.inv_noise_temp * self.pix_map[0],
                    self.inv_noise_pol * self.pix_map[1], self.inv_noise_pol * self.pix_map[2]], self.bl_fwhm)

        b_weiner_unpacked_temp = np.stack(b_weiner_unpacked_temp, axis=1)
        b_weiner = utils.matrix_product(all_chol_cls, b_weiner_unpacked_temp)
        b_fluctuations = np.random.normal(size=((config.L_MAX_SCALARS+1)**2, 3))

        weiner_map = utils.matrix_product(variance, b_weiner)
        fluctuations = utils.matrix_product(chol_variance, b_fluctuations)
        map = weiner_map + fluctuations
        time_to_solution = time.time() - start
        err = 0
        #print("Time to solution")
        #print(time_to_solution)
        return map, time_to_solution, err
"""





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









class PolarizationNonCenteredClsSampler(MHClsSampler):
    def __init__(self, pix_map, lmax, nside, bins, bl_map, noise_I, noise_Q, metropolis_blocks, proposal_variances, n_iter = 1, polarization=True):
        super().__init__(pix_map, lmax, bins, bl_map, noise_I, metropolis_blocks, proposal_variances, n_iter = n_iter,
                       polarization=polarization)
        self.nside=nside
        self.noise_I = noise_I
        self.noise_Q = noise_Q

    def compute_log_proposal(self, dl_old, dl_new, l_start, l_end):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low_TT = -dl_old[l_start:l_end, 0, 0] / np.sqrt(self.proposal_variances["TT"][l_start - 2:l_end - 2])
        probas_TT = truncnorm.logpdf(dl_new[l_start:l_end, 0, 0], a=clip_low_TT, b=np.inf, loc=dl_old[l_start:l_end, 0, 0],
                                   scale=np.sqrt(self.proposal_variances["TT"][l_start - 2:l_end - 2]))

        clip_low_EE = -dl_old[l_start:l_end, 1, 1] / np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2])
        probas_EE = truncnorm.logpdf(dl_new[l_start:l_end, 1, 1], a=clip_low_EE, b=np.inf, loc=dl_old[l_start:l_end, 1, 1],
                                 scale=np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2]))

        clip_low_BB = -dl_old[l_start:l_end, 2, 2] / np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2])
        probas_bb = truncnorm.logpdf(dl_new[l_start:l_end, 2, 2], a=clip_low_BB, b=np.inf, loc=dl_old[l_start:l_end, 2, 2],
                                 scale=np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2]))

        upp_bound = np.sqrt(dl_new[l_start:l_end, 0, 0] * dl_new[l_start:l_end, 1, 1])
        low_bound = - np.sqrt(dl_new[l_start:l_end, 0, 0] * dl_new[l_start:l_end, 1, 1])
        clip_high_TE = (upp_bound - dl_old[l_start:l_end, 1, 0]) / np.sqrt(self.proposal_variances["TE"][l_start - 2:l_end - 2])
        clip_low_TE = (low_bound - dl_old[l_start:l_end, 1, 0]) / np.sqrt(self.proposal_variances["TE"][l_start - 2:l_end - 2])
        probas_te = truncnorm.logpdf(dl_new[l_start:l_end, 1, 0], a=clip_low_TE, b=clip_high_TE, loc=dl_old[l_start:l_end, 1, 0],
                                 scale=np.sqrt(self.proposal_variances["TE"][l_start - 2:l_end - 2]))

        return np.sum(probas_TT) + np.sum(probas_EE) + np.sum(probas_bb) + np.sum(probas_te)

    def compute_log_likelihood(self, chol_cls, s_nonCentered):
        prod = self.bl_map.reshape((len(self.bl_map), -1)) * utils.matrix_product(chol_cls, s_nonCentered)
        alm_TT = utils.real_to_complex(prod[:, 0])
        alm_EE = utils.real_to_complex(prod[:, 1])
        alm_BB = utils.real_to_complex(prod[:, 2])

        map = hp.alm2map([alm_TT, alm_EE,alm_BB],lmax=self.lmax, nside=self.nside)

        return -(1 / 2) * np.sum(
            ((self.pix_map[0] - map[0])** 2)/self.noise_I) - (1 / 2) * np.sum(
            ((self.pix_map[1] - map[1])** 2)/self.noise_Q) - (1 / 2) * np.sum(
            ((self.pix_map[2] - map[2])** 2)/self.noise_Q)

    def compute_log_MH_ratio(self, chol_cls_old, chol_cls_new, dls_old, dls_new, s_nonCentered, l_start, l_end, old_lik):
        new_lik = self.compute_log_likelihood(chol_cls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = self.compute_log_proposal(dls_new, dls_old, l_start, l_end) - self.compute_log_proposal(dls_old,
                                                                                            dls_new, l_start, l_end)
        return part1 + part2, new_lik

    def sample(self, alm_map_non_centered, dls_old, chol_cls_old):
        """
        :param binned_cls_old: binned power spectrum, including monopole and dipole
        :param var_cls_old: variance associated to this power spectrum, including monopole and dipole
        :param alm_map_non_centered: non centered skymap expressed in harmonic domain
        :return: a new sampled power spectrum, using M-H algorithm

        Not that here l_start and l_end are not shifted by -2 because binned_cls_old contains ALL ell, including monopole
        and dipole
        """
        accept = []
        old_lik = self.compute_log_likelihood(chol_cls_old, alm_map_non_centered)
        for i, l_start in enumerate(self.metropolis_blocks[:-1]):
            l_end = self.metropolis_blocks[i + 1]
            for _ in range(self.n_iter):
                dls_new_block, chol_cls_new_block = self.propose_cl(dls_old, l_start, l_end)
                dls_new = dls_old.copy()
                chol_cls_new = chol_cls_old.copy()
                dls_new[l_start:l_end] = dls_new_block
                chol_cls_new[l_start:l_end] = chol_cls_new_block

                log_r, new_lik = self.compute_log_MH_ratio(chol_cls_old, chol_cls_new, dls_old, dls_new,
                                                  alm_map_non_centered, l_start, l_end, old_lik)

                if np.log(np.random.uniform()) < log_r:
                    dls_old = dls_new.copy()
                    chol_cls_old = chol_cls_new.copy()
                    old_lik = new_lik
                    accept.append(1)
                else:
                    accept.append(0)

        return dls_old, chol_cls_old, accept


class NonCenteredGibbs(GibbsSampler):
    def __init__(self, pix_map, noise_I, noise_Q, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1):
        super().__init__(pix_map, noise_I, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        if not polarization:
            self.constrained_sampler = NonCenteredConstrainedRealization(pix_map, noise_I, self.bl_map, lmax, Npix, isotropic=True)
            self.cls_sampler = NonCenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise_I, metropolis_blocks,
                                                     proposal_variances, n_iter = n_iter_metropolis)
        else:
            self.constrained_sampler = PolarizedNonCenteredConstrainedRealization(pix_map, noise_I, noise_Q,
                                                                                  self.bl_map, lmax, Npix, beam, isotropic=True)
            self.cls_sampler = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_I, noise_Q
                                                                 , metropolis_blocks, proposal_variances, n_iter = n_iter_metropolis)

    def run(self, cls_init):
        h_dls = []
        h_acceptance = []
        h_time_seconds = []
        binned_dls = cls_init.copy()
        h_dls.append(binned_dls.copy())
        chol_binned_cls = binned_dls.copy()
        for i in range(2, len(binned_dls)):
            chol_binned_cls[i, :, :] = np.linalg.cholesky(binned_dls[i, :, :]*(2*np.pi)/(i*(i+1)))

        h_dls.append(binned_dls)
        for i in range(self.n_iter):
            if i % 1 == 100:
                print("Default Gibbs, iteration:", i)

            start_time = time.process_time()
            alm_map, time_to_solution, err = self.constrained_sampler.sample(binned_dls.copy(), chol_binned_cls)
            binned_dls, chol_binned_cls, accept = self.cls_sampler.sample(alm_map, binned_dls, chol_binned_cls)
            end_time = time.process_time()
            h_dls.append(binned_dls)
            h_acceptance.append(accept)
            h_time_seconds.append(end_time - start_time)


        print(np.array(h_acceptance).shape)
        print("Non centered Gibbs acceptance rate:", np.mean(np.array(h_acceptance), axis = 0))
        return np.array(h_dls), h_time_seconds