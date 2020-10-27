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
import qcinv
from oldNonCenteredGibbs import *




class NonCenteredConstrainedRealization(ConstrainedRealization):

    def sample_no_mask(self, cls_, var_cls):
        b_weiner = np.sqrt(var_cls) * self.bl_map * utils.adjoint_synthesis_hp(self.pix_map * self.inv_noise)
        b_fluctuations = np.random.normal(size=len(var_cls)) + \
                         np.sqrt(var_cls) * self.bl_map * \
                         utils.adjoint_synthesis_hp(np.random.normal(size=self.Npix) * np.sqrt(self.inv_noise))

        Sigma = 1/(1 + (var_cls / self.noise[0]) * (self.Npix / (4 * np.pi)) * self.bl_map ** 2)
        weiner = Sigma* b_weiner
        flucs = Sigma * b_fluctuations
        map = weiner + flucs
        map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0

        return map, 1

    def sample_mask(self, cls_, var_cls, s_old, metropolis_step=False):
        self.s_cls.cltt = cls_
        self.s_cls.lmax = self.lmax
        cl_inv = np.zeros(len(cls_))
        cl_inv[np.where(cls_ !=0)] = 1/cls_[np.where(cls_ != 0)]

        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=('log_'))

        #b_weiner = self.bl_map * utils.adjoint_synthesis_hp(self.inv_noise * self.pix_map)
        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise))

        ####THINK ABOUT CHECKING THE STARTING POINT
        if metropolis_step:
            soltn_complex = -utils.real_to_complex(s_old)[:]
        else:
            soltn_complex = np.zeros(int(qcinv.util_alm.lmax2nlm(self.lmax)), dtype=np.complex)

        fluctuations_complex = utils.real_to_complex(b_fluctuations)
        b_system = chain.sample(soltn_complex, self.pix_map, fluctuations_complex)
        #### Since at the end of the solver the output is multiplied by C^-1, it's enough to remultiply it by C^(1/2) to
        ### To produce a non centered map !
        hp.almxfl(soltn_complex, cls_, inplace=True)
        hp.almxfl(soltn_complex, np.sqrt(cl_inv), inplace=True)
        soltn = utils.remove_monopole_dipole_contributions(utils.complex_to_real(soltn_complex))
        if not metropolis_step:
            return soltn, 1
        else:
            r = b_system - hp.map2alm(self.inv_noise*hp.alm2map(soltn_complex, nside=self.nside, lmax=self.lmax), lmax=self.lmax)\
                                      + hp.almxfl(soltn_complex,cl_inv, inplace=False)*(self.Npix/(4*np.pi))
            r = utils.complex_to_real(r)
            log_proba = min(0, -np.dot(r,(s_old - soltn)))
            log_proba2 = min(0, np.dot(r,(s_old - soltn)))
            print("log Probas")
            print(log_proba)
            print(log_proba2)

            if np.log(np.random.uniform()) < log_proba:
                return soltn, 1
            else:
                return s_old, 0

    def sample(self, cls_, var_cls, old_s, metropolis_step=False):
        #if self.mask_path is not None:
        if True:
            return self.sample_mask(cls_, var_cls, old_s, metropolis_step)
        else:
            return self.sample_no_mask(cls_, var_cls)



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
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix, isotropic=True)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        #self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([self.inv_noise_temp*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
        #                                self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
        #                                self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2], axis = 1)

        self.bl_fwhm = bl_fwhm
        self.pol_centered_constraint_realizer = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                            bl_map, lmax, Npix, bl_fwhm, isotropic=True)

    def sample_no_mask(self, all_dls):
        var_cls_E = utils.generate_var_cl(all_dls["EE"])
        var_cls_B = utils.generate_var_cl(all_dls["BB"])

        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0]

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0]

        sigma_E = 1/(1 + self.inv_noise_pol[0]*self.bl_map**2*var_cls_E*self.Npix/(4*np.pi))
        sigma_B = 1/(1 + self.inv_noise_pol[0] * self.bl_map ** 2 * var_cls_B * self.Npix / (4 * np.pi))


        _, r_E, r_B = hp.map2alm([np.zeros(self.Npix),self.pix_map["Q"]*self.inv_noise_pol, self.pix_map["U"]*self.inv_noise_pol],
                       lmax=self.lmax, pol=True)*self.Npix/(4*np.pi)

        r_E = np.sqrt(var_cls_E)*self.bl_map* utils.complex_to_real(r_E)
        r_B = np.sqrt(var_cls_B) * self.bl_map * utils.complex_to_real(r_B)

        mean_E = sigma_E*r_E
        mean_B = sigma_B*r_B

        alms_E = mean_E + np.random.normal(size=len(var_cls_E)) * np.sqrt(sigma_E)
        alms_B = mean_B + np.random.normal(size=len(var_cls_B)) * np.sqrt(sigma_B)

        return {"EE":alms_E, "BB":alms_B}, 0

    def sample(self, all_dls):
        return self.sample_no_mask(all_dls)


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
    def compute_log_proposal(self, dl_old, dl_new):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low = -dl_old[2:] / np.sqrt(self.proposal_variances)
        return np.concatenate([np.zeros(2), truncnorm.logpdf(dl_new[2:], a=clip_low, b=np.inf, loc=dl_old[2:],
                                   scale=np.sqrt(self.proposal_variances))])

    def sample(self, s_nonCentered, binned_dls_old, var_cls_old):
        """
        :param binned_dls_old: binned power spectrum, including monopole and dipole
        :param var_cls_old: variance associated to this power spectrum, including monopole and dipole
        :param alm_map_non_centered: non centered skymap expressed in harmonic domain
        :return: a new sampled power spectrum, using M-H algorithm

        Not that here l_start and l_end are not shifted by -2 because binned_cls_old contains ALL ell, including monopole
        and dipole
        """
        accept = []
        binned_dls_propose = self.propose_dl(binned_dls_old)
        log_r_prior_all = self.compute_log_proposal(binned_dls_propose, binned_dls_old) -\
                          self.compute_log_proposal(binned_dls_old, binned_dls_propose)
        old_lik = self.compute_log_likelihood(var_cls_old, s_nonCentered)
        for i, l_start in enumerate(self.metropolis_blocks[:-1]):
            l_end = self.metropolis_blocks[i + 1]

            for _ in range(self.n_iter):
                ###Be careful, the monopole and dipole are not included in the log_r_prior
                binned_dls_new = binned_dls_old.copy()
                binned_dls_new[l_start:l_end] = binned_dls_propose[l_start:l_end]
                dls_new = utils.unfold_bins(binned_dls_new, config.bins)
                var_cls_new = utils.generate_var_cl(dls_new)

                log_r_all = np.sum(log_r_prior_all[l_start:l_end])
                log_r, new_lik = self.compute_log_MH_ratio(log_r_all, var_cls_new,
                                                           s_nonCentered, old_lik)
                if np.log(np.random.uniform()) < log_r:
                    binned_dls_old = binned_dls_new
                    var_cls_old = var_cls_new
                    old_lik = new_lik
                    accept.append(1)
                else:
                    accept.append(0)

        return binned_dls_old, var_cls_old, accept

    """
    def compute_log_proposal(self, dl_old, dl_new):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low = -dl_old[2:] / np.sqrt(self.proposal_variances)
        return np.sum(truncnorm.logpdf(dl_new[2:], a=clip_low, b=np.inf, loc=dl_old[2:],
                                   scale=np.sqrt(self.proposal_variances)))
    """

    """
    def sample(self, s_nonCentered, binned_dls_old, var_cls_old):
        :param binned_dls_old: binned power spectrum, including monopole and dipole
        :param var_cls_old: variance associated to this power spectrum, including monopole and dipole
        :param alm_map_non_centered: non centered skymap expressed in harmonic domain
        :return: a new sampled power spectrum, using M-H algorithm

        Not that here l_start and l_end are not shifted by -2 because binned_cls_old contains ALL ell, including monopole
        and dipole
        accept = []
        old_lik = self.compute_log_likelihood(var_cls_old, s_nonCentered)
        for i, l_start in enumerate(self.metropolis_blocks[:-1]):
            l_end = self.metropolis_blocks[i + 1]

            for _ in range(self.n_iter):
                binned_dls_new_block = self.propose_dl(binned_dls_old, l_start, l_end)
                binned_dls_new = binned_dls_old.copy()
                binned_dls_new[l_start:l_end] = binned_dls_new_block
                dls_new = utils.unfold_bins(binned_dls_new, config.bins)
                var_cls_new = utils.generate_var_cl(dls_new)

                log_r, new_lik = self.compute_log_MH_ratio(binned_dls_old, binned_dls_new, var_cls_new,
                                                           s_nonCentered, old_lik)

                if np.log(np.random.uniform()) < log_r:
                    binned_dls_old = binned_dls_new
                    var_cls_old = var_cls_new
                    old_lik = new_lik
                    accept.append(1)
                else:
                    accept.append(0)

        return binned_dls_old, var_cls_old, accept
        """

class PolarizationNonCenteredClsSampler(MHClsSampler):
    def __init__(self, pix_map, lmax, nside, bins, bl_map, noise_I, noise_Q, metropolis_blocks, proposal_variances, n_iter = 1, polarization=True):
        super().__init__(pix_map, lmax, nside, bins, bl_map, noise_I, metropolis_blocks, proposal_variances, n_iter = n_iter,
                       polarization=polarization)
        self.nside=nside
        self.noise_temp = noise_I
        self.noise_pol = noise_Q
        self.inv_noise_pol = 1/self.noise_pol

    def propose_dl(self, dls_old, l_start, l_end, pol):
        """
        :param dls_old: old dls sample or if polarization mode on, coefficients of the lower triang chol matrix
        :param l_start: starting index of the block
        :param l_end: ending index (not included) of the block
        :return: propose dls

        Note that every index is shifted by -2: the first l_start is 2 - since we are not samplint l=0,1 - and the
        proposal variance also starts at l = 2. But then we need to take the first element of this array, hence setting
        l_start - 2:l_end - 2
        """
        clip_low_pol = -dls_old[pol][l_start:l_end] / np.sqrt(self.proposal_variances[pol][l_start-2:l_end-2])
        return truncnorm.rvs(a=clip_low_pol, b=np.inf, loc=dls_old[pol][l_start:l_end],
                             scale=np.sqrt(self.proposal_variances[pol][l_start-2:l_end-2]))


    def compute_log_proposal(self, dl_old, dl_new, l_start, l_end):
    ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low_EE = -dl_old["EE"][l_start:l_end] / np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2])
        probas_EE = truncnorm.logpdf(dl_new["EE"][l_start:l_end], a=clip_low_EE, b=np.inf, loc=dl_old["EE"][l_start:l_end],
                                 scale=np.sqrt(self.proposal_variances["EE"][l_start - 2:l_end - 2]))

        clip_low_BB = -dl_old["EE"][l_start:l_end]/np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2])
        probas_bb = truncnorm.logpdf(dl_new["BB"][l_start:l_end], a=clip_low_BB, b=np.inf, loc=dl_old["BB"][l_start:l_end],
                                 scale=np.sqrt(self.proposal_variances["BB"][l_start - 2:l_end - 2]))

        return np.sum(probas_EE) + np.sum(probas_bb)

    def compute_log_likelihood(self, dls, s_nonCentered):
        all_dls = {"EE":utils.unfold_bins(dls["EE"], self.bins["EE"]), "BB":utils.unfold_bins(dls["BB"], self.bins["BB"])}
        var_cls_E = utils.generate_var_cl(all_dls["EE"])
        var_cls_B = utils.generate_var_cl(all_dls["BB"])
        alm_E_centered = utils.real_to_complex(np.sqrt(var_cls_E)*s_nonCentered["EE"])
        alm_B_centered = utils.real_to_complex(np.sqrt(var_cls_B)*s_nonCentered["BB"])

        _, map_Q, map_U = hp.alm2map([utils.real_to_complex(np.zeros(len(var_cls_E))), alm_E_centered,alm_B_centered]
                                     ,lmax=self.lmax, nside=self.nside, pol=True)

        return - (1 / 2) * np.sum(
            ((self.pix_map["Q"] - map_Q)** 2)*self.inv_noise_pol) - (1 / 2) * np.sum(
            ((self.pix_map["U"] - map_U)** 2)*self.inv_noise_pol)

    def compute_log_MH_ratio(self, dls_old, dls_new, s_nonCentered, l_start, l_end, old_lik):
        new_lik = self.compute_log_likelihood(dls_new, s_nonCentered)
        part1 = new_lik - old_lik
        part2 = self.compute_log_proposal(dls_new, dls_old, l_start, l_end) - self.compute_log_proposal(dls_old,
                                                                                            dls_new, l_start, l_end)
        return part1 + part2, new_lik

    def sample(self, alm_map_non_centered, dls_old):
        """
        :param binned_cls_old: binned power spectrum, including monopole and dipole
        :param var_cls_old: variance associated to this power spectrum, including monopole and dipole
        :param alm_map_non_centered: non centered skymap expressed in harmonic domain
        :return: a new sampled power spectrum, using M-H algorithm

        Not that here l_start and l_end are not shifted by -2 because binned_cls_old contains ALL ell, including monopole
        and dipole
        """

        accept = []
        old_lik = self.compute_log_likelihood(dls_old, alm_map_non_centered)
        for pol in ["EE", "BB"]:
            for i, l_start in enumerate(self.metropolis_blocks[pol][:-1]):
                l_end = self.metropolis_blocks[pol][i + 1]
                for _ in range(self.n_iter):
                    dls_new_block = self.propose_dl(dls_old, l_start, l_end, pol)
                    dls_new = dls_old.copy()
                    dls_new[pol][l_start:l_end] = dls_new_block

                    log_r, new_lik = self.compute_log_MH_ratio(dls_old, dls_new,
                                                      alm_map_non_centered, l_start, l_end, old_lik)

                    if np.log(np.random.uniform()) < log_r:
                        dls_old = dls_new.copy()
                        old_lik = new_lik
                        accept.append(1)
                    else:
                        accept.append(0)



        return dls_old, accept



class NonCenteredGibbs(GibbsSampler):
    def __init__(self, pix_map, noise_I, noise_Q, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1, mask_path=None):
        super().__init__(pix_map, noise_I, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        if not polarization:
            self.constrained_sampler = NonCenteredConstrainedRealization(pix_map, noise_I, self.bl_map, beam, lmax, Npix, isotropic=True,
                                                                         mask_path = mask_path)
            self.cls_sampler = NonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_I, metropolis_blocks,
                                                     proposal_variances, n_iter = n_iter_metropolis, mask_path=mask_path)
        else:
            self.constrained_sampler = PolarizedNonCenteredConstrainedRealization(pix_map, noise_I, noise_Q,
                                                                                  self.bl_map, lmax, Npix, beam, isotropic=True)
            self.cls_sampler = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_I, noise_Q
                                                                 , metropolis_blocks, proposal_variances, n_iter = n_iter_metropolis)



    def run_temperature(self, dl_init):
        h_time_seconds = []
        total_accept = []

        binned_dls = dl_init
        dls = utils.unfold_bins(dl_init, config.bins)
        cls = self.dls_to_cls(dls)

        h_dl = []
        var_cls = utils.generate_var_cl(dls)
        for i in range(self.n_iter):
            if i % 100== 0:
                print("Non centered gibbs")
                print(i)

            start_time = time.process_time()
            s_nonCentered, accept = self.constrained_sampler.sample(cls, var_cls, None, False)
            binned_dls, var_cls, accept = self.cls_sampler.sample(s_nonCentered, binned_dls, var_cls)
            dls = utils.unfold_bins(binned_dls, self.bins)
            cls = self.dls_to_cls(dls)
            total_accept.append(accept)

            end_time = time.process_time()
            h_dl.append(binned_dls)
            h_time_seconds.append(end_time - start_time)

        total_accept = np.array(total_accept)
        print("Non centered acceptance rate:")
        print(np.mean(total_accept, axis=0))

        return np.array(h_dl), total_accept, np.array(h_time_seconds)

    def run_polarization(self, dls_init):
        total_accept = []
        h_dls = {"EE":[], "BB":[]}
        h_time_seconds = []
        binned_dls = dls_init
        dls_unbinned = {"EE":utils.unfold_bins(binned_dls["EE"].copy(), self.bins["EE"]), "BB":utils.unfold_bins(binned_dls["BB"].copy(), self.bins["BB"])}
        skymap, accept = self.constrained_sampler.sample(dls_unbinned.copy())
        h_dls["EE"].append(binned_dls["EE"].copy())
        h_dls["BB"].append(binned_dls["BB"].copy())
        for i in range(self.n_iter):
            if i % 100== 0:
                print("Non centered gibbs")
                print(i)

            start_time = time.process_time()
            s_nonCentered, _ = self.constrained_sampler.sample(dls_unbinned.copy())
            binned_dls, accept = self.cls_sampler.sample(s_nonCentered.copy(), binned_dls.copy())
            dls_unbinned["EE"] = utils.unfold_bins(binned_dls["EE"].copy(), self.bins["EE"].copy())
            dls_unbinned["BB"] = utils.unfold_bins(binned_dls["BB"].copy(), self.bins["BB"].copy())
            total_accept.append(accept)

            end_time = time.process_time()
            h_dls["EE"].append(binned_dls["EE"].copy())
            h_dls["BB"].append(binned_dls["BB"].copy())
            h_time_seconds.append(end_time - start_time)

        total_accept = np.array(total_accept)
        print("Non centered acceptance rate:")
        print(np.mean(total_accept, axis=0))

        h_dls["EE"] = np.array(h_dls["EE"])
        h_dls["BB"] = np.array(h_dls["BB"])

        return h_dls, total_accept, np.array(h_time_seconds)

    def run(self, dls_init):
        if not self.polarization:
            return self.run_temperature(dls_init)
        else:
            return self.run_polarization(dls_init)