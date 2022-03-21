from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import MHClsSampler
import healpy as hp
import numpy as np
import time
import config
import utils
from scipy.stats import truncnorm
from CenteredGibbs import PolarizedCenteredConstrainedRealization
import qcinv
from copy import deepcopy




class NonCenteredConstrainedRealization(ConstrainedRealization):
    """
     The non centered CR step class for temperature only.
    """

    def sample_no_mask(self, cls_, var_cls):
        """
        make the CR step when no mask is applied. In this case the system is diagonal and we can computee its solution directly.

        :param cls_: array of floats, size (L_max +1,), power spectrum. Useless
        :param var_cls: array of floats, size (L_max+1,)**2. This is the diagonal of the C matrix, see paper.
        :return: array of floats, size Npix. The sampled skymap.
        """
        b_weiner = np.sqrt(var_cls) * self.bl_map * utils.adjoint_synthesis_hp(self.pix_map * self.inv_noise) # Compute the mean part of b
        b_fluctuations = np.random.normal(size=len(var_cls)) + \
                         np.sqrt(var_cls) * self.bl_map * \
                         utils.adjoint_synthesis_hp(np.random.normal(size=self.Npix) * np.sqrt(self.inv_noise)) # compute the fluctuation part of b

        Sigma = 1/(1 + (var_cls / self.noise[0]) * (self.Npix / (4 * np.pi)) * self.bl_map ** 2) #Compute Q^-1 = \Sigma
        weiner = Sigma* b_weiner #Performs Q^-1*b
        flucs = Sigma * b_fluctuations # same
        map = weiner + flucs # add the two parts of the solution
        return map, 1

    def sample_mask(self, cls_, var_cls, s_old, metropolis_step=False):
        """
        This performs CR step when a mask is appied, using either regular PCG or RJPO
        :param cls_: array of floats, size (L_max +1, ) of the power spectrum C_\ell
        :param var_cls: array of floats, size (L_max +1)**2, the diagonal of the signal matrix C. see paper.
        :param s_old: array of float, size (L_max+1)**2, current skymap, in spherical harmonics in real convention.
        :param metropolis_step: boolean, if True, we use a RJPO algorithm. Otherwise just use the classic PCG with diagonal preconditionner.
        :return: array of floats, size (L_max +1)**2, sampled sky mask in sph harmonics, expressed in real convention.
        """
        self.s_cls.cltt = cls_ # set a s_cls object containing the power spectrum and L_max to be used by qcinv.
        self.s_cls.lmax = self.lmax #same
        cl_inv = np.zeros(len(cls_))
        cl_inv[np.where(cls_ !=0)] = 1/cls_[np.where(cls_ != 0)] #Invert the C_\ell.

        inv_var_cls = np.zeros(len(var_cls))
        inv_var_cls[var_cls != 0] = 1/var_cls[var_cls!=0] # Compute C^-1

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=('log_')) # Defining stuffs for qcinv.

        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise)) # Defining the flucuation part of the b part of the system.

        if metropolis_step:
            #If we use a PCG, then set the starting point at the current skymap.
            soltn_complex = -utils.real_to_complex(s_old)[:]
        else:
            #Otherwise set the starting point to 0.
            soltn_complex = np.zeros(int(qcinv.util_alm.lmax2nlm(self.lmax)), dtype=np.complex)

        fluctuations_complex = utils.real_to_complex(b_fluctuations) #Set the fluctuation art into complex convention.
        b_system = chain.sample(soltn_complex, self.pix_map, fluctuations_complex) # Actual resolution of the system.
        #### Since at the end of the solver the output is multiplied by C^-1, it's enough to remultiply it by C^(1/2) to
        ### To produce a non centered map !
        hp.almxfl(soltn_complex, cls_, inplace=True)
        hp.almxfl(soltn_complex, np.sqrt(cl_inv), inplace=True)
        soltn = utils.remove_monopole_dipole_contributions(utils.complex_to_real(soltn_complex))
        if not metropolis_step:
            # If not RJPO, return the solution.
            return soltn, 1
        else:
            #Otherwise, compute the MH ratio and accept with that proba.
            r = b_system - hp.map2alm(self.inv_noise*hp.alm2map(soltn_complex, nside=self.nside, lmax=self.lmax), lmax=self.lmax)\
                                      + hp.almxfl(soltn_complex,cl_inv, inplace=False)*(self.Npix/(4*np.pi))
            r = utils.complex_to_real(r)
            log_proba = min(0, -np.dot(r,(s_old - soltn)))

            if np.log(np.random.uniform()) < log_proba:
                return soltn, 1
            else:
                return s_old, 0

    def sample(self, cls_, var_cls, old_s, metropolis_step=False):
        """
        Make a CR step for TT only.
        """
        if self.mask_path is not None:
            return self.sample_mask(cls_, var_cls, old_s, metropolis_step)
        else:
            return self.sample_no_mask(cls_, var_cls)



class PolarizedNonCenteredConstrainedRealization(ConstrainedRealization):
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, mask_path=None, all_sph = False):
        """
        Class for making a non centered CR step for "EE" and "BB" only.

        :param pix_map: array of floats, size Npix. Observed sky mask.
        :param noise_temp: array of floats, size Npix. Noise level per pixel for intensity.
        :param noise_pol:array of floats, size Npix. Noise level per pixel for Q and U.
        :param bl_map: array of floats, size (L_max +1)**2, diagonal of the diagonal matrix B, see paper.
        :param lmax: integer, L_max
        :param Npix: integer, number of pixels.
        :param bl_fwhm: float, definition of the fwhm of the gaussian beam, in degree
        :param mask_path: string, path of the mask.
        :param all_sph: boolean, whether the model is diagonal and we can write everything in sph.
        """
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.mask_path = mask_path
        self.all_sph = all_sph

        self.bl_fwhm = bl_fwhm
        #The next line defines the CR sampler in centered parametrization.
        self.pol_centered_constraint_realizer = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                            bl_map, lmax, Npix, bl_fwhm, mask_path=mask_path)


    def sample_no_mask(self, all_dls):
        """
        Sampler when no mask is applied.

        :param all_dls: dict {"EE":array, "BB":array} where arrays of floats are size number of bins. D_\ell variables.
        :return: dict {"EE":array, "BB":array} where arrays of floats are of size (L_max+1,)**2
        """
        var_cls_E = utils.generate_var_cl(all_dls["EE"]) # Compute the diagonal of the C matrix. See paper.
        var_cls_B = utils.generate_var_cl(all_dls["BB"]) # Same

        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Invert the variance, computes C^-1

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # Same here

        sigma_E = 1/(1 + self.inv_noise_pol[0]*self.bl_map**2*var_cls_E*self.Npix/(4*np.pi)) # Computes Q^-1
        sigma_B = 1/(1 + self.inv_noise_pol[0] * self.bl_map ** 2 * var_cls_B * self.Npix / (4 * np.pi)) # same.

        if not self.all_sph:
            # If the noise covariance matrix is not prop to diagonal, we have to go to the pixel domain.
            _, r_E, r_B = hp.map2alm([np.zeros(self.Npix),self.pix_map["Q"]*self.inv_noise_pol, self.pix_map["U"]*self.inv_noise_pol],
                           lmax=self.lmax, pol=True)*self.Npix/(4*np.pi)

            r_E = np.sqrt(var_cls_E) * self.bl_map * utils.complex_to_real(r_E)
            r_B = np.sqrt(var_cls_B) * self.bl_map * utils.complex_to_real(r_B)
        else:
            # Otherwise, we can write everything in sph harmonic domain.
            r_E = (self.Npix*self.inv_noise_pol[0]/(4*np.pi))* self.pix_map["EE"]
            r_B = (self.Npix * self.inv_noise_pol[0] /(4 * np.pi)) * self.pix_map["BB"]

            r_E = np.sqrt(var_cls_E) * self.bl_map * r_E
            r_B = np.sqrt(var_cls_B) * self.bl_map * r_B


        mean_E = sigma_E*r_E # Compute the mean of the Gaussian distribution.
        mean_B = sigma_B*r_B # Compute the mean of the Gaussian distribution.


        alms_E = mean_E + np.random.normal(size=len(var_cls_E)) * np.sqrt(sigma_E) # Actual sampling
        alms_B = mean_B + np.random.normal(size=len(var_cls_B)) * np.sqrt(sigma_B) # same.

        return {"EE":alms_E, "BB":alms_B}, 0

    def sample_mask(self, all_dls):
        """
        Sampling when a sky mask is applied.

        :param all_dls: dict {"EE":array, "BB":array} where the arrays contain floats, are of size L_max +1 and contain the D_\ell
        :return: dict {"EE":array, "BB":array} where arrays of floats are of size (L_max+1,)**2, and an integer always 1 since we always accept.
        """
        var_cls_EE = utils.generate_var_cl(all_dls["EE"]) # Computes the variance, the diagonal of the C matrix.
        var_cls_BB = utils.generate_var_cl(all_dls["BB"]) # Same
        inv_var_cls_EE = np.zeros(len(var_cls_EE))
        inv_var_cls_BB = np.zeros(len(var_cls_BB))
        inv_var_cls_EE[var_cls_EE != 0] = 1/var_cls_EE[var_cls_EE != 0] # Computes the inverse of C.
        inv_var_cls_BB[var_cls_BB != 0] = 1 / var_cls_BB[var_cls_BB != 0] # Same.

        alms, _ = self.pol_centered_constraint_realizer.sample_mask(all_dls) # Make a centered CR step.
        alms["EE"] *=np.sqrt(inv_var_cls_EE) #Change parametrization to have a non centered skymap.
        alms["BB"] *=np.sqrt(inv_var_cls_BB) # same.
        return alms, 1


    def sample(self, all_dls):
        if self.mask_path is None:
            return self.sample_no_mask(all_dls)
        else:
            return self.sample_mask(all_dls)


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



class PolarizationNonCenteredClsSampler(MHClsSampler):
    ## Class which makes a Metropolis-within-Gibbs step for sampling the power specturm on the non centered parametrization.
    ## For "EE" and "BB" only.

    def __init__(self, pix_map, lmax, nside, bins, bl_map, noise_I, noise_Q, metropolis_blocks, proposal_variances, n_iter = 1, mask_path=None, polarization=True,
                 all_sph = False):
        """

        :param pix_map: array of floats, of size Npix. The observed skymap called d in the paper.
        :param lmax: integer, L_max
        :param nside: integer, nside used to generate the grid over the sphere.
        :param bins: array of integers, each integer denotes the start and end of a bin.
        :param bl_map: array of floats, diagonal of the matrix of beam B, see paper.
        :param noise_I: array of float, noise levels for each pixel for the intensity component. Ignore this argument for polarization exp
        :param noise_Q: array of float, noise levels for each pixel for the polarization components.
        :param metropolis_blocks: array of integers, each integer is the start and end of a block, see paper.
        :param proposal_variances: array of floats, variance of the proposal for each binned \ell.
        :param n_iter: integer, number of iterations of the Metropolis-within-Gibbs sampler.
        :param mask_path: string, path to a mask.
        :param polarization: bool, whether we deal with polarization or not.
        :param all_sph: bool, True if we have no skymask and the noise covariance matrix is proportional to identity. In
                this case the problem can be written entirely in spherical harmonic domain, avoiding unnnecessary computations.
        """
        super().__init__(pix_map, lmax, nside, bins, bl_map, noise_I, metropolis_blocks, proposal_variances, n_iter = n_iter,
                       polarization=polarization)

        self.nside=nside
        self.noise_temp = noise_I
        self.noise_pol = noise_Q
        self.inv_noise_pol = 1/self.noise_pol
        self.sigma = 0.8
        self.Npix = 12*nside**2
        self.all_sph = all_sph
        self.mask_path = mask_path
        if mask_path is not None:
            #If we did provide a mask, load it and downgrade it.
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise_pol *= self.mask


    def propose_dl(self, dls_old):
        """
        Note that the D_\ell arrays have the monopole and dipole as the two first elements. But we do NOT sample them,
        so we set them at 0.

        :param dls_old: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells which is the current state.
        :return: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells  which is a proposed state.
        """
        ## We se truncated normal distributions because the pow spec cannot be inferior to 0.
        clip_low_EE = -dls_old["EE"][2:] / np.sqrt(self.proposal_variances["EE"])
        dls_EE = np.concatenate([np.zeros(2), truncnorm.rvs(a=clip_low_EE, b=np.inf, loc=dls_old["EE"][2:],
                             scale=np.sqrt(self.proposal_variances["EE"]))])

        clip_low_BB = -dls_old["BB"][2:] / np.sqrt(self.proposal_variances["BB"])
        dls_BB = np.concatenate([np.zeros(2), truncnorm.rvs(a=clip_low_BB, b=np.inf, loc=dls_old["BB"][2:],
                             scale=np.sqrt(self.proposal_variances["BB"]))])

        return {"EE":dls_EE, "BB":dls_BB} # proposed move, including dipole and monopole which are 0.



    def compute_log_proposal(self, dl_old, dl_new):
        """
        evaluates the log proposal into the proposed and current point.

        :param dl_old: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells which is the current state.
        :param dl_new: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells  which is a proposed state.
        :return: dict {"EE":float, "BB":float}, the log evaluation of the proposal.
        """
        ## We don't take into account the monopole and dipole in the computation because we don't change it anyway (we keep them to 0)
        clip_low_EE = -dl_old["EE"][2:] / np.sqrt(self.proposal_variances["EE"])
        proba_EE = np.concatenate([np.zeros(2), truncnorm.logpdf(dl_new["EE"][2:], a=clip_low_EE, b=np.inf, loc=dl_old["EE"][2:],
                                                             scale=np.sqrt(self.proposal_variances["EE"]))])

        clip_low_BB = -dl_old["BB"][2:] / np.sqrt(self.proposal_variances["BB"])
        proba_BB = np.concatenate([np.zeros(2), truncnorm.logpdf(dl_new["BB"][2:], a=clip_low_BB, b=np.inf, loc=dl_old["BB"][2:],
                                                             scale=np.sqrt(self.proposal_variances["BB"]))])

        return {"EE":proba_EE, "BB":proba_BB}


    def compute_log_likelihood(self, dls, s_nonCentered):
        """
        evaluate the log likelihood in the D_\ell when the noise covariance matrix is not proportional to the diagonal
        or if we are masking part of the sky.

        :param dls: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells
        :param s_nonCentered: dict {"EE":array, "BB":array}, where arrays are the alms of the skymap, in real convention.
        :return: float, the log likelihood evaluated in D_\ell
        """
        all_dls = {"EE":utils.unfold_bins(dls["EE"], self.bins["EE"]), "BB":utils.unfold_bins(dls["BB"], self.bins["BB"])}
        var_cls_E = utils.generate_var_cl(all_dls["EE"]) # Compute the diagonal matrix C
        var_cls_B = utils.generate_var_cl(all_dls["BB"]) # same

        #All of the next step compute the quantity -(1/2)*(s C^(1/2) B A^T N^-1 A B C^(1/2) s)
        alm_E_centered = utils.real_to_complex(self.bl_map*np.sqrt(var_cls_E)*s_nonCentered["EE"])
        alm_B_centered = utils.real_to_complex(self.bl_map*np.sqrt(var_cls_B)*s_nonCentered["BB"])

        _, map_Q, map_U = hp.alm2map([utils.real_to_complex(np.zeros(len(var_cls_E))), alm_E_centered,alm_B_centered]
                                     ,lmax=self.lmax, nside=self.nside, pol=True)

        return - (1 / 2) *( np.sum(
            ((self.pix_map["Q"] - map_Q)** 2)*self.inv_noise_pol) + np.sum(
            ((self.pix_map["U"] - map_U)** 2)*self.inv_noise_pol))

    def compute_log_likelihood_all_sph(self,dls, s_nonCentered):
        """
        evaluate the log likelihood in the D_\ell when the noise covariance matrix is proportional to the diagonal
        and we are observing the full sky.

        :param dls: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells
        :param s_nonCentered: dict {"EE":array, "BB":array}, where arrays are the alms of the skymap, in real convention.
        :return: float, the log likelihood evaluated in D_\ell
        """
        all_dls = {"EE":utils.unfold_bins(dls["EE"], self.bins["EE"]), "BB":utils.unfold_bins(dls["BB"], self.bins["BB"])}
        var_cls_E = utils.generate_var_cl(all_dls["EE"])#Computes the diagonal matrix C
        var_cls_B = utils.generate_var_cl(all_dls["BB"])# same

        #All the next steps compute the quantity -(1/2)*(s C^(1/2) B A^T N^-1 A B C^(1/2) s) where this matrix is diagonal.
        alm_E_centered = self.bl_map*np.sqrt(var_cls_E)*s_nonCentered["EE"]
        alm_B_centered = self.bl_map*np.sqrt(var_cls_B)*s_nonCentered["BB"]


        return - (1 / 2) *( np.sum(
            ((self.pix_map["EE"] - alm_E_centered)** 2)*self.inv_noise_pol[0]*self.Npix/(4*np.pi)) + np.sum(
            ((self.pix_map["BB"] - alm_B_centered)** 2)*self.inv_noise_pol[0]*self.Npix/(4*np.pi)))


    def compute_log_MH_ratio(self, log_r_ratio, dls_new, s_nonCentered, old_lik):
        """

        :param log_r_ratio: float, log ratio of the proposal
        :param dls_new: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells  which is a proposed state.
        :param s_nonCentered: dict {"EE":array, "BB":array}, where arrays are the alms of the skymap, in real convention.
        :param old_lik: log likelihood evaluated at the current point dls_old. So we do not have to compute it a second time.
        :return: float, the log Metropolis ratio, and float, the log likelihood evaluated at the proposed state dls_new, so if
                it is accepted later, we will not have to recompute it again.
        """
        if not self.all_sph:
            ## If we can write the problem entirely in Sph basis, then we avoid some computations
            new_lik = self.compute_log_likelihood(dls_new, s_nonCentered)
        else:
            ## Otherwise, use the normal log likelihood computation.
            new_lik = self.compute_log_likelihood_all_sph(dls_new, s_nonCentered)

        part1 = new_lik - old_lik # Computed the log lik ratio
        part2 = log_r_ratio # The second part is the log proposal ratio.
        return part1 + part2, new_lik

    def sample(self, s_nonCentered, binned_dls_old):
        """

        :param s_nonCentered: dict {"EE":array, "BB":array}, where arrays are the alms of the current skymap, in real convention.
        :param binned_dls_old: dict {"EE":array, "BB":array}, where array are float arrays of the binned D_\ells which is the current state.
        :return: dict {"EE":array, "BB":array}, where arrays are float, the new (or old) D_\ell and 1 (or 0) if accepted (or not).
        """
        accept = {"EE": [], "BB":[]}
        binned_dls_propose = self.propose_dl(binned_dls_old) # propose a new state.
        log_prop_num = self.compute_log_proposal(binned_dls_propose, binned_dls_old) # Evaluate the log proposal
        log_prop_denom = self.compute_log_proposal(binned_dls_old, binned_dls_propose) # Same
        # Compute the ratio of log proposal for the MH step. We can do it once and for all, since the proposal between D_\ell are independent.
        log_r_proposal_all = {"EE": log_prop_num["EE"] - log_prop_denom["EE"], "BB":log_prop_num["BB"] - log_prop_denom["BB"]}
        if not self.all_sph:
            # If we can write the model in sph basis entirerly, then we save some computations.
            old_lik = self.compute_log_likelihood(binned_dls_old, s_nonCentered)
        else:
            # Otherwise, we do not
            old_lik = self.compute_log_likelihood_all_sph(binned_dls_old, s_nonCentered)

        for pol in ["EE", "BB"]:
            #For each polarization components:
            for i, l_start in enumerate(self.metropolis_blocks[pol][:-1]):
                #For each Metropolis block, find the \ell of start, l_start, then \ell of end of the block, l_end.
                l_end = self.metropolis_blocks[pol][i + 1]

                for _ in range(self.n_iter):
                    ### For the a specific number of iterations of MH algorithm.
                    ###Be careful, the monopole and dipole are not included in the log_r_prior
                    binned_dls_new = deepcopy(binned_dls_old)
                    binned_dls_new[pol][l_start:l_end] = binned_dls_propose[pol][l_start:l_end].copy() # We propose a new state.
                    log_r_all = np.sum(log_r_proposal_all[pol][l_start:l_end]) # We sum the log ratio of proposal over the \ell being part of the current block
                    log_r, new_lik = self.compute_log_MH_ratio(log_r_all, binned_dls_new, s_nonCentered, old_lik) # Computes the log MH ratio
                    if np.log(np.random.uniform()) < log_r:
                        #If acceptance, then we will output the proposed state and we keep the log likelihood evaluated in
                        #this new state. Since we had to compute it after proposing it, we keep it in memory so we won't have
                        # to compute it again. This halves the computational cost of this algorithm.
                        binned_dls_old = deepcopy(binned_dls_new)
                        old_lik = new_lik
                        accept[pol].append(1)
                    else:
                        accept[pol].append(0)


        return binned_dls_old, accept



class NonCenteredGibbs(GibbsSampler):
    def __init__(self, pix_map, noise_I, noise_Q, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1, mask_path=None, all_sph = False):
        """

        :param pix_map: array of floats, size Npix, observed skymap d.
        :param noise_I: array of floats, size Npix, noise level for each pixel
        :param noise_Q: array of floats, size Npix, noise level for each pixel. Same for Q and U maps
        :param beam: float, definition of the beam in degree.
        :param nside: integer, nside used to generate the grid over the sphere.
        :param lmax: integer, L_max
        :param Npix: integer, number of pixels
        :param proposal_variances: dict {"EE": array, "BB":array}, arrays of float, variances of the proposal distributions. NOT of size L_max, but of size number of blocks.
        :param metropolis_blocks: dict {"EE":array, "BB":array}, integers, starting and ending indexes of the blocks.
        :param polarization: boolean, whether we are dealing with "TT" only or "EE" and "BB" only.
        :param bins: dict {"EE":array, "BB":array}, integers, starting and ending indexes of the bins.
        :param n_iter: integer, number of iterations of the Gibbs sampler to do.
        :param n_iter_metropolis: integer, number of iterations of the Metropolis-within-Gibbs sampler to do.
        :param mask_path: string, path to a sky mask. If None, no sky mask is used.
        :param all_sph: boolean, if True, write the entire model in spherical harmonics basis. If False do as usual.
        """
        super().__init__(pix_map, noise_I, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        if not polarization:
            #Defines CR sampler for "TT" only:
            self.constrained_sampler = NonCenteredConstrainedRealization(pix_map, noise_I, self.bl_map, beam, lmax, Npix, isotropic=True,
                                                                         mask_path = mask_path)
            ##Defines the M-w-G sampler for "TT" only.
            self.cls_sampler = NonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_I, metropolis_blocks,
                                                     proposal_variances, n_iter = n_iter_metropolis, mask_path=mask_path)
        else:
            # Defines CR sampler for "EE" and "BB" only:
            self.constrained_sampler = PolarizedNonCenteredConstrainedRealization(pix_map, noise_I, noise_Q,
                                                                                  self.bl_map, lmax, Npix, beam,
                                                                                  mask_path=mask_path, all_sph=all_sph)
            ##Defines the M-w-G sampler for "EE" and "BB" only.
            self.cls_sampler = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_I, noise_Q
                                                                 , metropolis_blocks, proposal_variances, n_iter = n_iter_metropolis,
                                                                 mask_path=mask_path, all_sph=all_sph)

    def run_temperature(self, dl_init):
        """
        Run the Gibbs sampler for the temperature only.

        :param dl_init: array of float, initial D_\ell
        :return: array of size (n_iter, L_max +1) being the trajectory of the Gibbs sampler. Array of int of size (n_iter)
                being the history of acceptances of the M-w-G algo. Array of floats, size (n_iter, blocks), history of execution times.
        """
        h_time_seconds = []
        total_accept = []

        binned_dls = dl_init
        dls = utils.unfold_bins(dl_init, config.bins) # Unfold the binned D_\ell, that transforming the array of size number of bins to
        # an array of size L_max +1 .
        cls = self.dls_to_cls(dls) #convert to C_\ell

        h_dl = []
        var_cls = utils.generate_var_cl(dls) # Generate the diagonal matrix C, see paper.
        for i in range(self.n_iter):
            if i % 100== 0:
                print("Non centered gibbs")
                print(i)

            start_time = time.process_time()
            s_nonCentered, accept = self.constrained_sampler.sample(cls, var_cls, None, False) ## Make a CR step.
            #The next step used a M-w-G samper and outputs the binned sample D_/ell as well as its corresponding diagonal matrix C, see paper.
            binned_dls, var_cls, accept = self.cls_sampler.sample(s_nonCentered, binned_dls, var_cls)
            dls = utils.unfold_bins(binned_dls, self.bins) # unfold binned D_\ell
            cls = self.dls_to_cls(dls) # compute the corresponding C_\ell
            total_accept.append(accept)

            end_time = time.process_time()
            h_dl.append(binned_dls)
            h_time_seconds.append(end_time - start_time)

        total_accept = np.array(total_accept)
        print("Non centered acceptance rate:")
        print(np.mean(total_accept, axis=0))

        return np.array(h_dl), total_accept, np.array(h_time_seconds)

    def run_polarization(self, dls_init):
        """
        Non centered Gibbs for "EE" and "BB" only.

        :param dls_init: dict {"EE":array, "BB":array}, arrays of float, initial D_\ell
        :return: array of size (n_iter, L_max +1) being the trajectory of the Gibbs sampler. Array of int of size (n_iter, n_blocks)
                being the history of acceptances of the M-w-G algo for each block. Array of floats, history of execution times.
                Array of floats, history of the execution times of the M-w-G sampler.
        """
        h_duration_cr = []
        h_duration_cls_sampling = []
        total_accept = {"EE":[], "BB":[]}
        h_dls = {"EE":[], "BB":[]}
        binned_dls = dls_init
        dls_unbinned = {"EE":utils.unfold_bins(binned_dls["EE"], self.bins["EE"]), "BB":utils.unfold_bins(binned_dls["BB"], self.bins["BB"])}
        h_dls["EE"].append(binned_dls["EE"])
        h_dls["BB"].append(binned_dls["BB"])
        for i in range(self.n_iter):
            if i % 1== 0:
                print("Non centered gibbs")
                print(i)

            s_nonCentered, _ = self.constrained_sampler.sample(dls_unbinned) # Do a CR step

            binned_dls, accept = self.cls_sampler.sample(s_nonCentered, binned_dls) # Sample the power spectrum with M-w-G
            dls_unbinned["EE"] = utils.unfold_bins(binned_dls["EE"].copy(), self.bins["EE"]) # Unbin the D_\ell
            dls_unbinned["BB"] = utils.unfold_bins(binned_dls["BB"].copy(), self.bins["BB"]) # same
            total_accept["EE"].append(accept["EE"])
            total_accept["BB"].append(accept["BB"])

            h_dls["EE"].append(binned_dls["EE"])
            h_dls["BB"].append(binned_dls["BB"])

        total_accept = {"EE": np.array(total_accept["EE"]), "BB": np.array(total_accept["BB"])}
        print("Non Centered gibbs acceptance rate EE:")
        print(np.mean(total_accept["EE"], axis=0))
        print("Non Centered Gibbs rate BB:")
        print(np.mean(total_accept["BB"], axis=0))

        h_dls["EE"] = np.array(h_dls["EE"])
        h_dls["BB"] = np.array(h_dls["BB"])

        return h_dls, total_accept, np.array(h_duration_cr), np.array(h_duration_cls_sampling)

    def run(self, dls_init):
        """

        :param dls_init: dict {"EE":array, "BB":array}, arrays of float, initial D_\ell
        :return: whatever the called methods return.
        """
        if not self.polarization:
            return self.run_temperature(dls_init)
        else:
            return self.run_polarization(dls_init)