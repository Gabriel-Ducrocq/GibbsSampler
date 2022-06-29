from GibbsSampler import GibbsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import ClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma, invwishart
from scipy.stats import t as student
import time
import config
import scipy
import matplotlib.pyplot as plt
import qcinv
import copy
import warnings

warnings.simplefilter('always', UserWarning)



class CenteredClsSampler(ClsSampler):
    ## Sampler of the power spectrum for the centered Gibbs sampler and temperature only.

    def sample(self, alms):
        """
        :param alms: array of floats, alm skymap in m major.
        :return: Sample each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        alms_complex = utils.real_to_complex(alms) ## Change alms from real to complex
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax) ## Get the empirical power spectrum.
        exponent = np.array([(2 * l + 1) / 2 for l in range(self.lmax +1)])
        binned_betas = []
        binned_alphas = []
        betas = np.array([(2 * l + 1) * l * (l + 1) * (observed_Cl / (4 * np.pi)) for l, observed_Cl in
                          enumerate(observed_Cls)])

        for i, l in enumerate(self.bins[:-1]):
            ## For each bin \ell, computed the alpha, beta and exponent terms of the inverse gamma distribution
            somme_beta = np.sum(betas[l:self.bins[i + 1]])
            somme_exponent = np.sum(exponent[l:self.bins[i + 1]])
            alpha = somme_exponent - 1
            binned_alphas.append(alpha)
            binned_betas.append(somme_beta)

        binned_alphas[0] = 1
        sampled_dls = binned_betas * invgamma.rvs(a=binned_alphas) ##Actual sampling
        sampled_dls[:2] = 0  ## We do NOT infer the monopole and dipole and they are assumed to be 0.
        return sampled_dls


class PolarizedCenteredClsSampler(ClsSampler):
    ## Same as above but for "EE" and "BB" only. We sample first "EE" Pow spec and then "BB"

    def sample_one_pol(self, alms_complex, pol="EE"):
        """
        Sample either "EE" or "BB" polarrization
        :param alms: array of floats, alm skymap in m major, for the polarization "pol"
        :param pol: what polarization power spectrum to be sampled.
        :return: array of floats, size number of bins. Each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax)
        exponent = np.array([(2 * l + 1) / 2 for l in range(self.lmax + 1)])
        binned_betas = []
        binned_alphas = []
        betas = np.array([(2 * l + 1) * l * (l + 1) * (observed_Cl / (4 * np.pi)) for l, observed_Cl in
                          enumerate(observed_Cls)])

        for i, l in enumerate(self.bins[pol][:-1]):
            ##For each bin \ell, compute the alpha, beta and exponent of the inverse gamma distribution
            somme_beta = np.sum(betas[l:self.bins[pol][i + 1]])
            somme_exponent = np.sum(exponent[l:self.bins[pol][i + 1]])
            alpha = somme_exponent - 1
            binned_alphas.append(alpha)
            binned_betas.append(somme_beta)

        binned_alphas[0] = 1
        sampled_dls = binned_betas * invgamma.rvs(a=binned_alphas) # Acutal sampling.
        sampled_dls[:2] = 0 # We do NOT infer the monopole and dipole and they are assumed to be 0.
        return sampled_dls

    def sample(self, alms):
        """

        :param alms: array of float, size (L_max + 1)**2, representing the real and imaginary parts of the alm coeffs
        :return: dict of arrays of floats, size number of bins, of the polarization pow spec.
        """
        alms_EE_complex = utils.real_to_complex(alms["EE"]) #Turn from real to complex
        alms_BB_complex = utils.real_to_complex(alms["BB"]) #Idem

        binned_dls_EE = self.sample_one_pol(alms_EE_complex, "EE") #Sampling EE pow spec
        binned_dls_BB = self.sample_one_pol(alms_BB_complex, "BB") #Sampling BB pow spec

        return {"EE":binned_dls_EE, "BB":binned_dls_BB}









class CenteredConstrainedRealization(ConstrainedRealization):
    ##Class performing Constrained Realization step for TT only dataset.
    ##Note that we ALWAYS assume a noise covariance matrix proportional to the identity matrix. On top of this, we may or
    ##may not apply a skymask.

    def sample_no_mask(self, var_cls):
        """

        :param cls_: array of floats, size (L_max + 1), of the C_\ell
        :param var_cls: array of floatts, size (L_max + 1)**2, of the variances of the real and imaginary parts of alms coeffs.
                Tese variances are C_\ell/2 execpt for m != 0 and C_\ell for m = 0.
        :return: array of floats, size (L_max+1)**2, alms of the sampled skymap, with real and imaginary parts.
        """
        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls) ##Inverting the variance of the alms, except
        ##where this variance is 0, i.e for \ell = 0 and \ell = 1.
        b_weiner = self.bl_map * utils.adjoint_synthesis_hp(self.inv_noise * self.pix_map) #Computing the mean part of the solution of the system.
        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise)) # Computing the variance part of the solution of the system.

        Sigma = 1 / (inv_var_cls + self.inv_noise[0] * (self.Npix / (4 * np.pi)) * self.bl_map ** 2) # Matrix Sigma = Q^-1 of the system.
        #The three next step actually solves the system, which is diagonal since we assume that the noise covariance matrix is prop to
        # thee Identity matrix and no sky mask.
        weiner = Sigma * b_weiner
        flucs = Sigma * b_fluctuations
        map = weiner + flucs

        #returns the sampled map and 1, indicating we always accept the output of this algorithm..
        return map, 1


    def sample_mask(self, cls_, var_cls, s_old, metropolis_step=False):
        """

        :param cls_: array of floats, size (L_max + 1), of the C_\ell
        :param var_cls: array of floatts, size (L_max + 1)**2, of the variances of the real and imaginary parts of alms coeffs.
                Tese variances are C_\ell/2 execpt for m != 0 and C_\ell for m = 0.
        :param s_old: array of floats, size (L_max + 1)**2, real and imaginary parts of the alms of the map of the previous iteration.
        :param metropolis_step: boolean. True if we use a RJPO step, False if we just use a regular PCG sampler.
        :return: array of floats, size (L_max+1)**2, alms of the sampled skymap, with real and imaginary parts.
        """
        self.s_cls.cltt = cls_ #Creating the object s_cls that we have to give to qcinv code.
        self.s_cls.lmax = self.lmax
        cl_inv = np.zeros(len(cls_))
        cl_inv[np.where(cls_ !=0)] = 1/cls_[np.where(cls_ != 0)]#Inversion of the C_\ell, except where it is zero.

        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)#Inversion of the variance, i.e computing
        #C^-1.

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None)#Definition of a qcinv object.

        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise))#Computing the flucttuations part of the b of the system


        if metropolis_step:
            #If using a RJPO algorithm, change the starting point.
            soltn_complex = -utils.real_to_complex(s_old)[:]
        else:
            #Otherwise, keep it to zero.
            soltn_complex = np.zeros(int(qcinv.util_alm.lmax2nlm(self.lmax)), dtype=np.complex)


        fluctuations_complex = utils.real_to_complex(b_fluctuations)#Turn the fluctuations to complex for qcinv resolution.
        b_system = chain.sample(soltn_complex, self.pix_map, fluctuations_complex)#Actual solving, with solution in soltn_complex and  b_system is
        #the b of the system: weiner (computed by qcinv) + fluctuations part (that we computed above)/
        soltn = utils.complex_to_real(soltn_complex) #Turn to real.
        if not metropolis_step:
            #If regular PCG resolution, just output the sampled skymap and 1 indicating we always accept
            return soltn, 1
        else:
            #If using a RJPO step, compute the MH-ratio and accept with proba min(MH-ratio, 1)
            approx_sol_complex = hp.almxfl(hp.map2alm(hp.alm2map(hp.almxfl(soltn_complex, self.bl_gauss), nside=self.nside)*self.inv_noise, lmax=self.lmax)
                                   *self.Npix/(4*np.pi), self.bl_gauss) + hp.almxfl(soltn_complex, cl_inv, inplace=False)
            r = b_system - approx_sol_complex
            r = utils.complex_to_real(r)
            log_proba = min(0, -np.dot(r,(s_old - soltn)))
            print("log Proba")
            print(log_proba)
            if np.log(np.random.uniform()) < log_proba:
                #If we accept the solution, we output it and return 1
                return soltn, 1
            else:
                #Otheerwise we output the previous map and return 0
                return s_old, 0

    def sample_gibbs_change_variable(self, var_cls, old_s):
        """
        This function samples the skymap using the auxiliary variable scheme.

        :param var_cls: array of floatts, size (L_max + 1)**2, of the variances of the real and imaginary parts of alms coeffs.
                Tese variances are C_\ell/2 execpt for m != 0 and C_\ell for m = 0.
        :param old_s: array of floats, size (L_max + 1)**2, real and imaginary parts of the alms of the map of the previous iteration.
        :return: array of floats, size (L_max+1)**2, alms of the sampled skymap, with real and imaginary parts.
        """
        old_s = utils.real_to_complex(old_s)
        var_v = self.mu - self.inv_noise #Computes gamma
        mean_v = var_v * hp.alm2map(hp.almxfl(old_s, self.bl_gauss), nside=self.nside, lmax=self.lmax)
        v = np.random.normal(size=len(mean_v))*np.sqrt(var_v) + mean_v #Sample the auxiliary variable v

        inv_var_cls = np.zeros(len(var_cls))
        inv_var_cls[np.where(var_cls != 0)] = 1/var_cls[np.where(var_cls != 0)]
        var_s = 1/((self.mu/config.w)*self.bl_map**2 + inv_var_cls)
        mean_s = var_s*utils.complex_to_real(hp.almxfl(hp.map2alm((v + self.inv_noise*self.pix_map), lmax=self.lmax)*(1/config.w), self.bl_gauss))
        s_new = np.random.normal(size=len(mean_s))*np.sqrt(var_s) + mean_s #Sample the skymap s in spherical harmonics.
        return s_new, 1


    def sample(self, cls_, var_cls, old_s, metropolis_step=False, use_gibbs = False):
        """
        Choose what sampler to use in order to make the CR step. Note that if both RJPO and auxiliary variable steps are enabled,
        tthe sampler will use the auxiliary variable step only.

        :param cls_: array of floats, size (L_max + 1), of the C_\ell
        :param var_cls: array of floatts, size (L_max + 1)**2, of the variances of the real and imaginary parts of alms coeffs.
        :param old_s: array of floats, size (L_max + 1)**2, real and imaginary parts of the alms of the map of the previous iteration.
        :param metropolis_step: boolean. If True, use RJPO
        :param use_gibbs: boolean. If True, use auxiliary variable step.
        :return: sampled alms and 0 or 1 depending if the move has been accepted.
        """
        if use_gibbs:
            return self.sample_gibbs_change_variable(var_cls, old_s)
        if self.mask_path is not None:
            return self.sample_mask(cls_, var_cls, old_s, metropolis_step)
        else:
            return self.sample_no_mask(var_cls)



complex_dim = int((config.L_MAX_SCALARS+1)*(config.L_MAX_SCALARS+2)/2)


class PolarizedCenteredConstrainedRealization(ConstrainedRealization):
    ##Class performing Constrained Realization step for "EE" and "BB" only dataset.
    ##Note that we ALWAYS assume a noise covariance matrix proportional to the identity matrix. On top of this, we may or
    ##may not apply a skymask.
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, mask_path=None,
                 gibbs_cr = False, n_gibbs = 1, alpha = -0.995, overrelaxation = False, ula=False):
        """

        :param pix_map: array of floats, size Npix, observed skymap d.
        :param noise_temp: array of floats, size Npix, noise level for each pixel
        :param noise_pol: array of floats, size Npix, noise level for each pixel. Same for Q and U maps
        :param bl_map: array of floats, size (L_max + 1)**2, diagonal of the B matrix in the paper.
        :param lmax: integer, L_max
        :param Npix: integer, number of pixels
        :param bl_fwhm: gloatt, fwhm of the beam in degree.
        :param mask_path: string, path of the sky mask.
        :param gibbs_cr: boolean. If True, use the auxiliary varialbe scheme instead of PCG solver.
        :param n_gibbs: integer, number of iterations of
        """
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix, mask_path=mask_path)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.n_gibbs = n_gibbs
        self.ula = ula

        if mask_path is not None:
            #If we provide the path to a mask, we load it, downgrade it to the right resolution and then apply it the
            #inverse covariance matrix of the noise.
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise_temp *= self.mask
            self.inv_noise_pol *= self.mask
            self.inv_noise = [self.inv_noise_pol]
        else:
            self.inv_noise = [self.inv_noise_pol*np.ones(self.Npix)]

        self.mu = np.max(self.inv_noise) + 1e-14 #Mu is beta in the paper
        self.gibbs_cr = gibbs_cr
        self.overrelaxation = overrelaxation
        #The next lines define the PCG solver: diagonal preconditionner with 10^-6
        self.pcg_accuracy = 1.0e-5
        self.n_inv_filt = qcinv.opfilt_pp.alm_filter_ninv(self.inv_noise, self.bl_gauss, marge_maps = [])
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, 4000, self.pcg_accuracy, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]


        self.dls_to_cls_array = np.array([2 * np.pi / (l * (l + 1)) if l != 0 else 0 for l in range(lmax + 1)])#Array to convert the D_\ell to C_\ell
        self.alpha = alpha

        class cl(object):
            #Object with the cls to pass to qcinv.
            pass

        self.s_cls = cl
        self.bl_fwhm = bl_fwhm
        self.tau = 0.02

        if self.mask_path is not None:
            _, second_part_grad_E, second_part_grad_B = hp.map2alm([np.zeros(len(pix_map["Q"])), pix_map["Q"]*self.inv_noise_pol,
                                                pix_map["U"]*self.inv_noise_pol], lmax=lmax, iter=0, pol=True)

            second_part_grad_E *= (self.Npix/(4*np.pi))
            second_part_grad_B *= (self.Npix/(4*np.pi))

            hp.almxfl(second_part_grad_E, self.bl_gauss, inplace = True)
            hp.almxfl(second_part_grad_B, self.bl_gauss, inplace=True)

            self.second_part_grad_E = utils.complex_to_real(second_part_grad_E)
            self.second_part_grad_B = utils.complex_to_real(second_part_grad_B)


    def sample_no_mask(self, all_dls):
        """
        Makes the CR step when the full sky is observed. In this case the system becomes diagonal

        :param all_dls: dict {"EE":array, "BB":array}, with arrays representing the binned power spectra
        :return: dict {"EE":array, "BB":array}, a new alms skymap, with real and imaginary parts.
        """
        var_cls_E = utils.generate_var_cl(all_dls["EE"]) # Generating the C diagonal matrix
        var_cls_B = utils.generate_var_cl(all_dls["BB"]) # same here

        print(all_dls["EE"].shape)
        print(all_dls["BB"].shape)

        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Inverting the C matrix

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # same here

        sigma_E = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_E ) # Computing the inverse of the Q
        sigma_B = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_B ) # same here


        r_E = (self.Npix*self.inv_noise_pol[0]/(4*np.pi)) * self.pix_map["EE"] # Computing the "weiner" term of the system
        r_B = (self.Npix * self.inv_noise_pol[0] /(4 * np.pi)) * self.pix_map["BB"] # same here

        r_E = self.bl_map * r_E # same
        r_B = self.bl_map * r_B # same


        mean_E = sigma_E*r_E #computing the mean of the normal distribution
        mean_B = sigma_B*r_B # same here

        alms_E = mean_E + np.random.normal(size=len(var_cls_E)) * np.sqrt(sigma_E) #Actuall sampling: adding the fluctuations part
        alms_B = mean_B + np.random.normal(size=len(var_cls_B)) * np.sqrt(sigma_B) # Actual sampling: adding the fluctuations part

        return {"EE": alms_E, "BB": alms_B}, 1 #Returning the sampled sky map and 1 because we always accept

    def compute_gradient_no_mask(self, var_cls_E, var_cls_B, s_old):
        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Inverting the C matrix

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # same here

        sigma_E = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_E ) # Computing the inverse of the Q
        sigma_B = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_B ) # same here

        r_E = (self.Npix*self.inv_noise_pol[0]/(4*np.pi)) * self.pix_map["EE"] # Computing the "weiner" term of the system
        r_B = (self.Npix * self.inv_noise_pol[0] /(4 * np.pi)) * self.pix_map["BB"] # same here

        r_E = self.bl_map * r_E # same
        r_B = self.bl_map * r_B # same


        mean_E = sigma_E * r_E #computing the mean of the normal distribution
        mean_B = sigma_B * r_B  # same here

        grad_E = -(1/sigma_E)*(s_old["EE"]- mean_E)
        grad_B = -(1/sigma_B)*(s_old["BB"]- mean_B)
        return grad_E, grad_B

    def compute_log_proposal(self, var_cls_E, var_cls_B, s_new, s_old):
        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Inverting the C matrix

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # same here

        sigma_E = 1 / ((self.Npix / (self.noise_pol[0] * 4 * np.pi)) * self.bl_map ** 2 + inv_var_cls_E)  # Computing the inverse of the Q
        sigma_B = 1 / ((self.Npix / (self.noise_pol[0] * 4 * np.pi)) * self.bl_map ** 2 + inv_var_cls_B)  # same here

        grad_E, grad_B = self.compute_gradient_no_mask(var_cls_E, var_cls_B, s_old)

        return -(1/2)*np.sum((s_new["EE"] - s_old["EE"] - self.tau*sigma_E*grad_E)**2/(2*self.tau*sigma_E))\
        -(1/2)*np.sum((s_new["BB"] - s_old["BB"] - self.tau*sigma_B*grad_B)**2/(2*self.tau*sigma_B))

    def compute_log_density(self, s, var_cls_E, var_cls_B):
        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Inverting the C matrix

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # same here

        sigma_E = 1/ ((self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_E ) # Computing the inverse of the Q
        sigma_B = 1/ ((self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_B ) # same here

        r_E = (self.Npix*self.inv_noise_pol[0]/(4*np.pi)) * self.pix_map["EE"] # Computing the "weiner" term of the system
        r_B = (self.Npix * self.inv_noise_pol[0] /(4 * np.pi)) * self.pix_map["BB"] # same here

        r_E = self.bl_map * r_E # same
        r_B = self.bl_map * r_B # same


        mean_E = sigma_E * r_E #computing the mean of the normal distribution
        mean_B = sigma_B * r_B  # same here

        return -(1/2)*np.sum((s["EE"] - mean_E)**2/sigma_E) -(1/2)*np.sum((s["BB"] - mean_B)**2/sigma_B)


    def ULA_no_mask(self, all_dls, s_old):
        #Right tau = 0.0000001
        var_cls_E = utils.generate_var_cl(all_dls["EE"]) # Generating the C diagonal matrix
        var_cls_B = utils.generate_var_cl(all_dls["BB"]) # same here

        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0] # Inverting the C matrix

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0] # same here

        sigma_E = 1/ ((self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_E ) # Computing the inverse of the Q
        sigma_B = 1/ ((self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_B ) # same here

        grad_E, grad_B = self.compute_gradient_no_mask(var_cls_E, var_cls_B, s_old)

        s_new_EE = s_old["EE"] + self.tau*sigma_E*grad_E + np.sqrt(2*self.tau*sigma_E) * np.random.normal(size=len(grad_E))
        s_new_BB = s_old["BB"] + self.tau*sigma_B*grad_B + np.sqrt(2*self.tau*sigma_B) * np.random.normal(size=len(grad_B))
        s_new = {"EE":s_new_EE, "BB":s_new_BB}

        log_ratio = self.compute_log_density(s_new, var_cls_E, var_cls_B) \
                    + self.compute_log_proposal(var_cls_E, var_cls_B, s_old, s_new)\
                    - (self.compute_log_density(s_old, var_cls_E, var_cls_B) +
                       self.compute_log_proposal(var_cls_E, var_cls_B, s_new, s_old))

        if np.log(np.random.uniform()) < log_ratio:
            print("Accept !")
            return s_new, 1

        print("Reject !")
        return s_old, 0

    def sample_mask(self, all_dls):
        """
        CR step with a PCG solver, when a sky mask is applied.

        :param all_dls: dict {"EE":array, "BB":array}, with arrays representing the binned power spectra
        :return: dict {"EE":array, "BB":array}, a new alms skymap, with real and imaginary parts.
        """
        cls_EE = all_dls["EE"]*self.dls_to_cls_array #Converting D_\ell to C_\ell
        cls_BB = all_dls["BB"]*self.dls_to_cls_array #Same
        self.s_cls.clee = cls_EE #Setting the s_cls object that qcinv needs. It contains the power spectra and lmax
        self.s_cls.clbb = cls_BB #same
        self.s_cls.lmax = self.lmax #same

        var_cls_EE = utils.generate_var_cl(all_dls["EE"]) # Generating the C matrix
        var_cls_BB = utils.generate_var_cl(all_dls["BB"]) # Generating the C matrix
        var_cls_EE_inv = np.zeros(len(var_cls_EE))
        var_cls_EE_inv[np.where(var_cls_EE !=0)] = 1/var_cls_EE[np.where(var_cls_EE != 0)] #Inverting the C matrix
        var_cls_BB_inv = np.zeros(len(var_cls_BB))
        var_cls_BB_inv[np.where(var_cls_BB !=0)] = 1/var_cls_BB[np.where(var_cls_BB != 0)] # same
        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_pp, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None) #setting for qcinv

        #The next step computes the first term of the fluctuations in the CR step, BA^T N^-1 AB, see paper
        first_term_fluc = utils.adjoint_synthesis_hp([np.zeros(self.Npix),
                            np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol),
                           np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol)], bl_map=self.bl_map)

        #The next step computes the second term of the fluctuations in the CR step, C^-1
        second_term_fluc = [np.sqrt(var_cls_EE_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm),
                            np.sqrt(var_cls_BB_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)]

        #The next step goes from the real sph harmonics convention to the complex convention for qcinv.
        b_fluctuations = {"elm":utils.real_to_complex(first_term_fluc[1] + second_term_fluc[0]),
                          "blm":utils.real_to_complex(first_term_fluc[2] + second_term_fluc[1])}

        #Setting the solution to 0
        soltn = qcinv.opfilt_pp.eblm(np.zeros((2, int(qcinv.util_alm.lmax2nlm(self.lmax))), dtype=np.complex))
        pix_map = [self.pix_map["Q"], self.pix_map["U"]] # Setting the observed sky map
        _ = chain.sample(soltn, pix_map, b_fluctuations, pol=True) # Actual solving
        solution = {"EE":utils.complex_to_real(soltn.elm), "BB":utils.complex_to_real(soltn.blm)} # Going from complex to real for my code.

        return solution, 1 #returning solution and 1, to say we always accept.


    def sample_ula(self, all_dls, s_old):
        cls_EE = all_dls["EE"]*self.dls_to_cls_array # Convert D_\ell to C_\ell
        cls_BB = all_dls["BB"]*self.dls_to_cls_array # Same


        var_cls_EE = utils.generate_var_cl(all_dls["EE"]) #generate the C diagonal matrix
        var_cls_BB = utils.generate_var_cl(all_dls["BB"]) #same
        var_cls_EE_inv = np.zeros(len(var_cls_EE))
        var_cls_EE_inv[np.where(var_cls_EE !=0)] = 1/var_cls_EE[np.where(var_cls_EE != 0)] # invert the C diagonal matrix
        var_cls_BB_inv = np.zeros(len(var_cls_BB))
        var_cls_BB_inv[np.where(var_cls_BB !=0)] = 1/var_cls_BB[np.where(var_cls_BB != 0)] # same

        first_term_EE = - var_cls_EE_inv*s_old["EE"]
        first_term_BB = - var_cls_BB_inv*s_old["BB"]

        I, Q, U = hp.alm2map([utils.real_to_complex(np.zeros(len(s_old["EE"]))) ,
                              hp.almxfl(utils.real_to_complex(s_old["EE"]), self.bl_gauss),
                    hp.almxfl(utils.real_to_complex(s_old["BB"]), self.bl_gauss)],
                             pol=True, nside=self.nside, lmax=self.lmax, iter=0)

        Q *= self.inv_noise_pol
        U *= self.inv_noise_pol

        T, E, B = hp.map2alm([I, Q, U], lmax=self.lmax, pol=True, iter = 0)
        second_term_E = -hp.almxfl(E/config.w, self.bl_gauss, inplace =False)
        second_term_B = -hp.almxfl(B/config.w, self.bl_gauss, inplace=False)

        grad_E = first_term_EE + second_term_E + self.second_part_grad_E
        grad_B = first_term_BB + second_term_B + self.second_part_grad_B


        new_EE = s_old["EE"] + tau*grad_E + np.sqrt(2*tau)*np.random.normal(size=len(grad_E))
        new_BB = s_old["BB"] + tau*grad_B + np.sqrt(2*tau) * np.random.normal(size=len(grad_B))

        return {"EE":new_EE, "BB":new_BB}




    def sample_mask_rj(self, all_dls, s_old):
        """

        :param all_dls: dict {"EE":array, "BB":array}, with arrays representing the binned power spectra
        :param s_old: dict {"EE":array, "BB":array}, arrays representing the alms coefficients of the previous skymap in real format.
        :return: dict {"EE":array, "BB":array}, a new alms skymap, with real and imaginary parts and O/1 depending if we accept.
        """

        cls_EE = all_dls["EE"]*self.dls_to_cls_array # Convert D_\ell to C_\ell
        cls_BB = all_dls["BB"]*self.dls_to_cls_array # Same
        self.s_cls.clee = cls_EE # Setting the s_cls object for qcinv, this objects contains the power spectra and lmax.
        self.s_cls.clbb = cls_BB
        self.s_cls.lmax = self.lmax

        var_cls_EE = utils.generate_var_cl(all_dls["EE"]) #generate the C diagonal matrix
        var_cls_BB = utils.generate_var_cl(all_dls["BB"]) #same
        var_cls_EE_inv = np.zeros(len(var_cls_EE))
        var_cls_EE_inv[np.where(var_cls_EE !=0)] = 1/var_cls_EE[np.where(var_cls_EE != 0)] # invert the C diagonal matrix
        var_cls_BB_inv = np.zeros(len(var_cls_BB))
        var_cls_BB_inv[np.where(var_cls_BB !=0)] = 1/var_cls_BB[np.where(var_cls_BB != 0)] # same
        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_pp, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None) #setting qcinv

        fwd_op = chain.opfilt.fwd_op(self.s_cls, self.n_inv_filt) # The function fwd_op defined in qcinv will compute the product of a vector with the Q matrix, defined in the paper.

        # The next step computes the BA^T N^-1 AB term of the right hand term of the system.
        _, first_term_fluc_EE, first_term_fluc_BB = utils.adjoint_synthesis_hp([np.zeros(self.Npix),
                            np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol),
                           np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol)], bl_map=self.bl_map)

        # The next step computes the C^-1 part of the right hand term of the system
        second_term_fluc_EE = np.sqrt(var_cls_EE_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)
        second_term_fluc_BB = np.sqrt(var_cls_BB_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)


        b_flucs = {"elm":utils.real_to_complex(first_term_fluc_EE + second_term_fluc_EE),
                   "blm":utils.real_to_complex(first_term_fluc_BB + second_term_fluc_BB)} # Converts the right hand term to complex
        filling_soltn = np.zeros((2, int((self.lmax+1)*(self.lmax+2)/2)), dtype=np.complex)
        filling_soltn[0, :] = utils.real_to_complex(s_old["EE"]) # Setting the starting point of the PCG to the old skymap
        filling_soltn[1, :] = utils.real_to_complex(s_old["BB"]) # same
        soltn = qcinv.opfilt_pp.eblm(-filling_soltn) # same

        pix_map = [self.pix_map["Q"], self.pix_map["U"]]
        b_system = chain.sample(soltn, pix_map, b_flucs, pol = True) # Actual sampling, result in soltn, and also outputs "b_system" which is the right hand term of the system.

        ### Compute Q times solution of system
        soltn_bis = copy.deepcopy(soltn)

        result = fwd_op(soltn_bis) # Computing Q times the solution of the system provided by the PCG solver.
        alm_E = utils.complex_to_real(result.elm) # converting to real
        alm_B = utils.complex_to_real(result.blm) # same



        ## Once Qf(z) is computed, we compute the error:
        eta_E, eta_B = utils.complex_to_real(b_system.elm), utils.complex_to_real(b_system.blm) # same for b_system
        r_E = eta_E - alm_E # Difference between the right hand term of the system and Q times the solution provided by the PCG
        r_B = eta_B - alm_B # same
        diff_E = s_old["EE"] - utils.complex_to_real(soltn.elm) # Difference between old and new sky map
        diff_B = s_old["BB"] - utils.complex_to_real(soltn.blm) # Same


        log_proba = -np.sum(r_E*diff_E + r_B*diff_B) # Compute the acceptance rate
        if np.log(np.random.uniform()) < log_proba:
            return {"EE":utils.complex_to_real(soltn.elm), "BB":utils.complex_to_real(soltn.blm)}, 1 # If acceptance, output the new map and 1

        return s_old, 0 #Otherwise output the old map and 0

    def sample_gibbs_change_variable(self, all_dls, old_s):
        """
        This part does the CR step thanks to the auxiliary variable scheme.

        :param all_dls: dict {"EE":array, "BB":array}, with arrays representing the binned power spectra
        :param old_s: dict {"EE":array, "BB":array}, arrays representing the alms coefficients of the previous skymap in real format.
        :return: dict {"EE":array, "BB":array}, a new alms skymap, with real and imaginary parts and 1 because we always accept.
        """
        var_cls_EE = utils.generate_var_cl(all_dls["EE"]) # Computing the C diagonal matrix
        var_cls_BB = utils.generate_var_cl(all_dls["BB"]) # Same

        var_v_Q = self.mu - self.inv_noise_pol # Computing the Gamma matrix, see paper.
        var_v_U = self.mu - self.inv_noise_pol # Same

        for m in range(self.n_gibbs): # For n_gibbs number of iterations of the auxiliary variables scheme
            print("Gibbs CR iteration:", m)

            #This part samples v|s
            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])

            # The next steps computes the mean of the distribution of v|s
            _, map_Q, map_U = hp.alm2map([np.zeros(len(old_s_EE)) + 0j*np.zeros(len(old_s_EE)),hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                        hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q*map_Q # same
            mean_U = var_v_U*map_U # same

            v_Q = np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q) + mean_Q # Actual sampling of v|s
            v_U = np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) + mean_U # same


            ## This part samples s|v
            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)] # Computing C^-1
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)] # same

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE) #Computes the M matrix, see paper.
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB)   # same

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0) # mean part

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False)) # same
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False)) # same

            s_new_EE = np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE) + mean_s_EE # Actual sampling of s|v.
            s_new_BB = np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB) + mean_s_BB #Actual sampling of s|v.

            old_s = {"EE":s_new_EE, "BB":s_new_BB}

        return old_s, 1


    
    def overrelaxation_sampler(self, all_dls, old_s):
        """
        This method does the auxiliary variable step using overrelaxation.

        :param all_dls: dict {"EE":array, "BB":array}, with arrays representing the binned power spectra
        :param old_s: dict {"EE":array, "BB":array}, arrays representing the alms coefficients of the previous skymap in real format.
        :return: dict {"EE":array, "BB":array}, a new alms skymap, with real and imaginary parts and 1 because we always accept.
        """
        var_cls_EE = utils.generate_var_cl(all_dls["EE"])# Generating the C diagonal matrix.
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])# Same

        var_v_Q = self.mu - self.inv_noise_pol # computing Gamma, see paper.
        var_v_U = self.mu - self.inv_noise_pol # Same

        # This part samples v|s with regular Gibbs, cause we need an old v to perform overrelaxation
        old_s_EE = utils.real_to_complex(old_s["EE"])
        old_s_BB = utils.real_to_complex(old_s["BB"])

        _, map_Q, map_U = hp.alm2map(
            [np.zeros(len(old_s_EE)) + 0j * np.zeros(len(old_s_EE)), hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
             hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

        mean_Q = var_v_Q * map_Q  # computing the mean
        mean_U = var_v_U * map_U  # same

        v_Q = np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q) + mean_Q  # Actual sampling of v|s
        v_U = np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) + mean_U  # same

        for m in range(self.n_gibbs): # For n_gibbs number of iterations of the auxiliary variables scheme
            print("Gibbs CR iteration:", m)

            # This part samples s|v with overrelaxation
            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]#Computing C^-1
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]#same

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE) # Computing the M matrix, see paper
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB) # same

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False)) # Computing the mean of s|v
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False)) # Same

            #The two next steps actually sample s|v with overrelaxation
            old_s_EE = mean_s_EE + self.alpha * (old_s["EE"] - mean_s_EE) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE)
            old_s_BB = mean_s_BB + self.alpha * (old_s["BB"] - mean_s_BB) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB)

            old_s = {"EE": old_s_EE, "BB":old_s_BB}


            # This part samples v|s with overrelaxation
            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])

            _, map_Q, map_U = hp.alm2map(
                [np.zeros(len(old_s_EE)) + 0j * np.zeros(len(old_s_EE)),
                 hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                 hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q * map_Q  # computing the mean
            mean_U = var_v_U * map_U  # same

            v_Q = mean_Q + self.alpha * (v_Q - mean_Q) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q)# Actual sampling of v|s
            v_U = mean_U + self.alpha * (v_U - mean_U) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) # same

            # This part samples s|v with overrelaxation again.
            # In total we do s|v, v|s, s|v, all with overrelaxation, so we can beneficiate of overrelaxation on bith v and s.
            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]#Computing C^-1
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]#same

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE) # Computing the M matrix, see paper
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB) # same

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False)) # Computing the mean of s|v
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False)) # Same

            old_s_EE = mean_s_EE + self.alpha * (old_s["EE"] - mean_s_EE) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE)
            old_s_BB = mean_s_BB + self.alpha * (old_s["BB"] - mean_s_BB) + np.sqrt(1-self.alpha**2) * np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB)

            old_s = {"EE": old_s_EE, "BB": old_s_BB}


        return old_s, 1


    def sample(self, all_dls, s_old = None):
        if self.gibbs_cr == True and s_old is not None and self.overrelaxation == True and self.mask_path is not None:
            return self.overrelaxation_sampler(all_dls, s_old)
        if self.gibbs_cr == True and s_old is not None and self.overrelaxation == False and self.mask_path is not None:
            return self.sample_gibbs_change_variable(all_dls, s_old)
        if s_old is not None and self.mask_path is not None:
            return self.sample_mask_rj(all_dls, s_old)
        if self.mask_path is None and s_old is not None and self.ula == True:
            print("ULA no mask !")
            return self.ULA_no_mask(all_dls, s_old)
        if self.mask_path is None:
            print("No Mask !")
            return self.sample_no_mask(all_dls)
        else:
            return self.sample_mask(all_dls)








class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise_temp, noise_pol, beam, nside, lmax, Npix, mask_path = None,
                 polarization = False, bins=None, n_iter = 100000, rj_step = False, all_sph=False, gibbs_cr = False,
                overrelaxation=False, ula=False):
        super().__init__(pix_map, noise_temp, beam, nside, lmax, polarization = polarization, bins=bins, n_iter = n_iter
                         ,rj_step=rj_step, gibbs_cr = gibbs_cr)
        
        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise_temp, self.bl_map, beam, lmax, Npix, mask_path,
                                                                      isotropic=True)
            self.cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp)
        else:
            self.cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp, mask_path=mask_path)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                               self.bl_map, lmax, Npix, beam,
                                                                               mask_path=mask_path,
                                                                               gibbs_cr =gibbs_cr, overrelaxation=overrelaxation, ula=ula)
