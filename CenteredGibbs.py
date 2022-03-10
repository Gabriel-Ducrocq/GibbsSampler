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
from linear_algebra import compute_inverse_matrices, compute_matrix_product
from numba import njit, prange
from splines import sample_splines
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
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, mask_path=None, all_sph=False,
                 gibbs_cr = False, n_gibbs = 1):
        """

        :param pix_map: array of floats, size Npix, observed skymap d.
        :param noise_temp: array of floats, size Npix, noise level for each pixel
        :param noise_pol: array of floats, size Npix, noise level for each pixel. Same for Q and U maps
        :param bl_map: array of floats, size (L_max + 1)**2, diagonal of the B matrix in the paper.
        :param lmax: integer, L_max
        :param Npix: integer, number of pixels
        :param bl_fwhm: gloatt, fwhm of the beam in degree.
        :param mask_path: string, path of the sky mask.
        :param all_sph:
        :param gibbs_cr:
        :param n_gibbs:
        """
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix, mask_path=mask_path)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.all_sph = all_sph
        self.n_gibbs = n_gibbs
        self.delta = self.noise_pol[0]*0.00000015

        if mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise_temp *= self.mask
            self.inv_noise_pol *= self.mask
            self.inv_noise = [self.inv_noise_pol]
        else:
            self.inv_noise = [self.inv_noise_pol*np.ones(self.Npix)]

        self.mu = np.max(self.inv_noise) + 1e-14   #0.0000001
        self.gibbs_cr = gibbs_cr
        self.pcg_accuracy = 1.0e-5
        self.n_inv_filt = qcinv.opfilt_pp.alm_filter_ninv(self.inv_noise, self.bl_gauss, marge_maps = [])
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, 4000, self.pcg_accuracy, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]
        self.dls_to_cls_array = np.array([2 * np.pi / (l * (l + 1)) if l != 0 else 0 for l in range(lmax + 1)])
        self.alpha = 0.00000001

        class cl(object):
            pass

        self.s_cls = cl
        self.bl_fwhm = bl_fwhm
        _, second_part_grad_E, second_part_grad_B = hp.map2alm([np.zeros(len(pix_map["Q"])), pix_map["Q"]*self.inv_noise_pol,
                                            pix_map["U"]*self.inv_noise_pol], lmax=lmax, iter=0, pol=True)

        second_part_grad_E *= (self.Npix/(4*np.pi))
        second_part_grad_B *= (self.Npix/(4*np.pi))

        hp.almxfl(second_part_grad_E, self.bl_gauss, inplace = True)
        hp.almxfl(second_part_grad_B, self.bl_gauss, inplace=True)

        self.second_part_grad_E = utils.complex_to_real(second_part_grad_E)
        self.second_part_grad_B = utils.complex_to_real(second_part_grad_B)



    def sample_no_mask(self, all_dls):
        var_cls_E = utils.generate_var_cl(all_dls["EE"])
        var_cls_B = utils.generate_var_cl(all_dls["BB"])

        inv_var_cls_E = np.zeros(len(var_cls_E))
        inv_var_cls_E[var_cls_E != 0] = 1/var_cls_E[var_cls_E != 0]

        inv_var_cls_B = np.zeros(len(var_cls_B))
        inv_var_cls_B[var_cls_B != 0] = 1/var_cls_B[var_cls_B != 0]

        sigma_E = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_E )
        sigma_B = 1/ ( (self.Npix/(self.noise_pol[0]*4*np.pi)) * self.bl_map**2 + inv_var_cls_B )

        if not self.all_sph:
            _, r_E, r_B = hp.map2alm([np.zeros(self.Npix),self.pix_map["Q"]*self.inv_noise_pol, self.pix_map["U"]*self.inv_noise_pol],
                       lmax=self.lmax, pol=True)*self.Npix/(4*np.pi)

            r_E = self.bl_map * utils.complex_to_real(r_E)
            r_B = self.bl_map * utils.complex_to_real(r_B)

        else:
            r_E = (self.Npix*self.inv_noise_pol[0]/(4*np.pi)) * self.pix_map["EE"]
            r_B = (self.Npix * self.inv_noise_pol[0] /(4 * np.pi)) * self.pix_map["BB"]

            r_E = self.bl_map * r_E
            r_B = self.bl_map * r_B


        mean_E = sigma_E*r_E
        mean_B = sigma_B*r_B

        alms_E = mean_E + np.random.normal(size=len(var_cls_E)) * np.sqrt(sigma_E)
        alms_B = mean_B + np.random.normal(size=len(var_cls_B)) * np.sqrt(sigma_B)

        return {"EE": utils.remove_monopole_dipole_contributions(alms_E),
                "BB": utils.remove_monopole_dipole_contributions(alms_B)}, 0

    def sample_mask(self, all_dls):
        cls_EE = all_dls["EE"]*self.dls_to_cls_array
        cls_BB = all_dls["BB"]*self.dls_to_cls_array
        self.s_cls.clee = cls_EE
        self.s_cls.clbb = cls_BB
        self.s_cls.lmax = self.lmax

        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])
        var_cls_EE_inv = np.zeros(len(var_cls_EE))
        var_cls_EE_inv[np.where(var_cls_EE !=0)] = 1/var_cls_EE[np.where(var_cls_EE != 0)]
        var_cls_BB_inv = np.zeros(len(var_cls_BB))
        var_cls_BB_inv[np.where(var_cls_BB !=0)] = 1/var_cls_BB[np.where(var_cls_BB != 0)]
        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_pp, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None)


        first_term_fluc = utils.adjoint_synthesis_hp([np.zeros(self.Npix),
                            np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol),
                           np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol)], bl_map=self.bl_map)

        second_term_fluc = [np.sqrt(var_cls_EE_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm),
                            np.sqrt(var_cls_BB_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)]

        b_fluctuations = {"elm":utils.real_to_complex(first_term_fluc[1] + second_term_fluc[0]),
                          "blm":utils.real_to_complex(first_term_fluc[2] + second_term_fluc[1])}


        soltn = qcinv.opfilt_pp.eblm(np.zeros((2, int(qcinv.util_alm.lmax2nlm(self.lmax))), dtype=np.complex))
        pix_map = [self.pix_map["Q"], self.pix_map["U"]]
        _ = chain.sample(soltn, pix_map, b_fluctuations, pol=True)
        solution = {"EE":utils.complex_to_real(soltn.elm), "BB":utils.complex_to_real(soltn.blm)}

        return solution, 0

    def sample_mask_rj(self, all_dls, s_old):
        true_sol, _ = self.sample_no_mask(all_dls)

        cls_EE = all_dls["EE"]*self.dls_to_cls_array
        cls_BB = all_dls["BB"]*self.dls_to_cls_array
        self.s_cls.clee = cls_EE
        self.s_cls.clbb = cls_BB
        self.s_cls.lmax = self.lmax

        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])
        var_cls_EE_inv = np.zeros(len(var_cls_EE))
        var_cls_EE_inv[np.where(var_cls_EE !=0)] = 1/var_cls_EE[np.where(var_cls_EE != 0)]
        var_cls_BB_inv = np.zeros(len(var_cls_BB))
        var_cls_BB_inv[np.where(var_cls_BB !=0)] = 1/var_cls_BB[np.where(var_cls_BB != 0)]
        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_pp, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None)

        fwd_op = chain.opfilt.fwd_op(self.s_cls, self.n_inv_filt)

        _, first_term_fluc_EE, first_term_fluc_BB = utils.adjoint_synthesis_hp([np.zeros(self.Npix),
                            np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol),
                           np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise_pol)], bl_map=self.bl_map)

        second_term_fluc_EE = np.sqrt(var_cls_EE_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)
        second_term_fluc_BB = np.sqrt(var_cls_BB_inv)*np.random.normal(loc=0, scale=1, size=self.dimension_alm)


        b_flucs = {"elm":utils.real_to_complex(first_term_fluc_EE + second_term_fluc_EE),
                   "blm":utils.real_to_complex(first_term_fluc_BB + second_term_fluc_BB)}
        filling_soltn = np.zeros((2, int((self.lmax+1)*(self.lmax+2)/2)), dtype=np.complex)
        filling_soltn[0, :] = utils.real_to_complex(s_old["EE"])
        filling_soltn[1, :] = utils.real_to_complex(s_old["BB"])
        soltn = qcinv.opfilt_pp.eblm(-filling_soltn)

        pix_map = [self.pix_map["Q"], self.pix_map["U"]]
        b_system = chain.sample(soltn, pix_map, b_flucs, pol = True)

        ### Compute Q times solution of system
        soltn_bis = copy.deepcopy(soltn)

        result = fwd_op(soltn_bis)
        alm_E = utils.complex_to_real(result.elm)
        alm_B = utils.complex_to_real(result.blm)



        ## Once Qf(z) is computed, we compute the error:
        eta_E, eta_B = utils.complex_to_real(b_system.elm), utils.complex_to_real(b_system.blm)
        r_E = eta_E - alm_E
        r_B = eta_B - alm_B
        diff_E = s_old["EE"] - utils.complex_to_real(soltn.elm)
        diff_B = s_old["BB"] - utils.complex_to_real(soltn.blm)


        log_proba = -np.sum(r_E*diff_E + r_B*diff_B)
        if np.log(np.random.uniform()) < log_proba:
            return {"EE":utils.complex_to_real(soltn.elm), "BB":utils.complex_to_real(soltn.blm)}, 1

        return s_old, 0

    def sample_gibbs_change_variable(self, all_dls, old_s):
        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])

        var_v_Q = self.mu - self.inv_noise_pol
        var_v_U = self.mu - self.inv_noise_pol

        for m in range(self.n_gibbs):
            print("Gibbs CR iteration:", m)

            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])


            _, map_Q, map_U = hp.alm2map([np.zeros(len(old_s_EE)) + 0j*np.zeros(len(old_s_EE)),hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                        hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q*map_Q
            mean_U = var_v_U*map_U

            v_Q = np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q) + mean_Q
            v_U = np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) + mean_U



            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE)
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB)

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False))
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False))

            s_new_EE = np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE) + mean_s_EE
            s_new_BB = np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB) + mean_s_BB

            old_s = {"EE":s_new_EE, "BB":s_new_BB}

        return old_s, 1


    def sample_gibbs_change_variable2(self, all_dls, old_s):
        self.mu = np.max(self.inv_noise) + 1
        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])

        var_v_Q = self.mu - self.inv_noise_pol
        var_v_U = self.mu - self.inv_noise_pol

        for m in range(self.n_gibbs):
            print("Gibbs CR iteration:", m)

            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])


            _, map_Q, map_U = hp.alm2map([np.zeros(len(old_s_EE)) + 0j*np.zeros(len(old_s_EE)),hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                        hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q*map_Q
            mean_U = var_v_U*map_U

            v_Q = np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q) + self.alpha * mean_Q
            v_U = np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) + self.alpha * mean_U



            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]

            var_s_EE = 1 / ((self.mu / config.w) * self.alpha**2 * self.bl_map ** 2 + inv_var_cls_EE)
            var_s_BB = 1 / ((self.mu / config.w) * self.alpha**2 * self.bl_map ** 2 + inv_var_cls_BB)

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        self.alpha*v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        self.alpha*v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False))
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False))

            s_new_EE = np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE) + mean_s_EE
            s_new_BB = np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB) + mean_s_BB

            old_s = {"EE":s_new_EE, "BB":s_new_BB}

        return old_s, 1


    def compute_log_lik(self, s_E, s_B):
        s_E_complex = hp.almxfl(utils.real_to_complex(s_E), self.bl_gauss, inplace=False)
        s_B_complex = hp.almxfl(utils.real_to_complex(s_B), self.bl_gauss, inplace=False)

        _, s_Q, s_U = hp.alm2map([np.zeros(len(s_E_complex), dtype=np.complex), s_E_complex, s_B_complex], lmax=self.lmax
                                 , nside=self.nside, pol=True)

        Q_part = -(1/2)*np.sum((self.pix_map["Q"] -s_Q)**2*self.inv_noise_pol)
        U_part = -(1/2)*np.sum((self.pix_map["U"] -s_U)**2*self.inv_noise_pol)

        return Q_part + U_part


    def compute_grad_log_lik(self, s_E, s_B):
        s_E_complex = utils.real_to_complex(s_E)
        s_B_complex = utils.real_to_complex(s_B)
        s_E_complex = hp.almxfl(s_E_complex, self.bl_gauss, inplace=False)
        s_B_complex = hp.almxfl(s_B_complex, self.bl_gauss, inplace=False)

        _, s_Q, s_U = hp.alm2map([np.zeros(len(s_E_complex), dtype=np.complex), s_E_complex, s_B_complex],
                                 lmax=self.lmax
                                 , nside=self.nside, pol=True)


        s_Q *= self.inv_noise_pol
        s_U *= self.inv_noise_pol

        _, first_term_E, first_term_B = hp.map2alm([np.zeros(len(s_Q)), s_Q, s_U], lmax=self.lmax, iter=0, pol=True)
        first_term_E *= (self.Npix/(4*np.pi))
        first_term_B *= (self.Npix/(4*np.pi))

        hp.almxfl(first_term_E, self.bl_gauss, inplace=True)
        hp.almxfl(first_term_B, self.bl_gauss, inplace=True)

        first_term_E = utils.complex_to_real(first_term_E)
        first_term_B = utils.complex_to_real(first_term_B)

        return -first_term_E + self.second_part_grad_E, -first_term_B + self.second_part_grad_B


    def compute_h(self, s_old_E, s_old_B, y_E, y_B, A_EE, A_BB):
        grad_y_E, grad_y_B = self.compute_grad_log_lik(y_E, y_B)

        h_E = np.sum((s_old_E - (2/self.delta)*A_EE *(y_E + (self.delta/4)* grad_y_E))* grad_y_E/((2/self.delta)*A_EE+ 1))

        h_B = np.sum((s_old_B - (2 / self.delta)* A_BB *( y_B + (self.delta / 4) * grad_y_B))*
                     grad_y_B / ((2/self.delta) * A_BB + 1))

        return h_E + h_B

    def compute_log_ratio(self, s_old_E, s_old_B, y_E, y_B, A_EE, A_BB):
        first_part = self.compute_log_lik(y_E[:], y_B[:]) - self.compute_log_lik(s_old_E[:], s_old_B[:])
        second_part = self.compute_h(s_old_E[:], s_old_B[:], y_E[:], y_B[:], A_EE[:], A_BB[:]) \
                      - self.compute_h(y_E[:], y_B[:], s_old_E[:], s_old_B[:],A_EE[:], A_BB[:])
        return first_part + second_part


    def aux_grad(self, all_dls, old_s):
        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])

        A_EE = (self.delta/2) * var_cls_EE/(var_cls_EE + self.delta/2)
        A_BB = (self.delta/2) * var_cls_BB/(var_cls_BB + self.delta/2)
        sigma_proposal_EE = (2/self.delta)*A_EE**2 + A_EE
        sigma_proposal_BB = (2 / self.delta) * A_BB ** 2 + A_BB

        mean_E, mean_B = self.compute_grad_log_lik(old_s["EE"], old_s["BB"])
        mean_E *= self.delta/2
        mean_B *= self.delta/2
        mean_E += old_s["EE"]
        mean_B += old_s["BB"]

        mean_E *= (2/self.delta) * A_EE
        mean_B *= (2/self.delta) * A_BB

        y_E = np.random.normal(size=len(mean_E))*np.sqrt(sigma_proposal_EE) + mean_E
        y_B = np.random.normal(size=len(mean_B))*np.sqrt(sigma_proposal_BB) + mean_B


        log_r = self.compute_log_ratio(old_s["EE"], old_s["BB"], y_E, y_B, A_EE, A_BB)

        if np.log(np.random.uniform()) < log_r:
            return {"EE":y_E, "BB":y_B}, 1


        return old_s, 0
    
    
    def overrelaxation(self, all_dls, old_s):
        #alpha = -0.98
        alpha = -0.995
        #alpha = -0.9999
        #alpha = -0.9999999
        var_cls_EE = utils.generate_var_cl(all_dls["EE"])
        var_cls_BB = utils.generate_var_cl(all_dls["BB"])

        var_v_Q = self.mu - self.inv_noise_pol
        var_v_U = self.mu - self.inv_noise_pol

        for m in range(self.n_gibbs):
            print("Gibbs CR iteration:", m)

            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])


            _, map_Q, map_U = hp.alm2map([np.zeros(len(old_s_EE)) + 0j*np.zeros(len(old_s_EE)),hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                        hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q*map_Q
            mean_U = var_v_U*map_U

            v_Q = np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q) + mean_Q
            v_U = np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U) + mean_U


            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE)
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB)

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False))
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False))

            s_new_EE = mean_s_EE + alpha * (old_s["EE"] - mean_s_EE) + np.sqrt(1-alpha**2) * np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE)
            s_new_BB = mean_s_BB + alpha * (old_s["BB"] - mean_s_BB) + np.sqrt(1-alpha**2) * np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB)

            old_s = {"EE": s_new_EE, "BB": s_new_BB}

            
            ###Deactivate this part for a moment
            """
            ##Again
            old_s_EE = utils.real_to_complex(old_s["EE"])
            old_s_BB = utils.real_to_complex(old_s["BB"])


            _, map_Q, map_U = hp.alm2map([np.zeros(len(old_s_EE)) + 0j*np.zeros(len(old_s_EE)),hp.almxfl(old_s_EE, self.bl_gauss, inplace=False),
                        hp.almxfl(old_s_BB, self.bl_gauss, inplace=False)], nside=self.nside, lmax=self.lmax, pol=True)

            mean_Q = var_v_Q*map_Q
            mean_U = var_v_U*map_U

            v_Q = mean_Q + alpha*(v_Q - mean_Q) + np.sqrt(1- alpha**2) * np.random.normal(size=len(mean_Q)) * np.sqrt(var_v_Q)
            v_U = mean_U + alpha * (v_U - mean_U) + np.sqrt(1-alpha**2) * np.random.normal(size=len(mean_U)) * np.sqrt(var_v_U)

            ###again S.
            inv_var_cls_EE = np.zeros(len(var_cls_EE))
            inv_var_cls_EE[np.where(var_cls_EE != 0)] = 1 / var_cls_EE[np.where(var_cls_EE != 0)]
            inv_var_cls_BB = np.zeros(len(var_cls_BB))
            inv_var_cls_BB[np.where(var_cls_BB != 0)] = 1 / var_cls_BB[np.where(var_cls_BB != 0)]

            var_s_EE = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_EE)
            var_s_BB = 1 / ((self.mu / config.w) * self.bl_map ** 2 + inv_var_cls_BB)

            _, alm_EE, alm_BB = hp.map2alm([np.zeros(len(v_Q)),
                        v_Q + self.inv_noise_pol * self.pix_map["Q"],
                        v_U + self.inv_noise_pol * self.pix_map["U"]], lmax=self.lmax, pol=True, iter = 0)

            mean_s_EE = var_s_EE * utils.complex_to_real(hp.almxfl(alm_EE/config.w, self.bl_gauss, inplace=False))
            mean_s_BB = var_s_BB * utils.complex_to_real(hp.almxfl(alm_BB/config.w, self.bl_gauss, inplace=False))

            s_new_EE = mean_s_EE + alpha * (old_s["EE"] - mean_s_EE) + np.sqrt(1-alpha**2) * np.random.normal(size=len(mean_s_EE)) * np.sqrt(var_s_EE)
            s_new_BB = mean_s_BB + alpha * (old_s["BB"] - mean_s_BB) + np.sqrt(1-alpha**2) * np.random.normal(size=len(mean_s_BB)) * np.sqrt(var_s_BB)

            old_s = {"EE":s_new_EE, "BB":s_new_BB}
            """
            
        return old_s, 1


    def sample(self, all_dls, s_old = None):
        if self.gibbs_cr == True and s_old is not None:
            return self.overrelaxation(all_dls, s_old)
            #s_old, _ =self.overrelaxation(all_dls, s_old)
            #return self.sample_gibbs_change_variable(all_dls, s_old)
        if s_old is not None:
            return self.sample_mask_rj(all_dls, s_old)
        if self.mask_path is None:
            return self.sample_no_mask(all_dls)
        else:
            return self.sample_mask(all_dls)







class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise_temp, noise_pol, beam, nside, lmax, Npix, mask_path = None,
                 polarization = False, bins=None, n_iter = 100000, rj_step = False, all_sph=False, gibbs_cr = False):
        super().__init__(pix_map, noise_temp, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter
                         ,rj_step=rj_step, gibbs_cr = gibbs_cr)
        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise_temp, self.bl_map, beam, lmax, Npix, mask_path,
                                                                      isotropic=True)
            self.cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp)
        else:
            self.cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp, mask_path=mask_path)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                               self.bl_map, lmax, Npix, beam,
                                                                               mask_path=mask_path, all_sph=all_sph,
                                                                               gibbs_cr =gibbs_cr)
