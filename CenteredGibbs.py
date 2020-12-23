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

import warnings

warnings.simplefilter('always', UserWarning)



class CenteredClsSampler(ClsSampler):


    def sample(self, alms):
        """
        :param alms: alm skymap
        :return: Sample each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        alms_complex = utils.real_to_complex(alms)
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax)
        exponent = np.array([(2 * l + 1) / 2 for l in range(self.lmax +1)])
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
        sampled_dls = binned_betas * invgamma.rvs(a=binned_alphas)
        sampled_dls[:2] = 0
        return sampled_dls


class PolarizedCenteredClsSampler(ClsSampler):

    def sample_one_pol(self, alms_complex, pol="EE"):
        """
        :param alms: alm skymap of the polarization
        :return: Sample each the - potentially binned - Dls from an inverse gamma. NOT THE CLs !
        """
        observed_Cls = hp.alm2cl(alms_complex, lmax=self.lmax)
        exponent = np.array([(2 * l + 1) / 2 for l in range(self.lmax + 1)])
        binned_betas = []
        binned_alphas = []
        betas = np.array([(2 * l + 1) * l * (l + 1) * (observed_Cl / (4 * np.pi)) for l, observed_Cl in
                          enumerate(observed_Cls)])

        for i, l in enumerate(self.bins[pol][:-1]):
            somme_beta = np.sum(betas[l:self.bins[pol][i + 1]])
            somme_exponent = np.sum(exponent[l:self.bins[pol][i + 1]])
            alpha = somme_exponent - 1
            binned_alphas.append(alpha)
            binned_betas.append(somme_beta)

        binned_alphas[0] = 1
        sampled_dls = binned_betas * invgamma.rvs(a=binned_alphas)
        sampled_dls[:2] = 0
        return sampled_dls

    def sample(self, alms):
        alms_EE_complex = utils.real_to_complex(alms["EE"])
        alms_BB_complex = utils.real_to_complex(alms["BB"])

        binned_dls_EE = self.sample_one_pol(alms_EE_complex, "EE")
        binned_dls_BB = self.sample_one_pol(alms_BB_complex, "BB")

        return {"EE":binned_dls_EE, "BB":binned_dls_BB}









class CenteredConstrainedRealization(ConstrainedRealization):

    def sample_no_mask(self, cls_, var_cls):
        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)
        b_weiner = self.bl_map * utils.adjoint_synthesis_hp(self.inv_noise * self.pix_map)
        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise))

        start = time.time()
        ###To change !!!!
        ###Sigma = 1/(inv_var_cls + self.inv_noise * (self.Npix / (4 * np.pi)) * self.bl_map ** 2)
        Sigma = 1 / (inv_var_cls + self.inv_noise[0] * (self.Npix / (4 * np.pi)) * self.bl_map ** 2)
        weiner = Sigma * b_weiner
        flucs = Sigma * b_fluctuations
        map = weiner + flucs
        err = 0
        #map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0
        time_to_solution = time.time() - start
        return map, 1


    def sample_mask(self, cls_, var_cls, s_old, metropolis_step=False):
        self.s_cls.cltt = cls_
        self.s_cls.lmax = self.lmax
        cl_inv = np.zeros(len(cls_))
        cl_inv[np.where(cls_ !=0)] = 1/cls_[np.where(cls_ != 0)]

        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None)

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
        soltn = utils.complex_to_real(soltn_complex)
        if not metropolis_step:
            return soltn, 1
        else:
            approx_sol_complex = hp.almxfl(hp.map2alm(hp.alm2map(hp.almxfl(soltn_complex, self.bl_gauss), nside=self.nside)*self.inv_noise, lmax=self.lmax)
                                   *self.Npix/(4*np.pi), self.bl_gauss) + hp.almxfl(soltn_complex, cl_inv, inplace=False)
            r = b_system - approx_sol_complex
            r = utils.complex_to_real(r)
            log_proba = min(0, -np.dot(r,(s_old - soltn)))
            print("log Proba")
            print(log_proba)
            if np.log(np.random.uniform()) < log_proba:
                return soltn, 1
            else:
                return s_old, 0

    def sample_gibbs_change_variable(self, var_cls, old_s):
        old_s = utils.real_to_complex(old_s)
        var_v = self.mu - self.inv_noise
        mean_v = var_v * hp.alm2map(hp.almxfl(old_s, self.bl_gauss), nside=self.nside, lmax=self.lmax)
        v = np.random.normal(size=len(mean_v))*np.sqrt(var_v) + mean_v

        inv_var_cls = np.zeros(len(var_cls))
        inv_var_cls[np.where(var_cls != 0)] = 1/var_cls[np.where(var_cls != 0)]
        var_s = 1/((self.mu/config.w)*self.bl_map**2 + inv_var_cls)
        mean_s = var_s*utils.complex_to_real(hp.almxfl(hp.map2alm((v + self.inv_noise*self.pix_map), lmax=self.lmax)*(1/config.w), self.bl_gauss))
        s_new = np.random.normal(size=len(mean_s))*np.sqrt(var_s) + mean_s
        return s_new, 1

    def sample_gibbs(self, var_cls, old_s):
        for _ in range(1):
            old_s = utils.real_to_complex(old_s)
            var_u = 1/(self.mu - self.inv_noise)
            mean_u = hp.alm2map(hp.almxfl(old_s, self.bl_gauss), nside=self.nside, lmax=self.lmax)
            u = np.random.normal(size=len(mean_u)) * np.sqrt(var_u) + mean_u

            inv_var_cls = np.zeros(len(var_cls))
            inv_var_cls[np.where(var_cls != 0)] = 1/var_cls[np.where(var_cls != 0)]
            var_s = 1/((self.mu/config.w)*self.bl_map**2 + inv_var_cls)
            mean_s = var_s * utils.complex_to_real(
                hp.almxfl(hp.map2alm((u*(self.mu - self.inv_noise) + self.inv_noise * self.pix_map), lmax=self.lmax) * (1 / config.w), self.bl_gauss))

            s_new = np.random.normal(size=len(mean_s)) * np.sqrt(var_s) + mean_s
            old_s = s_new[:]

        return s_new, 1

    def sample_gibbs_pix(self, var_cls, old_s):
        ##Ne ps oublier le changement de base de départ:
        old_s = hp.alm2map(utils.real_to_complex(self.bl_map*old_s), lmax=self.lmax, nside=self.nside)
        bl_times_var_cls = self.bl_map**2*var_cls
        inv_bl_times_var_cls = np.zeros(len(bl_times_var_cls))
        inv_bl_times_var_cls[bl_times_var_cls != 0] = 1/bl_times_var_cls[bl_times_var_cls != 0]
        mu = np.max(inv_bl_times_var_cls) + 1e-5

        variance_v = (mu - inv_bl_times_var_cls)
        mean_v = utils.complex_to_real(hp.map2alm(old_s, lmax=self.lmax))*variance_v
        v_new = mean_v + np.random.normal(size = len(mean_v))*np.sqrt(variance_v)

        variance_u = 1/(mu*4*np.pi/self.Npix + self.inv_noise)
        mean_u = variance_u*(hp.alm2map(utils.real_to_complex(v_new), nside=self.nside, lmax=self.lmax)*4*np.pi/self.Npix
                 + self.inv_noise*self.pix_map)
        new_u = np.random.normal(size=len(mean_u))*np.sqrt(variance_u) + mean_u
        ##Ne pas oublier le changementde base de fin:
        new_u = (1/self.bl_map)*utils.complex_to_real(hp.map2alm(new_u, lmax=self.lmax))
        return new_u, 1


    def sample(self, cls_, var_cls, old_s, metropolis_step=False, use_gibbs = False):
        if use_gibbs:
            return self.sample_gibbs_change_variable(var_cls, old_s)
        if self.mask_path is not None:
            return self.sample_mask(cls_, var_cls, old_s, metropolis_step)
        else:
            return self.sample_no_mask(cls_, var_cls)



complex_dim = int((config.L_MAX_SCALARS+1)*(config.L_MAX_SCALARS+2)/2)


"""
@njit(parallel=True)
def matrix_product(dls_, b):
    alms_shape = np.zeros((complex_dim, 3, 3))
    result = np.zeros(((config.L_MAX_SCALARS+1)**2, 3))

    for l in prange(config.L_MAX_SCALARS + 1):
        for m in range(l + 1):
            idx = m * (2 * config.L_MAX_SCALARS + 1 - m) // 2 + l
            if l == 0:
                alms_shape[idx, :, :] = dls_[l, :, :]
            else:
                alms_shape[idx, :, :] = dls_[l, :, :] #* 2 * np.pi / (l * (l + 1))

    for i in prange(config.L_MAX_SCALARS + 1):
        result[i, 0] = alms_shape[i, 0, 0] * b[i, 0] + alms_shape[i, 0, 1] * b[i, 1]
        result[i, 1] = alms_shape[i, 1, 0] * b[i, 0] + alms_shape[i, 1, 1] * b[i, 1]
        result[i, 2] = alms_shape[i, 2, 2] * b[i, 2]

    for i in prange(config.L_MAX_SCALARS+1, complex_dim):
        result[2*i - (config.L_MAX_SCALARS+1), 0] = alms_shape[i, 0, 0]*b[2*i - (config.L_MAX_SCALARS+1), 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1), 1]
        result[2*i - (config.L_MAX_SCALARS+1), 1] = alms_shape[i, 1, 0]*b[2*i - (config.L_MAX_SCALARS+1), 0] + alms_shape[i, 1, 1]*b[2*i - (config.L_MAX_SCALARS+1), 1]
        result[2*i - (config.L_MAX_SCALARS+1), 2] = alms_shape[i, 2, 2]*b[2*i - (config.L_MAX_SCALARS+1), 2]

        result[2*i - (config.L_MAX_SCALARS+1) + 1, 0] = alms_shape[i, 0, 0]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 0] + alms_shape[i, 0, 1]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 1]
        result[2*i - (config.L_MAX_SCALARS+1) + 1, 1] = alms_shape[i, 1, 0]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 0] + alms_shape[i, 1, 1]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 1]
        result[2*i - (config.L_MAX_SCALARS+1) + 1, 2] = alms_shape[i, 2, 2]*b[2*i - (config.L_MAX_SCALARS+1) + 1, 2]

    return result
"""

@njit(parallel=False)
def matrix_product(dls_, b):
    alms_shape = np.zeros((complex_dim, 3, 3))
    result = np.zeros(((config.L_MAX_SCALARS+1)**2, 3))

    for l in prange(config.L_MAX_SCALARS + 1):
        for m in range(l + 1):
            idx = m * (2 * config.L_MAX_SCALARS + 1 - m) // 2 + l
            if l == 0:
                alms_shape[idx, :, :] = dls_[l, :, :]
            else:
                alms_shape[idx, :, :] = dls_[l, :, :]

    for i in prange(config.L_MAX_SCALARS + 1):
        result[i, :] = np.dot(alms_shape[i, :, :], b[i, :])

    for i in prange(config.L_MAX_SCALARS + 1, complex_dim):
        result[2*i - (config.L_MAX_SCALARS+1), :] = np.dot(alms_shape[i, :, :], b[2*i - (config.L_MAX_SCALARS+1), :])
        result[2*i - (config.L_MAX_SCALARS+1) +1, :] = np.dot(alms_shape[i, :, :],
                                                               b[2 * i - (config.L_MAX_SCALARS + 1) + 1, :])


    return result


def compute_inverse_and_cholesky(all_cls, pix_part_variance):
    inv_cls = np.zeros((len(all_cls), 3, 3))
    chol_cls = np.zeros((len(all_cls), 3, 3))

    for i in prange(2, len(all_cls)):
        inv_cls[i, :2, :2] = scipy.linalg.inv(all_cls[i, :2, :2])
        inv_cls[i, 2, 2] = 1/all_cls[i, 2, 2]
        inv_cls[i, :, :] += np.diag(pix_part_variance[i, :])
        inv_cls[i, :, :] = np.linalg.inv(inv_cls[i, :, :])
        chol_cls[i, :, :] = np.linalg.cholesky(inv_cls[i, :, :])

    return inv_cls,chol_cls


class PolarizedCenteredConstrainedRealization(ConstrainedRealization):
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, mask_path=None, all_sph=False):
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix, mask_path=mask_path)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.all_sph = all_sph

        if mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise_temp *= self.mask
            self.inv_noise_pol *= self.mask
            self.inv_noise = [self.inv_noise_pol]
        else:
            self.inv_noise = [self.inv_noise_pol*np.ones(self.Npix)]

        self.pcg_accuracy = 1.0e-4
        self.n_inv_filt = qcinv.opfilt_pp.alm_filter_ninv(self.inv_noise, self.bl_gauss, marge_maps = [])
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, 4000, self.pcg_accuracy, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]
        self.dls_to_cls_array = np.array([2 * np.pi / (l * (l + 1)) if l != 0 else 0 for l in range(lmax + 1)])

        class cl(object):
            pass

        self.s_cls = cl
        #self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([self.inv_noise_temp*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
        #                                self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
        #                                self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2], axis = 1)

        self.bl_fwhm = bl_fwhm

    """
    def sample(self, all_dls):
        start = time.time()
        rescaling = [0 if l == 0 else 2*np.pi/(l*(l+1)) for l in range(self.lmax+1)]
        all_cls = all_dls.copy()
        for i in range(self.lmax+1):
            all_cls[i, :, :] *= rescaling[i]


        inv_cls, chol_cls = compute_inverse_and_cholesky(all_cls, self.pix_part_variance)


        b_weiner_unpacked = utils.adjoint_synthesis_hp([self.inv_noise_temp * self.pix_map[0],
                    self.inv_noise_pol * self.pix_map[1], self.inv_noise_pol * self.pix_map[2]], self.bl_map)

        b_weiner = np.stack(b_weiner_unpacked, axis = 1)
        b_fluctuations = np.random.normal(size=((config.L_MAX_SCALARS+1)**2, 3))

        weiner_map = matrix_product(inv_cls, b_weiner)
        fluctuations = matrix_product(chol_cls, b_fluctuations)
        map = weiner_map + fluctuations
        time_to_solution = time.time() - start
        err = 0
        return map, time_to_solution, err
    """
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
        #chain.solve(soltn, pix_map)
        solution = {"EE":utils.complex_to_real(soltn.elm), "BB":utils.complex_to_real(soltn.blm)}

        return solution, 0

    def sample_mask_rj(self, all_dls, s_old):
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

        sol_elm = soltn.elm.copy()
        sol_blm = soltn.blm.copy()

        ### Compute Q times solution of system
        hp.almxfl(sol_elm, self.bl_gauss, inplace=True)
        hp.almxfl(sol_blm, self.bl_gauss, inplace=True)


        _, Q, U = hp.alm2map([np.zeros(len(sol_elm), dtype=np.complex), sol_elm, sol_blm], nside=self.nside, lmax=self.lmax, pol=True)
        Q *= self.inv_noise_pol
        U *= self.inv_noise_pol

        _, alm_E, alm_B = hp.map2alm([np.zeros(self.Npix), Q, U], pol=True, lmax=config.L_MAX_SCALARS)*self.Npix/(4*np.pi)
        hp.almxfl(alm_E, self.bl_gauss, inplace=True)
        hp.almxfl(alm_B, self.bl_gauss, inplace=True)

        alm_E = utils.complex_to_real(alm_E)
        alm_B = utils.complex_to_real(alm_B)

        alm_E += var_cls_EE_inv * utils.complex_to_real(soltn.elm)
        alm_B += var_cls_BB_inv * utils.complex_to_real(soltn.blm)

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


    def sample(self, all_dls, s_old = None):
        if s_old is not None:
            return self.sample_mask_rj(all_dls, s_old)
        if self.mask_path is None:
            return self.sample_no_mask(all_dls)
        else:
            return self.sample_mask(all_dls)




class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise_temp, noise_pol, beam, nside, lmax, Npix, mask_path = None,
                 polarization = False, bins=None, n_iter = 100000, rj_step = False, all_sph=False):
        super().__init__(pix_map, noise_temp, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter
                         ,rj_step=rj_step)
        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise_temp, self.bl_map, beam, lmax, Npix, mask_path,
                                                                      isotropic=True)
            self.cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp)
        else:
            self.cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp, mask_path=mask_path)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol,
                                                                               self.bl_map, lmax, Npix, beam,
                                                                               mask_path=mask_path, all_sph=all_sph)
