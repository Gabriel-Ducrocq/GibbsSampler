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
        return sampled_dls


alm_obj = hp.Alm()

class PolarizedCenteredClsSampler(ClsSampler):
    def sample_unbinned(self, alms):
        alms_TT_complex = utils.real_to_complex(alms[:, 0])
        alms_EE_complex = utils.real_to_complex(alms[:, 1])
        alms_BB_complex = utils.real_to_complex(alms[:, 2])
        spec_TT, spec_EE, spec_BB, spec_TE, _, _ = hp.alm2cl([alms_TT_complex, alms_EE_complex, alms_BB_complex], lmax=self.lmax)

        sampled_power_spec = np.zeros((self.lmax+1, 3, 3))
        for i in range(2, self.lmax+1):
            deg_freed = 2*i-2
            param_mat = np.zeros((2, 2))
            param_mat[0, 0] = spec_TT[i]
            param_mat[1, 0] = param_mat[0, 1] = spec_TE[i]
            param_mat[1, 1] = spec_EE[i]
            param_mat *= (2*i+1)*i*(i+1)/(2*np.pi)
            sampled_TT_TE_EE = invwishart.rvs(deg_freed, param_mat)

            beta = (2*i+1)*i*(i+1)*spec_BB[i]/(4*np.pi)
            sampled_BB = beta*invgamma.rvs(a=(2*i-1)/2)
            sampled_power_spec[i, :2, :2] = sampled_TT_TE_EE
            sampled_power_spec[i, 2, 2] = sampled_BB

        return sampled_power_spec

    def compute_conditional_TT(self, x, l, scale_mat, cl_EE, cl_TE):
        param_mat = np.array([[x, cl_TE], [cl_TE, cl_EE]])
        if x <= cl_TE ** 2 / cl_EE:
            return 0
        else:
            return invwishart.pdf(param_mat, df=2 * l - 2, scale=scale_mat)

    def compute_log_conditional_TT(self, x, l, scale_mat, cl_EE, cl_TE):
        param_mat = np.array([[x, cl_TE], [cl_TE, cl_EE]])
        if x <= cl_TE ** 2 / cl_EE:
            return -np.inf
        else:
            return invwishart.logpdf(param_mat, df=2 * l - 2, scale=scale_mat)

    def compute_rescale_conditional_TT(self, x, l, scale_mat, cl_EE, cl_TE, maximum):
        return maximum*self.compute_conditional_TT(x*maximum, l, scale_mat, cl_EE, cl_TE)

    def compute_log_rescale_conditional_TT(self, x, l, scale_mat, cl_EE, cl_TE, maximum):
        return np.log(maximum) + self.compute_log_conditional_TT(x*maximum, l, scale_mat, cl_EE, cl_TE)

    def find_upper_bound_rescaled(self, l, scale_mat, cl_EE, cl_TE, maximum, max_log_val):
        x0 = 1
        x1 = 1.618*x0
        log_ratio = max_log_val - self.compute_log_rescale_conditional_TT(x1, l, scale_mat, cl_EE, cl_TE, maximum)
        while log_ratio < 12.5:
            x_new = x1 + 1.618*(x1 - x0)
            x0 = x1
            x1 = x_new
            log_ratio = max_log_val - self.compute_log_rescale_conditional_TT(x1, l, scale_mat, cl_EE, cl_TE, maximum)

        return x1

    def find_lower_bound_rescaled(self,l, scale_mat, cl_EE, cl_TE, maximum, max_log_val):
        hard_lower_bound = (cl_TE**2/cl_EE)/maximum
        x0 = 1
        x1 = (x0 + hard_lower_bound)/2
        log_ratio = max_log_val - self.compute_log_rescale_conditional_TT(x1, l, scale_mat, cl_EE, cl_TE, maximum)
        while log_ratio < 12.5:
            x_new = (x1 + hard_lower_bound)/2
            x1 = x_new
            log_ratio = max_log_val - self.compute_log_rescale_conditional_TT(x1, l, scale_mat, cl_EE, cl_TE, maximum)

        return x1

    def sample_bin(self, alms, i):
        alms_TT_complex = utils.real_to_complex(alms[:, 0])
        alms_EE_complex = utils.real_to_complex(alms[:, 1])
        alms_BB_complex = utils.real_to_complex(alms[:, 2])
        spec_TT, spec_EE, spec_BB, spec_TE, _, _ = hp.alm2cl([alms_TT_complex, alms_EE_complex, alms_BB_complex],
                                                             lmax=self.lmax)

        rescale = np.array([(2 * i + 1) for i in range(0, config.L_MAX_SCALARS + 1)])
        scale_mat = np.zeros((config.L_MAX_SCALARS+1, 2, 2))
        scale_mat[:, 0, 0] = spec_TT*rescale
        scale_mat[:, 1, 1] = spec_EE*rescale
        scale_mat[:, 1, 0] = scale_mat[:, 0, 1] = spec_TE*rescale
        alpha = np.array([(2*i-1)/2 for i in range(2,config.L_MAX_SCALARS+1)])
        sampled_power_spec = np.zeros((self.lmax + 1, 3, 3))

        beta_BB = spec_BB*rescale/2
        sample_BB = invgamma.rvs(a=alpha, scale=beta_BB[2:])
        sampled_power_spec[2:, 2, 2] = sample_BB

        #for i in self.bins["EE"]:
        beta_EE = scale_mat[i, 1, 1]
        cl_EE = invgamma.rvs(a=(2 * i - 3) / 2, scale=beta_EE / 2)
        sampled_power_spec[i, 1, 1] = cl_EE

        #for i in self.bins["TE"]:
        determinant = np.linalg.det(scale_mat[i, :, :])
        student_TE = student.rvs(df=2 * i - 2)
        cl_TE = (np.sqrt(determinant) * sampled_power_spec[i, 1, 1] * student_TE / np.sqrt(2 * i - 2) + scale_mat[i, 0, 1] * sampled_power_spec[i, 1, 1]) / \
                    scale_mat[i, 1, 1]
        sampled_power_spec[i, 1, 0] = sampled_power_spec[i, 0, 1] = cl_TE

        #for i in self.bins["TT"]:
        cl_EE = sampled_power_spec[i, 1, 1]
        cl_TE = sampled_power_spec[i, 0, 1]
        ratio = cl_TE ** 2 / cl_EE
        maximum = (cl_EE ** 2 * scale_mat[i, 0, 0] + cl_TE ** 2 * scale_mat[i, 1, 1] + cl_TE ** 2 * (
                        2 * i + 1) * cl_EE - 2 * cl_TE * cl_EE * scale_mat[i, 0, 1]) / ((2 * i + 1) * cl_EE ** 2)

        max_log_value = self.compute_log_rescale_conditional_TT(1,i, scale_mat[i, :, :], cl_EE, cl_TE, maximum)
        upper_bound = self.find_upper_bound_rescaled(i, scale_mat[i, :, :], cl_EE, cl_TE, maximum, max_log_value)
        lower_bound = self.find_lower_bound_rescaled(i, scale_mat[i, :, :], cl_EE, cl_TE, maximum, max_log_value)

        xx = np.linspace(lower_bound, upper_bound, 2*6400)
        y_cs = np.array([self.compute_rescale_conditional_TT(x, i, scale_mat[i, :, :], cl_EE, cl_TE, maximum) for x in xx])
        cs = scipy.interpolate.CubicSpline(xx,y_cs)
        u = np.random.uniform()
        integs = np.array([cs.integrate(lower_bound, x) for x in xx])
        integs /= integs[-1]
        position = np.searchsorted(integs, u)
        sample = (u - integs[position-1])*(xx[position] - xx[position-1])/(integs[position] - integs[position-1]) + xx[position-1]

        cl_TT = sample*maximum

        return cl_TT
        #print("Sampled cl_TT", cl_TT)
        #sampled_power_spec[i, 0, 0] = cl_TT

        #sampled_power_spec *= np.array([i*(i+1)/(2*np.pi) for i in range(config.L_MAX_SCALARS+1)])[:, None, None]
        #return sampled_power_spec

    def sample(self, alm_map):
        if False:
            return self.sample_unbinned(alm_map)
        else:
            return self.sample_bin(alm_map)









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
        map[[0, 1, self.lmax + 1, self.lmax + 2]] = 0.0
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
        #### Since at the end of the solver the output is multiplied by C^-1, it's enough to remultiply it by C^(1/2) to
        ### To produce a non centered map !
        hp.almxfl(soltn_complex, cls_, inplace=True)
        soltn = utils.remove_monopole_dipole_contributions(utils.complex_to_real(soltn_complex))
        if not metropolis_step:
            return soltn, 1
        else:
            approx_sol_complex = hp.almxfl(hp.map2alm(hp.alm2map(hp.almxfl(soltn_complex, self.bl_gauss), nside=self.nside)*self.inv_noise, lmax=self.lmax)
                                   *self.Npix/(4*np.pi), self.bl_gauss) + hp.almxfl(soltn_complex, cl_inv, inplace=False)
            r = b_system - approx_sol_complex
            r = utils.complex_to_real(r)
            log_proba = min(0, -np.dot(r,(s_old - soltn)))
            print("Proba")
            print(log_proba)
            if np.log(np.random.uniform()) < log_proba:
                return soltn, 1
            else:
                return s_old, 0

    def sample_gibbs(self, var_cls, old_s):
        """
        var_z = 1/(self.mu - self.inv_noise)
        old_s = utils.real_to_complex(old_s)
        mean_z = hp.alm2map(hp.almxfl(old_s, self.bl_gauss), nside=self.nside, lmax=self.lmax)
        z = np.random.normal(size = len(mean_z))*np.sqrt(var_z) + mean_z

        inv_var_cls = np.zeros(len(var_cls))
        inv_var_cls[np.where(var_cls != 0)] = 1/var_cls[np.where(var_cls !=0)]
        var_s = 1/((self.mu/config.w)*self.bl_map**2 + inv_var_cls)
        mu = self.inv_noise*self.pix_map + z*(self.mu - self.inv_noise)
        mean_s = var_s*self.bl_map*(1/config.w)*utils.complex_to_real(hp.map2alm(mu, lmax=self.lmax))
        new_s = np.random.normal(size = len(mean_s))*np.sqrt(var_s) + mean_s
        """
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

    def sample(self, cls_, var_cls, old_s, metropolis_step=True, use_gibbs = False):
        if use_gibbs:
            return self.sample_gibbs(var_cls, old_s)
        #if self.mask_path is not None:
        if False:
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

#@njit(parallel=False)
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
    def __init__(self, pix_map, noise_temp, noise_pol, bl_map, lmax, Npix, bl_fwhm, isotropic=True):
        super().__init__(pix_map, noise_temp, bl_map, bl_fwhm, lmax, Npix, isotropic=True)
        self.noise_temp = noise_temp
        self.noise_pol = noise_pol
        self.inv_noise_temp = 1/self.noise_temp
        self.inv_noise_pol = 1/self.noise_pol
        self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([self.inv_noise_temp*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2,
                                        self.inv_noise_pol*np.ones((config.L_MAX_SCALARS+1)**2)*self.bl_map**2], axis = 1)

        self.bl_fwhm = bl_fwhm

    def sample(self, all_dls, var_all_dls):
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



class CenteredGibbs(GibbsSampler):

    def __init__(self, pix_map, noise_temp, noise_pol, beam, nside, lmax, Npix, mask_path = None,
                 polarization = False, bins=None, n_iter = 10000):
        super().__init__(pix_map, noise_temp, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise_temp, self.bl_map, beam, lmax, Npix, mask_path,
                                                                      isotropic=True)
            self.cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise_temp)
        else:
            self.cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise_temp)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol, self.bl_map, lmax, Npix, beam, isotropic=True)
