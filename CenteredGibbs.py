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

    def root_to_find(self, x, u, l, scale_mat, cl_EE, cl_TE, norm):
        low_bound = cl_TE ** 2 / cl_EE
        integral, err = scipy.integrate.quad(self.compute_conditional_TT, a=low_bound, b=x,
                                             args=(l, scale_mat, cl_EE, cl_TE))
        return integral / norm - u

    def deriv_root_to_find(self, x, u, l, scale_mat, cl_EE, cl_TE, norm):
        return self.compute_conditional_TT(x, l, scale_mat, cl_EE, cl_TE)/norm


    def sample_bin(self, alms):
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

        for i in self.bins["EE"]:
            beta_EE = scale_mat[i, 1, 1]
            cl_EE = invgamma.rvs(a=(2 * i - 3) / 2, scale=beta_EE / 2)
            sampled_power_spec[i, 1, 1] = cl_EE

        for i in self.bins["TE"]:
            determinant = np.linalg.det(scale_mat[i, :, :])
            student_TE = student.rvs(df=2 * i - 2)
            cl_TE = (np.sqrt(determinant) * sampled_power_spec[i, 1, 1] * student_TE / np.sqrt(2 * i - 2) + scale_mat[i, 0, 1] * sampled_power_spec[i, 1, 1]) / \
                    scale_mat[i, 1, 1]
            sampled_power_spec[i, 1, 0] = sampled_power_spec[i, 0, 1] = cl_TE

        for i in self.bins["TT"]:
            u = np.random.uniform()
            cl_EE = sampled_power_spec[i, 1, 1]
            cl_TE = sampled_power_spec[i, 0, 1]
            ratio = cl_TE ** 2 / cl_EE
            maximum = (cl_EE ** 2 * scale_mat[i, 0, 0] + cl_TE ** 2 * scale_mat[i, 1, 1] + cl_TE ** 2 * (
                        2 * i + 1) * cl_EE - 2 * cl_TE * cl_EE * scale_mat[i, 0, 1]) / ((2 * i + 1) * cl_EE ** 2)

            precision = -self.compute_conditional_TT(maximum, i, scale_mat[i], cl_EE, cl_TE) * (
                        1 / (maximum * cl_EE - cl_TE ** 2) ** 2) * (((2 * i + 1) / 2) * cl_EE ** 2 - (
                        scale_mat[i, 1, 1] * cl_TE ** 2 + scale_mat[i, 0, 0] * cl_EE ** 2 - 2 * scale_mat[i,
                    0, 1] * cl_TE * cl_EE) * cl_EE / (maximum * cl_EE - cl_TE ** 2))
            var = 1 / precision
            stdd = np.sqrt(var)

            norm1, err = scipy.integrate.quad(self.compute_conditional_TT, a=ratio, b=maximum,
                                             args=(i, scale_mat[i, :, :], cl_EE, cl_TE))

            norm2, err = scipy.integrate.quad(self.compute_conditional_TT, a=maximum, b=100*maximum,
                                             args=(i, scale_mat[i, :, :], cl_EE, cl_TE))

            norm = norm1 + norm2
            print("Norm, l=",i)
            print(norm)
            if False:
                xx = np.linspace(ratio, maximum*10, 10000)
                yy = []
                for x in xx:
                    #print(ratio, x)
                    y = self.compute_conditional_TT(x, i, scale_mat[i, :, :], cl_EE, cl_TE)
                    yy.append(y)

                plt.plot(xx,yy)
                plt.axvline(x=maximum)
                plt.axvline(x=max2, color="red")
                plt.show()

            #sol = scipy.optimize.root_scalar(self.root_to_find, x0=maximum-0.1, x1=maximum+0.1, args=(u, i, scale_mat[i, :, :], cl_EE, cl_TE, norm))
            sol = scipy.optimize.root_scalar(self.root_to_find, x0=maximum, fprime = self.deriv_root_to_find,
                                             args=(u, i, scale_mat[i, :, :], cl_EE, cl_TE, norm))
            has_converged = sol.converged
            if not has_converged:
                print("No root found")
                print("MAXIMUM")
                print(maximum)
                print("NORM")
                print(norm)
                print("Trying a second time")
                norm1, err = scipy.integrate.quad(self.compute_conditional_TT, a=ratio, b=maximum,
                                                  args=(i, scale_mat[i, :, :], cl_EE, cl_TE))

                norm2, err = scipy.integrate.quad(self.compute_conditional_TT, a=maximum, b=100*maximum,
                                                  args=(i, scale_mat[i, :, :], cl_EE, cl_TE))

                norm = norm1 + norm2
                low_bound = maximum - np.random.uniform(0, 0.5)
                up_bound = maximum + np.random.uniform(0, 0.5)
                sol = scipy.optimize.root_scalar(self.root_to_find, x0 = low_bound, x1 = up_bound,
                                                 args=(u, i, scale_mat[i, :, :], cl_EE, cl_TE, norm), method="secant")
                has_converged = sol.converged
                if not has_converged:
                    print("Second time didn't converge either")
                    return None
                else:
                    print("Second time converged")

            cl_TT = sol.root
            sampled_power_spec[i, 0, 0] = cl_TT

        sampled_power_spec *= np.array([i*(i+1)/(2*np.pi) for i in range(config.L_MAX_SCALARS+1)])[:, None, None]
        return sampled_power_spec

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
        #print("INPUT MAP 2 ALM DIAG")
        #print(self.inv_noise * self.pix_map)
        print("weiner 50")
        print(utils.real_to_complex(b_weiner))
        return map, time_to_solution, err, weiner


    def sample_mask(self, cls_, var_cls):
        self.s_cls.cltt = cls_
        self.s_cls.lmax = self.lmax

        inv_var_cls = np.zeros(len(var_cls))
        np.reciprocal(var_cls, where=config.mask_inversion, out=inv_var_cls)

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=('log_'))

        #b_weiner = self.bl_map * utils.adjoint_synthesis_hp(self.inv_noise * self.pix_map)
        b_fluctuations = np.random.normal(loc=0, scale=1, size=self.dimension_alm) * np.sqrt(inv_var_cls) + \
                         self.bl_map * utils.adjoint_synthesis_hp(np.random.normal(loc=0, scale=1, size=self.Npix)
                                                              * np.sqrt(self.inv_noise))

        soltn_complex = np.zeros(int(qcinv.util_alm.lmax2nlm(self.lmax)), dtype=np.complex)
        #soltn_complex = utils.real_to_complex(b_fluctuations)
        chain.solve(soltn_complex, self.pix_map)
        hp.almxfl(soltn_complex, cls_, inplace=True)
        soltn = utils.complex_to_real(soltn_complex)
        return soltn, 0, 0,


    def sample(self, cls_, var_cls):
        if self.mask_path is not None:
            return self.sample_mask(cls_, var_cls)
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

    def __init__(self, pix_map, noise_temp, noise_pol, beam, nside, lmax, Npix, polarization = False, bins=None, n_iter = 10000):
        super().__init__(pix_map, noise_temp, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise_temp, self.bl_map, lmax, Npix, isotropic=True)
            self.cls_sampler = CenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise_temp)
        else:
            self.cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise_temp)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise_temp, noise_pol, self.bl_map, lmax, Npix, beam, isotropic=True)





















"""
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
"""