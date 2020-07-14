import utils
import numpy as np
import healpy as hp
import config
from utils import generate_var_cl
from scipy.stats import invwishart, invgamma
import matplotlib.pyplot as plt







class ExactSampler():

    def __init__(self, pix_map, nside, lmax, noise_I, noise_Q, bl_fwhm):
        self.pix_map = pix_map
        self.nside = nside
        self.lmax = lmax
        self.Npix = 12*self.nside**2
        self.noise_I = noise_I
        self.noise_Q = noise_Q
        self.bl_fwhm = bl_fwhm
        ####A remplacer pas un call à une fonction healpy
        fwhm_radians = (np.pi / 180) * self.bl_fwhm
        bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=self.lmax)
        self.bl_map = generate_var_cl(bl_gauss)
        self.pix_part_variance =(self.Npix/(4*np.pi))*np.stack([(1/self.noise_I)*np.ones((self.lmax+1)**2)*self.bl_map**2,
                                        (1/self.noise_Q)*np.ones((self.lmax+1)**2)*self.bl_map**2,
                                        (1/self.noise_Q)*np.ones((self.lmax+1)**2)*self.bl_map**2], axis = 1)

        self.Sigma = 1/self.pix_part_variance
        alms_T, alms_E, alms_B = utils.adjoint_synthesis_hp([pix_map[0]/self.noise_I, pix_map[1]/self.noise_Q,
                                                                     pix_map[2]/self.noise_Q], self.bl_fwhm)
        temp = np.stack([alms_T, alms_E, alms_B], axis = 1)

        self.mu = self.Sigma*temp

    def sample_skymap(self):
        w = np.random.normal(size = ((self.lmax+1)**2, 3))
        s = np.sqrt(self.Sigma)*w + self.mu

        return s

    def sample_cls(self, alms):
        alms_TT_complex = utils.real_to_complex(alms[:, 0])
        alms_EE_complex = utils.real_to_complex(alms[:, 1])
        alms_BB_complex = utils.real_to_complex(alms[:, 2])
        spec_TT, spec_EE, spec_BB, spec_TE, _, _ = hp.alm2cl([alms_TT_complex, alms_EE_complex, alms_BB_complex],
                                                             lmax=self.lmax)

        sampled_power_spec = np.zeros((self.lmax + 1, 3, 3))
        for i in range(2, self.lmax + 1):
            deg_freed = 2 * i - 2
            param_mat = np.zeros((2, 2))
            param_mat[0, 0] = spec_TT[i]
            param_mat[1, 0] = param_mat[0, 1] = spec_TE[i]
            param_mat[1, 1] = spec_EE[i]
            param_mat *= (2 * i + 1) * i * (i + 1) / (2 * np.pi)
            sampled_TT_TE_EE = invwishart.rvs(deg_freed, param_mat)
            beta = (2 * i + 1) * i * (i + 1) * spec_BB[i] / (4 * np.pi)
            sampled_BB = beta * invgamma.rvs(a=(2 * i - 1) / 2)
            sampled_power_spec[i, :2, :2] = sampled_TT_TE_EE
            sampled_power_spec[i, 2, 2] = sampled_BB

        return sampled_power_spec

    def sample_joint(self, n_sample):
        h_dls = []
        for i in range(n_sample):
            if i % 100 == 0:
                print("Exact sampler iteration:", i)

            s = self.sample_skymap()
            dls = self.sample_cls(s)
            h_dls.append(dls)

        return np.array(h_dls)





d = np.load("test_polarization.npy", allow_pickle=True)
d = d.item()
pix_map = d["pix_map"]
h_dls_centered = d["h_cls_centered"]


exact_sampler = ExactSampler(pix_map, config.NSIDE, config.L_MAX_SCALARS, config.noise_covar_temp, config.noise_covar_pol,
                             config.beam_fwhm)


h_dls = exact_sampler.sample_joint(10000)

l_interest = 4
i = 1
j = 0
plt.plot(h_dls[:, l_interest, i, j],label="Exact", alpha=0.5)
plt.plot(h_dls_centered[:, l_interest, i, j], label="Centered Gibbs", alpha=0.5)
plt.legend(loc="upper right")
#plt.axhline(y=init_cls[l_interest, i, j])
plt.show()

plt.hist(h_dls[:, l_interest, i, j], density=True, bins=50, label="Exact", alpha=0.5)
plt.hist(h_dls_centered[:, l_interest, i, j], density=True, bins=50, label=">Centered", alpha=0.5)
plt.legend(loc="upper right")
plt.show()


print(h_dls_centered.shape)

