import numpy as np
import healpy as hp
import qcinv

class ConstrainedRealization():

    def __init__(self, pix_map, noise, bl_map, fwhm, lmax, Npix, mask_path=None, isotropic=True):
        self.pix_map = pix_map
        self.isotropic = isotropic
        self.noise = noise
        self.inv_noise = 1/noise
        self.bl_map = bl_map
        self.lmax = lmax
        self.dimension_alm = (lmax + 1) ** 2
        self.Npix = Npix
        self.nside = hp.npix2nside(Npix)
        self.fwhm_radians = (np.pi/180)*fwhm
        self.bl_gauss = hp.gauss_beam(fwhm=self.fwhm_radians, lmax=lmax)
        self.mask_path = mask_path
        if mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise *= self.mask

        self.n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv(self.inv_noise, self.bl_gauss)
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, np.inf, 1.0e-6, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]
        #self.chain_descr = [[3, ["split(dense,     64, diag_cl)"], 128, 64, 3, 0.0, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()],
        #               [2, ["split(stage(3), 128, diag_cl)"], 256, 128, 3, 0.0, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()],
        #               [1, ["split(stage(2), 256, diag_cl)"], 512, 256, 3, 0.0, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()],
        #               [0, ["split(stage(1), 512, diag_cl)"], self.lmax, self.nside, np.inf, 1.0e-6, qcinv.cd_solve.tr_cg,
        #                qcinv.cd_solve.cache_mem()]]

        self.mu = np.max(self.inv_noise) + 0.0001
        class cl(object):
            pass

        self.s_cls = cl

    def sample(self, cls, var_cls):
        return None

