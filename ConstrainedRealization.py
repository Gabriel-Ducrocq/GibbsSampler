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
            self.mask = hp.read_map(mask_path)
            self.inv_noise *= self.mask

        self.n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv(self.inv_noise, self.bl_gauss)
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, np.inf, 1.0e-6, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]

        class cl(object):
            pass

        self.s_cls = cl

    def sample(self, cls, var_cls):
        return None

