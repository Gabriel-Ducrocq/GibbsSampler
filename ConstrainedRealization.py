import numpy as np
import healpy as hp
import qcinv

class ConstrainedRealization():
    ##Basic class of constrained realization step, from which all CR sampler will inherit

    def __init__(self, pix_map, noise, bl_map, fwhm_deg, lmax, Npix, mask_path=None, isotropic=True):
        """

        :param pix_map: array of float, observed skymap called d in the paper.
        :param noise: array of floats, size Npix, representing the noise level.
        :param bl_map: array of floats, the b_\ells expension over all the spherical harmonics coefficients. This is the diagonal
                of the B matrix in the paper.
        :param fwhm_deg: float, definition of the beam in degrees
        :param lmax: integer, L_max
        :param Npix: integer, size of the skymap
        :param mask_path: string, path to the skymask.
        :param isotropic: boolean, True if the noise is an isotropic Gaussian. Otherwise False.
        """
        self.pix_map = pix_map
        self.isotropic = isotropic
        self.noise = noise
        self.inv_noise = 1/noise
        self.bl_map = bl_map
        self.lmax = lmax
        self.dimension_alm = (lmax + 1) ** 2
        self.Npix = Npix
        self.nside = hp.npix2nside(Npix) # Get NSIDE from Npix
        self.fwhm_radians = (np.pi/180)*fwhm_deg
        self.bl_gauss = hp.gauss_beam(fwhm=self.fwhm_radians, lmax=lmax) # Compute the B_\ell coeffs
        self.mask_path = mask_path
        if mask_path is not None:
            ##If a path to a skymask is provided, load it, downgrade to the right resolution and multiply the diagonal
            ## of the noise covariance matrix, to apply the mak.
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise *= self.mask

        ##Settings for the PCG solver: diagonal preconditionner, lmax, 4000 iterations max and a prrecision threshold of 10^-6.
        self.n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv(self.inv_noise, self.bl_gauss)
        self.chain_descr = [[0, ["diag_cl"], lmax, self.nside, 4000, 1.0e-6, qcinv.cd_solve.tr_cg, qcinv.cd_solve.cache_mem()]]


        self.mu = np.max(self.inv_noise) + 0.0000001 ## mu is the inverse of the beta coeff in the auxiliary variable scheme.
        class cl(object):
            ## Defines the class of the object s_cls that we will have to pass to the qcinv code forr PCG resolution.
            pass

        self.s_cls = cl


    def sample(self, cls, var_cls):
        return None

