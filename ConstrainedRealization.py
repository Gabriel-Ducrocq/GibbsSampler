import numpy as np
import healpy as hp


class ConstrainedRealization():

    def __init__(self, pix_map, noise, bl_map, lmax, Npix, isotropic=True):
        self.pix_map = pix_map
        self.isotropic = isotropic
        self.noise = noise
        self.inv_noise = 1/noise
        self.bl_map = bl_map
        self.lmax = lmax
        self.dimension_alm = (lmax + 1) ** 2
        self.Npix = Npix

    def sample(self, var_cls):
        return None