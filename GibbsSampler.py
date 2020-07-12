import config
import utils
import numpy as np
from scipy.stats import invgamma, truncnorm
import healpy as hp
import time


class GibbsSampler():
    def __init__(self, pix_map, noise, beam_fwhm, nside, lmax, Npix, polarization = False, bins=None, n_iter = 10000):
        self.noise = noise
        self.beam = beam_fwhm
        self.nside = nside
        self.lmax = lmax
        self.polarization = polarization
        self.bins = bins
        self.pix_map = pix_map
        self.Npix = Npix
        self.bl_map = self.compute_bl_map(beam_fwhm)
        self.constrained_sampler = None
        self.cls_sampler = None
        self.n_iter = n_iter
        if bins == None:
            self.bins = np.array([l for l in range(lmax+2)])
        else:
            self.bins = bins

    def compute_bl_map(self, beam_fwhm):
        fwhm_radians = (np.pi / 180) * beam_fwhm
        bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=self.lmax)
        bl_map = np.concatenate([bl_gauss,np.array([cl for m in range(1, self.lmax + 1) for cl in bl_gauss[m:] for _ in range(2)])])
        return bl_map

    def run(self, cls_init):
        h_cls = []
        h_time_seconds = []
        binned_cls = cls_init
        h_cls.append(binned_cls)
        for i in range(self.n_iter):
            if i % 1 == 0:
                print("Default Gibbs, iteration:", i)

            start_time = time.process_time()
            if not self.polarization:
                cls = utils.unfold_bins(binned_cls, self.bins)
                var_cls_full = utils.generate_var_cl(cls)
            else:
                cls = binned_cls
                var_cls_full = cls


            alm_map, time_to_solution, err = self.constrained_sampler.sample(var_cls_full)
            binned_cls = self.cls_sampler.sample(alm_map)

            end_time = time.process_time()
            h_cls.append(binned_cls)
            h_time_seconds.append(end_time - start_time)

        return np.array(h_cls), h_time_seconds








