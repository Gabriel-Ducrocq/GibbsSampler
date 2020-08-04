import config
import utils
import numpy as np
from scipy.stats import invgamma, truncnorm
import healpy as hp
import time
from default_gibbs import sample_cls


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
            if not polarization:
                self.bins = np.array([l for l in range(lmax+2)])
            else:
                bins = np.array([l for l in range(2, lmax + 1)])
                self.bins = {"TT":bins, "EE":bins, "TE":bins, "BB":bins}
        else:
            self.bins = bins

        self.dls_to_cls_array = np.array([2*np.pi/(l*(l+1)) if l !=0 else 0 for l in range(lmax+1)])

    def dls_to_cls(self, dls_):
        return dls_[:]*self.dls_to_cls_array

    def compute_bl_map(self, beam_fwhm):
        fwhm_radians = (np.pi / 180) * beam_fwhm
        bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=self.lmax)
        bl_map = np.concatenate([bl_gauss,np.array([cl for m in range(1, self.lmax + 1) for cl in bl_gauss[m:] for _ in range(2)])])
        return bl_map

    def run(self, dls_init):
        h_accept_cr = []
        h_dls = []
        h_time_seconds = []
        binned_dls = dls_init
        dls = utils.unfold_bins(binned_dls, config.bins)
        cls = self.dls_to_cls(dls)
        var_cls_full = utils.generate_var_cl(dls)
        skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), None, metropolis_step=False)
        for i in range(10000):
            if i % 1000 == 0:
                print("Default Gibbs")
                print(i)

            start_time = time.process_time()
            binned_dls = self.cls_sampler.sample(skymap[:])
            dls = utils.unfold_bins(binned_dls, config.bins)
            cls = self.dls_to_cls(dls)
            var_cls_full = utils.generate_var_cl(dls)

            skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), skymap, metropolis_step=True)

            h_accept_cr.append(accept)
            end_time = time.process_time()
            h_dls.append(binned_dls)
            h_time_seconds.append(end_time - start_time)

        print("Acception rate constrained realization:", np.mean(h_accept_cr))
        return np.array(h_dls), h_time_seconds








