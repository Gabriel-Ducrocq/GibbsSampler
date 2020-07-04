from GibbsSampler import GibbsSampler
from NonCenteredGibbs import NonCenteredClsSampler
from CenteredGibbs import CenteredConstrainedRealization, CenteredClsSampler
from ConstrainedRealization import ConstrainedRealization
from ClsSampler import ClsSampler, MHClsSampler
import utils
import healpy as hp
import numpy as np
from scipy.stats import invgamma
import time
import config
import utils



class ASIS(GibbsSampler):

    def __init__(self, pix_map, noise, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, lmax, Npix, isotropic=True)
        self.non_centered_cls_sampler = NonCenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise, metropolis_blocks,
                                                 proposal_variances, n_iter = n_iter_metropolis)
        self.centered_cls_sampler = CenteredClsSampler(pix_map, lmax, self.bins, self.bl_map, noise)

    def run(self, cls_init):
        h_accept = []
        h_cls = []
        h_time_seconds = []
        binned_cls = cls_init
        cls = utils.unfold_bins(binned_cls, self.bins)
        var_cls_full = utils.generate_var_cl(cls)
        h_cls.append(binned_cls)
        for i in range(self.n_iter):
            if i % 1000 == 0:
                print("Interweaving, iteration:", i)

            start_time = time.process_time()
            skymap, _, _ = self.constrained_sampler.sample(var_cls_full)
            binned_cls_temp = self.centered_cls_sampler.sample(skymap)
            cls_temp_unfolded = utils.unfold_bins(binned_cls_temp, self.bins)
            var_cls_temp = utils.generate_var_cl(cls_temp_unfolded)
            inv_var_cls_temp = np.zeros(len(var_cls_temp))
            np.reciprocal(var_cls_temp, out=inv_var_cls_temp, where=config.mask_inversion)
            s_nonCentered = np.sqrt(inv_var_cls_temp) * skymap
            binned_cls, var_cls_full, accept = self.non_centered_cls_sampler.sample(s_nonCentered, binned_cls_temp, var_cls_temp)
            h_accept.append(accept)

            end_time = time.process_time()
            h_cls.append(binned_cls)
            h_time_seconds.append(end_time - start_time)


        h_accept = np.array(h_accept)
        print("Acceptance rate ASIS:", np.mean(h_accept, axis = 0))
        return np.array(h_cls), np.array(h_accept), np.array(h_time_seconds)