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

    def __init__(self, pix_map, noise, noise_Q, beam, nside, lmax, Npix, proposal_variances, metropolis_blocks = None,
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1, mask_path=None):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins, n_iter = n_iter)
        self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, beam, lmax, Npix, mask_path=mask_path)
        self.non_centered_cls_sampler = NonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise, metropolis_blocks,
                                                 proposal_variances, n_iter = n_iter_metropolis, mask_path=mask_path)
        self.centered_cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise)

    def run(self, dls_init):
        h_accept_cr = []
        h_accept = []
        h_dls = []
        h_time_seconds = []
        binned_dls = dls_init
        print(self.bins)
        print(binned_dls)
        dls = utils.unfold_bins(binned_dls, self.bins)
        cls = self.dls_to_cls(dls)
        var_cls_full = utils.generate_var_cl(dls)
        h_dls.append(binned_dls)
        skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), None, metropolis_step=False)
        for i in range(self.n_iter):
            if i % 10 == 0:
                print("Interweaving, iteration:", i)

            start_time_cr = time.process_time()
            skymap, accept_cr = self.constrained_sampler.sample(cls[:], var_cls_full[:], skymap, use_gibbs=False)
            end_time_cr = time.process_time()
            total_time_cr = end_time_cr - start_time_cr
            print("Time CR:")
            print(total_time_cr)
            h_accept_cr.append(accept_cr)

            start_time_centered_cls = time.time()
            binned_dls_temp = self.centered_cls_sampler.sample(skymap[:])
            end_time_centered_cls = time.time()
            total_time_centered_cls =end_time_centered_cls - start_time_centered_cls
            print("Time centered cls:", total_time_centered_cls)
            dls_temp_unfolded = utils.unfold_bins(binned_dls_temp, self.bins)
            var_cls_temp = utils.generate_var_cl(dls_temp_unfolded)

            inv_var_cls_temp = np.zeros(len(var_cls_temp))
            np.reciprocal(var_cls_temp, out=inv_var_cls_temp, where=config.mask_inversion)
            end_time_noncentered_cls = time.time()
            s_nonCentered = np.sqrt(inv_var_cls_temp) * skymap

            start_time_noncentered_cls = time.time()
            binned_dls, var_cls_full, accept = self.non_centered_cls_sampler.sample(s_nonCentered[:], binned_dls_temp[:], var_cls_temp[:])
            print("Time non centered cls:")
            print(end_time_noncentered_cls - start_time_noncentered_cls)
            dls = utils.unfold_bins(binned_dls, self.bins)
            cls = self.dls_to_cls(dls)
            skymap = np.sqrt(var_cls_full)*s_nonCentered
            h_accept.append(accept)

            h_dls.append(binned_dls)

        h_accept = np.array(h_accept)
        print("Acceptance rate ASIS:", np.mean(h_accept, axis = 0))
        print("Acceptance rate constrained realizations:", np.mean(h_accept_cr))
        return np.array(h_dls), np.array(h_accept), np.array(h_accept_cr), np.array(h_time_seconds)