from GibbsSampler import GibbsSampler
from NonCenteredGibbs import NonCenteredClsSampler, PolarizedNonCenteredConstrainedRealization, PolarizationNonCenteredClsSampler
from CenteredGibbs import CenteredConstrainedRealization, CenteredClsSampler, PolarizedCenteredConstrainedRealization, PolarizedCenteredClsSampler
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
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1, mask_path=None, gibbs_cr = False,
                 rj_step=False, all_sph=False, n_gibbs = 20, overrelaxation= False):
        """
        Class for Interweaving

        :param pix_map: array of floats, size Npix, observed skymap d.
        :param noise: array of floats, size Npix, noise level for each pixel
        :param noise_Q: array of floats, size Npix, noise level for each pixel. Same for Q and U maps
        :param beam: float, definition of the beam in degree.
        :param nside: integer, nside used to generate the grid over the sphere.
        :param lmax: integer, L_max.
        :param Npix: integer, number of pixels
        :param proposal_variances: dict {"EE": array, "BB":array}, arrays of float, variances of the proposal distributions. NOT of size L_max, but of size number of blocks.
        :param metropolis_blocks: dict {"EE":array, "BB":array}, integers, starting and ending indexes of the blocks.
        :param polarization: boolean, whether we are dealing with "TT" only or "EE" and "BB" only.
        :param bins: dict {"EE":array, "BB":array}, integers, starting and ending indexes of the bins.
        :param n_iter: integer, number of iterations of the Gibbs sampler to do.
        :param n_iter_metropolis: integer, number of iterations of the Metropolis-within-Gibbs sampler to do.
        :param mask_path: string, path to a sky mask. If None, no sky mask is used.
        :param gibbs_cr: boolean. If True, use the auxiliary varialbe scheme instead of PCG solver.
        :param rj_step: boolean. If True, use a RPJO step.
        :param all_sph: boolean, if True, write the entire model in spherical harmonics basis. If False do as usual.
        :param n_gibbs: integer, number of auxiliary variable steps.
        """
        super().__init__(pix_map, noise, beam, nside, lmax, polarization = polarization, bins=bins,
                         n_iter = n_iter, gibbs_cr=gibbs_cr, rj_step=rj_step)

        if not polarization:
            #Creation of the CR sampler for "TT" only:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, beam, lmax, Npix,
                                                                      mask_path=mask_path)
            #Creation of the Non centered Cls sampler for "TT" only:
            self.non_centered_cls_sampler = NonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise, metropolis_blocks,
                                                     proposal_variances, n_iter = n_iter_metropolis, mask_path=mask_path)
            #Creation of the centered Cls sampler for "TT" only:
            self.centered_cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise)
        else:
            #Creation of the non centered Cls sampler for "EE" and "BB" only
            self.non_centered_cls_sampler = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise, noise_Q
                                                                 , metropolis_blocks, proposal_variances, n_iter = n_iter_metropolis,
                                                                              mask_path = mask_path, all_sph=all_sph)
            #Creation of the centered Cls sampler for "EE" and "BB" only.
            self.centered_cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise)
            #Creation of the CR sampler for "EE" and "BB" only.
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise, noise_Q, self.bl_map, lmax, Npix, beam,
                                                                               mask_path= mask_path,
                                                                               gibbs_cr = gibbs_cr, n_gibbs = n_gibbs, overrelaxation=overrelaxation)



    def run_temperature(self, dls_init):
        """
        Run temperature only Interweaving

        :param dls_init: dls_init: dict {"EE":array, "BB":array}, arrays of float, initial D_\ell
        :return: array of size (n_iter, L_max +1) being the trajectory of the Gibbs sampler. Array of int of size (n_iter, n_blocks)
                being the history of acceptances of the M-w-G algo for each block. Array of floats, history of CR execution times.
                Array of floats, history of the execution times of the Interweaving iteration.
        """
        h_accept_cr = []
        h_accept = []
        h_dls = []
        h_time_seconds = []
        binned_dls = dls_init
        dls = utils.unfold_bins(binned_dls, self.bins)
        cls = self.dls_to_cls(dls)
        var_cls_full = utils.generate_var_cl(dls)
        h_dls.append(binned_dls)
        skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), None, metropolis_step=False)# Make a first constrained realization before starting the iterations
        for i in range(self.n_iter):
            if i % 1 == 0:
                print("Interweaving, iteration:", i)

            start_iteration = time.time()
            start_time_cr = time.time()
            skymap, accept_cr = self.constrained_sampler.sample(cls[:], var_cls_full[:], skymap, use_gibbs=self.gibbs_cr) # Do CR step.
            end_time_cr = time.time()
            total_time_cr = end_time_cr - start_time_cr
            print("Time CR:")
            print(total_time_cr)
            h_accept_cr.append(accept_cr)

            start_time_centered_cls = time.time()
            binned_dls_temp = self.centered_cls_sampler.sample(skymap[:]) # Make centered power spectrum sampling move.
            end_time_centered_cls = time.time()
            total_time_centered_cls = end_time_centered_cls - start_time_centered_cls
            print("Time centered cls:", total_time_centered_cls)
            dls_temp_unfolded = utils.unfold_bins(binned_dls_temp, self.bins) # Unfold the power spectrum.
            var_cls_temp = utils.generate_var_cl(dls_temp_unfolded) # Generate the associated variance.

            inv_var_cls_temp = np.zeros(len(var_cls_temp))
            np.reciprocal(var_cls_temp, out=inv_var_cls_temp, where=config.mask_inversion) # Invert the variance
            s_nonCentered = np.sqrt(inv_var_cls_temp) * skymap # Turn the centered skymap to the non centered skymap

            start_time_noncentered_cls = time.time()
            binned_dls, var_cls_full, accept = self.non_centered_cls_sampler.sample(s_nonCentered[:], binned_dls_temp[:], var_cls_temp[:]) # Make a non centered power spectrum sampling step.
            end_time_noncentered_cls = time.time()
            print("Time non centered cls:")
            print(end_time_noncentered_cls - start_time_noncentered_cls)
            dls = utils.unfold_bins(binned_dls, self.bins)
            cls = self.dls_to_cls(dls)
            skymap = np.sqrt(var_cls_full)*s_nonCentered # Re-center the skymap with the newly sampled power spectrum.
            end_iteration = time.time()
            time_iteration = end_iteration - start_iteration
            h_time_seconds.append(time_iteration)
            h_accept.append(accept)

            h_dls.append(binned_dls)

        h_accept = np.array(h_accept)
        print("Acceptance rate ASIS:", np.mean(h_accept, axis = 0))
        print("Acceptance rate constrained realizations:", np.mean(h_accept_cr))
        return np.array(h_dls), np.array(h_accept), np.array(h_accept_cr), np.array(h_time_seconds)


    def run_polarization(self, dls_init):
        """

        :param dls_init: dls_init: dls_init: dict {"EE":array, "BB":array}, arrays of float, initial D_\ell
        :return: array of size (n_iter, L_max +1) being the trajectory of the Gibbs sampler. Array of int of size (n_iter,)
                acceptances of the CR step, array of int, size (n_iter,), history of the acceptances for the CR step. array floats size (n_iter,), history of the durations of each iteration.
                Array, size (n_iter,), history of the durations of the CR step. Array, size (n_iter,) durations of the centered Cls sampling
                Array, size (n_iter,) history of the durations of the Cls non centered Cls sampling step.

        """
        h_duration_cr = []
        h_duration_cls_nc_sampling = []
        h_duration_cls_sampling = []
        h_iteration_duration = []
        accept = {"EE":[], "BB":[]}
        accept_cr = []
        h_dls = {"EE":[], "BB":[]}
        binned_dls = dls_init
        all_dls = {"EE": utils.unfold_bins(binned_dls["EE"], self.bins["EE"]),
                   "BB": utils.unfold_bins(binned_dls["BB"], self.bins["BB"])}
        if self.rj_step == True or self.gibbs_cr == True:
            ##If we use a RJPO or auxiliary variable algo, we need an initial skymap. We draw it with a PCG resolution.
            skymap, _ = self.constrained_sampler.sample(all_dls)

        for i in range(self.n_iter):
            if i % 1 == 0:
                print("Interweaving, iteration: "+str(i))

            start_iteration = time.clock()
            start_time = time.clock()
            if self.rj_step is False and self.gibbs_cr is False:
                # If not RJPO nor auxiliary, use PCG resolution for CR step.
                skymap, _ = self.constrained_sampler.sample(all_dls)
            else:
                #Otherwise, use RJPO or auxiliary variable step for CR step.
                skymap, acc = self.constrained_sampler.sample(all_dls, skymap)
                accept_cr.append(acc)

            end_time = time.clock()
            duration = end_time - start_time
            h_duration_cr.append(duration)

            start_time = time.clock()
            binned_dls_temp = self.centered_cls_sampler.sample(skymap) # Make a centered power spectrum sampling move.
            end_time = time.clock()
            duration =end_time - start_time
            h_duration_cls_sampling.append(duration)
            dls_temp = {"EE":utils.unfold_bins(binned_dls_temp["EE"], self.bins["EE"]), "BB":utils.unfold_bins(binned_dls_temp["BB"], self.bins["BB"])}
            var_cls = {"EE": utils.generate_var_cl(dls_temp["EE"]),
                             "BB": utils.generate_var_cl(dls_temp["BB"])}

            inv_var_EE = np.zeros(len(var_cls["EE"]))
            inv_var_BB = np.zeros(len(var_cls["BB"]))
            inv_var_EE[var_cls["EE"] != 0] = 1/var_cls["EE"][var_cls["EE"] != 0]#Invert the variance
            inv_var_BB[var_cls["BB"] != 0] = 1 / var_cls["BB"][var_cls["BB"] != 0] # Same here
            s_nonCentered = {"EE": np.sqrt(inv_var_EE)*skymap["EE"], "BB": np.sqrt(inv_var_BB)*skymap["BB"]} # Compute the non centered sky map.
            start_time = time.clock()
            binned_dls, acception = self.non_centered_cls_sampler.sample(s_nonCentered, binned_dls_temp) # Power spectrum sampling in the non centered parametrization.
            end_time = time.clock()
            duration =end_time - start_time
            h_duration_cls_nc_sampling.append(duration)
            accept["EE"].append(acception["EE"])
            accept["BB"].append(acception["BB"])

            all_dls = {"EE": utils.unfold_bins(binned_dls["EE"], self.bins["EE"]),
                       "BB": utils.unfold_bins(binned_dls["BB"], self.bins["BB"])}

            var_cls = {"EE": utils.generate_var_cl(all_dls["EE"]),
                             "BB": utils.generate_var_cl(all_dls["BB"])}
            skymap = {"EE": np.sqrt(var_cls["EE"])*skymap["EE"], "BB": np.sqrt(var_cls["BB"])*skymap["BB"]} # compute the centered skymap from the non centered pow spec.
            
            h_dls["EE"].append(binned_dls["EE"])
            h_dls["BB"].append(binned_dls["BB"])
            end_iteration = time.clock()
            h_iteration_duration.append(end_iteration - start_iteration)

        total_accept = {"EE":np.array(accept["EE"]), "BB":np.array(accept["BB"])}
        print("Interweaving acceptance rate EE:")
        print(np.mean(total_accept["EE"], axis=0))
        print("Interweaving acceptance rate BB:")
        print(np.mean(total_accept["BB"], axis=0))
        if self.rj_step is True:
            print("Acceptance rate constrained realization:")
            print(np.mean(accept_cr))

        h_dls["EE"] = np.array(h_dls["EE"])
        h_dls["BB"] = np.array(h_dls["BB"])

        if not self.rj_step:
            return h_dls, total_accept, None, np.array(h_iteration_duration), np.array(h_duration_cr), np.array(h_duration_cls_sampling), np.array(h_duration_cls_nc_sampling)

        else:
            return h_dls, total_accept, np.array(accept_cr), np.array(h_iteration_duration), np.array(h_duration_cr), np.array(h_duration_cls_sampling), np.array(h_duration_cls_nc_sampling)

    def run(self, dls_init):
        if self.polarization:
            return self.run_polarization(dls_init)
        else:
            return self.run_temperature(dls_init)


