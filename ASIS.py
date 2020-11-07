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
                 polarization = False, bins = None, n_iter = 10000, n_iter_metropolis=1, mask_path=None, gibbs_cr = False):
        super().__init__(pix_map, noise, beam, nside, lmax, Npix, polarization = polarization, bins=bins,
                         n_iter = n_iter, gibbs_cr=gibbs_cr)

        if not polarization:
            self.constrained_sampler = CenteredConstrainedRealization(pix_map, noise, self.bl_map, beam, lmax, Npix,
                                                                      mask_path=mask_path)
            self.non_centered_cls_sampler = NonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise, metropolis_blocks,
                                                     proposal_variances, n_iter = n_iter_metropolis, mask_path=mask_path)
            self.centered_cls_sampler = CenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise)
        else:
            self.non_centered_cls_sampler = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise, noise_Q
                                                                 , metropolis_blocks, proposal_variances, n_iter = n_iter_metropolis,
                                                                              mask_path = mask_path)

            self.centered_cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, self.bins, self.bl_map, noise)
            self.constrained_sampler = PolarizedCenteredConstrainedRealization(pix_map, noise, noise_Q, self.bl_map, lmax, Npix, beam,
                                                                               mask_path= mask_path)



    def run_temperature(self, dls_init):
        h_accept_cr = []
        h_accept = []
        h_dls = []
        h_time_seconds = []
        h_time_cpu = []
        binned_dls = dls_init
        dls = utils.unfold_bins(binned_dls, self.bins)
        cls = self.dls_to_cls(dls)
        var_cls_full = utils.generate_var_cl(dls)
        h_dls.append(binned_dls)
        skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), None, metropolis_step=False)
        for i in range(self.n_iter):
            if i % 1 == 0:
                print("Interweaving, iteration:", i)

            start_iteration = time.time()
            #start_clock = time.clock()
            start_time_cr = time.time()
            skymap, accept_cr = self.constrained_sampler.sample(cls[:], var_cls_full[:], skymap, use_gibbs=self.gibbs_cr)
            end_time_cr = time.time()
            total_time_cr = end_time_cr - start_time_cr
            print("Time CR:")
            print(total_time_cr)
            h_accept_cr.append(accept_cr)

            start_time_centered_cls = time.time()
            binned_dls_temp = self.centered_cls_sampler.sample(skymap[:])
            end_time_centered_cls = time.time()
            total_time_centered_cls = end_time_centered_cls - start_time_centered_cls
            print("Time centered cls:", total_time_centered_cls)
            dls_temp_unfolded = utils.unfold_bins(binned_dls_temp, self.bins)
            var_cls_temp = utils.generate_var_cl(dls_temp_unfolded)

            inv_var_cls_temp = np.zeros(len(var_cls_temp))
            np.reciprocal(var_cls_temp, out=inv_var_cls_temp, where=config.mask_inversion)
            s_nonCentered = np.sqrt(inv_var_cls_temp) * skymap

            start_time_noncentered_cls = time.time()
            binned_dls, var_cls_full, accept = self.non_centered_cls_sampler.sample(s_nonCentered[:], binned_dls_temp[:], var_cls_temp[:])
            end_time_noncentered_cls = time.time()
            print("Time non centered cls:")
            print(end_time_noncentered_cls - start_time_noncentered_cls)
            dls = utils.unfold_bins(binned_dls, self.bins)
            cls = self.dls_to_cls(dls)
            skymap = np.sqrt(var_cls_full)*s_nonCentered
            end_iteration = time.time()
            #end_clock = time.clock()
            #time_cpu = end_clock - start_clock
            time_iteration = end_iteration - start_iteration
            h_time_seconds.append(time_iteration)
            #h_time_cpu.append(time_cpu)
            h_accept.append(accept)

            h_dls.append(binned_dls)
            
            #save_path = config.scratch_path + \
            #    "/data/non_isotropic_runs/asis/run/asis_" + str(config.slurm_task_id) + ".npy"
            save_path = "test_nside_512.npy"

            d = {"h_cls":np.array(h_dls), "bins":config.bins, "metropolis_blocks":config.blocks, "h_accept":np.array(h_accept),
             "h_times_iteration":np.array(h_time_seconds),"h_cpu_time":None}

            np.save(save_path, d, allow_pickle=True)

        h_accept = np.array(h_accept)
        print("Acceptance rate ASIS:", np.mean(h_accept, axis = 0))
        print("Acceptance rate constrained realizations:", np.mean(h_accept_cr))
        return np.array(h_dls), np.array(h_accept), np.array(h_accept_cr), np.array(h_time_seconds)


    def run_polarization(self, dls_init):
        accept = []
        h_dls = {"EE":[], "BB":[]}
        binned_dls = dls_init
        for i in range(self.n_iter):
            if i % 100 == 0:
                print("Interweaving, iteration: "+str(i))

            all_dls = {"EE": utils.unfold_bins(binned_dls["EE"], self.bins["EE"]),
                       "BB": utils.unfold_bins(binned_dls["BB"], self.bins["BB"])}
            skymap, _ = self.constrained_sampler.sample(all_dls)
            binned_dls_temp = self.centered_cls_sampler.sample(skymap)
            dls_temp = {"EE":utils.unfold_bins(binned_dls_temp["EE"], self.bins["EE"]), "BB":utils.unfold_bins(binned_dls_temp["BB"], self.bins["BB"])}
            var_cls = {"EE": utils.generate_var_cl(dls_temp["EE"]),
                             "BB": utils.generate_var_cl(dls_temp["BB"])}

            inv_var_EE = np.zeros(len(var_cls["EE"]))
            inv_var_BB = np.zeros(len(var_cls["BB"]))
            inv_var_EE[var_cls["EE"] != 0] = 1/var_cls["EE"][var_cls["EE"] != 0]
            inv_var_BB[var_cls["BB"] != 0] = 1 / var_cls["BB"][var_cls["BB"] != 0]
            s_nonCentered = {"EE": np.sqrt(inv_var_EE)*skymap["EE"], "BB": np.sqrt(inv_var_BB)*skymap["BB"]}
            binned_dls, acception = self.non_centered_cls_sampler.sample(s_nonCentered, binned_dls_temp)
            accept.append(acception)

            h_dls["EE"].append(binned_dls["EE"])
            h_dls["BB"].append(binned_dls["BB"])


        total_accept = np.array(accept)
        print("Interweaving acceptance rate:")
        print(np.mean(total_accept, axis=0))

        h_dls["EE"] = np.array(h_dls["EE"])
        h_dls["BB"] = np.array(h_dls["BB"])

        return h_dls, total_accept, None


    def run(self, dls_init):
        if self.polarization:
            return self.run_polarization(dls_init)
        else:
            return self.run_temperature(dls_init)


