import config
import utils
import numpy as np
import healpy as hp
import time


class GibbsSampler():
    def __init__(self, pix_map, noise, beam_fwhm_deg, nside, lmax, polarization = False, bins=None, n_iter = 10000,
                 gibbs_cr = False, rj_step = False, ula = False):
        """
        This is the basic Gibbs sampler class, from which any variant will inherit.
        The set of alm coefficients is in m major, as expected by the healpy library.

        :param pix_map: array of size Npix, observed pixel map, called d in the equations
        :param noise: float, noise level. This assumes a noise covariance matrix prop to identity.
        :param beam_fwhm_deg: float, definition of the beam in degree.
        :param nside: int, NSIDE used to define the grid over the sphere.
        :param lmax: int, maximum \ell on which we perform inference
        :param polarization: boolean, whether we are making inference on EE and BB or just TT.
        :param bins: array of integers, each element is the starting index and ending index of a power spectrum bin.
        :param n_iter: integer, number of iterations of Gibbs sampler to perform.
        :param gibbs_cr: whether we use the classical PCG resolution for the constrained realization step, or the augmented
                variable scheme described in he paper.
        :param rj_step: boolean,  whether we use the PCG resolution for the CR step or the RJPO algorithm.
        """
        self.noise = noise
        self.beam = beam_fwhm_deg
        self.nside = nside
        self.lmax = lmax
        self.polarization = polarization
        self.bins = bins
        self.pix_map = pix_map
        self.Npix = 12*nside**2
        self.bl_map = self.compute_bl_map(beam_fwhm_deg)
        self.constrained_sampler = None
        self.cls_sampler = None
        self.n_iter = n_iter
        self.gibbs_cr = gibbs_cr
        self.rj_step = rj_step
        self.ula = ula
        if bins is None:
            ## If the user provides no binning scheme, then each bin is made of only one \ell. This equivalent to
            ## no binning.
            if not polarization:
                self.bins = np.array([l for l in range(lmax+2)])
            else:
                bins = np.array([l for l in range(2, lmax + 1)])
                self.bins = {"TT":bins, "EE":bins, "TE":bins, "BB":bins}
        else:
            self.bins = bins

        # This is the array we use to convert D_\ell to C_\ell
        self.dls_to_cls_array = np.array([2*np.pi/(l*(l+1)) if l !=0 else 0 for l in range(lmax+1)])

    def dls_to_cls(self, dls_):
        """

        :param dls_: array of floats, representing the D_\ells
        :return: array of floats, representing the C_\ells
        """
        return dls_[:]*self.dls_to_cls_array

    def compute_bl_map(self, beam_fwhm_deg):
        """

        :param beam_fwhm_deg: float, definition of the beam in degree
        :return: array of floats, the b_\ells expension over all the spherical harmonics coefficients. This is the diagonal
                of the B matrix in the paper.
        """
        fwhm_radians = (np.pi / 180) * beam_fwhm_deg
        bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=self.lmax)
        bl_map = np.concatenate([bl_gauss,np.array([cl for m in range(1, self.lmax + 1) for cl in bl_gauss[m:] for _ in range(2)])])
        return bl_map

    def run_temperature(self, dls_init):
        """
        This function runs the Gibbs sampler on temperature only.

        :param dls_init: array of floats, initial binned D_\ell of the Gibbs sampler.
        :return: array of float, size (n_iter, lmax), being the trajectory of the Gibbs sampler.
                array of float, size n_iter, being the history of acceptances of the contrained realization algorithm being used.
                array of float, size n_iter, being the the history of cpu execution time in second for each iteration.
        """
        h_accept_cr = []
        h_dls = []
        h_time_seconds = []
        binned_dls = dls_init
        dls = utils.unfold_bins(binned_dls, config.bins) #Expending the binned D_\ell to the unbinned size vector of D_\ell
        cls = self.dls_to_cls(dls) # turn D_\ell to C_\ell
        var_cls_full = utils.generate_var_cl(dls) # Expand the array of C_\ell into an array of the size of alm array, m major.
        skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), None, metropolis_step=False) #make a first CR step.
        ## Note that this first CR step is useless if we use a regular PCG in the CR step in the loop. Otherwise, we actually need an
        ## intialization of the map before the first iteration, and it is natural to make a first CR step to get it.
        h_dls.append(binned_dls)
        for i in range(self.n_iter):
            if i % 1 == 0:
                print("Default Gibbs")
                print(i)

            start_time = time.process_time()
            skymap, accept = self.constrained_sampler.sample(cls[:], var_cls_full.copy(), skymap, metropolis_step=False,
                                                             use_gibbs=False) #Make a CR step. first step of Gibbs sampler.
            binned_dls = self.cls_sampler.sample(skymap[:]) #Sample the binned D_\ells, second step of Gibbs sampler
            dls = utils.unfold_bins(binned_dls, self.bins) #Unfold the binned D_\ell
            cls = self.dls_to_cls(dls) #Turn it into C_\ell
            var_cls_full = utils.generate_var_cl(dls) # Expand the array of C_\ell

            ##Then we keep all the information in memory for output:
            h_accept_cr.append(accept)
            end_time = time.process_time()
            h_dls.append(binned_dls)
            h_time_seconds.append(end_time - start_time)

        print("Acception rate constrained realization:", np.mean(h_accept_cr))
        return np.array(h_dls), np.array(h_accept_cr), h_time_seconds

    def run_polarization(self, dls_init):
        """
        This one runs the Gibbs sampler on EE and BB only.

        :param dls_init: dictionnary of arrays of float, {"EE":array1, "BB":array2}, arrays being the binned D_\ell of polarization.
        :return: dict of array of float, size (n_iter, lmax), being the trajectory of the Gibbs sampler.
                array of float, size n_iter, being the history of acceptances of the contrained realization algorithm being used.
                array of float, size n_iter, being the the history of cpu execution time in second for the CR step.
                array of float, size n_iter, being the the history of cpu execution time in second for the Pow Spec sampling step.
        """

        #Intialize everything
        h_accept_cr = []
        h_duration_cr = []
        h_duration_cls_sampling = []
        h_dls = {"EE":[], "BB":[]}
        binned_dls = dls_init
        dls_unbinned = {"EE":utils.unfold_bins(binned_dls["EE"].copy(), self.bins["EE"]), "BB":utils.unfold_bins(binned_dls["BB"].copy(), self.bins["BB"])}
        if self.rj_step == True or self.gibbs_cr == True or self.ula==True:
            ## If we use a RJPO algo or auxiliary variable scheme instead of regular PCG, we need this initialization of the skymap.
            skymap, accept = self.constrained_sampler.sample(dls_unbinned)

        h_dls["EE"].append(binned_dls["EE"])
        h_dls["BB"].append(binned_dls["BB"])
        for i in range(self.n_iter):
            if i % 10 == 0:
                if not self.ula:
                    print("Default Gibbs")
                else:
                    print('ULA')

                print(i)

            start_time = time.clock()
            if self.rj_step is False and self.gibbs_cr is False and self.ula is False:
                ## If we the PCG, CR step
                skymap, _ = self.constrained_sampler.sample(dls_unbinned.copy())
            else:
                ## If we use the PCG or the auxiliary variable scheme, different CR step
                skymap, accept = self.constrained_sampler.sample(dls_unbinned.copy(), skymap)
                h_accept_cr.append(accept)

            end_time = time.clock()
            duration = end_time - start_time
            h_duration_cr.append(duration)

            start_time = time.clock()
            binned_dls = self.cls_sampler.sample(skymap.copy()) # Sample the binned D_\ell
            end_time = time.clock()
            duration =end_time - start_time
            h_duration_cls_sampling.append(duration)
            ## Unbin the power spectrum for the next iteration:
            dls_unbinned = {"EE":utils.unfold_bins(binned_dls["EE"].copy(), self.bins["EE"]), "BB":utils.unfold_bins(binned_dls["BB"].copy(), self.bins["BB"])}
            
            h_dls["EE"].append(binned_dls["EE"])
            h_dls["BB"].append(binned_dls["BB"])

        if self.rj_step == True or self.ula == True:
            print("Acception rate constrained realization:", np.mean(h_accept_cr))

        h_dls["EE"] = np.array(h_dls["EE"])
        h_dls["BB"] = np.array(h_dls["BB"])
        return h_dls, np.array(h_accept_cr), np.array(h_duration_cr), np.array(h_duration_cls_sampling)


    def run(self, dls_init):
        """
        This function runs either the temperature or polarization only Gibbs sampler.
        :param dls_init: initial binned D_\ell
        :return: return whatever the run_temperature/polarization returns.
        """
        if not self.polarization:
            return self.run_temperature(dls_init)
        else:
            return self.run_polarization(dls_init)








