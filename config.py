import numpy as np
import os
import healpy as hp


scratch_path = os.environ['SCRATCH'] #Scratch path to save and load data on NERSC
slurm_task_id = os.environ["SLURM_ARRAY_TASK_ID"] #Slurm ID for using multiple machines on NERSC.


COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"] # Parameters names
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]) # Prior mean
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) # Prior std
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

#NSIDE = 256 # NSIDE for generating the pixel grid over the sphere.
NSIDE = 32
Npix = 12 * NSIDE ** 2 # Number of pixels
L_MAX_SCALARS=int(2*NSIDE) # L_max
##The next lines are the paths to some sky masks. If no mask is used, set mask_path = None.
#mask_path = scratch_path + "/data/non_isotropic_runs/skymask/wamp_temperature_kq85_analysis_mask_r9_9yr_v5.fits"
#mask_path = "wmap_temperature_kq85_analysis_mask_r9_9yr_v5(1).fits"
#mask_path = "HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
mask_path = scratch_path + "/data/non_isotropic_runs/skymask/HFI_Mask_GalPlane-apo0_2048_R2_80%_bis.00.fits"
#mask_path = scratch_path + "/data/simon/cut-sky/skymask/mask_SO_for_Gabriel.fits"
#mask_path = None


def noise_covariance_in_freq(nside):
    ##Prendre les plus basses fréquences pour le bruit (là où il est le plus petit)
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov

noise_covar_temp = 40**2 # temperature noise covariance (NOT stdd)
#Think about changing the noise level again !
##noise_covar_pol = 0.2**2 # polarization noise covariance.
noise_covar_pol = 0.002**2
var_noise_temp = np.ones(Npix) * noise_covar_temp
var_noise_pol = np.ones(Npix) * noise_covar_pol


#The two next lines define the bins. Each integer in the arrays are the start of a bin and the end of another.
##Planck bins:
#bins_BB = np.concatenate([range(0, 396), np.array([396, 398, 400, 402, 406, 410, 415, 420, 425, 430, 435, 440, 445, 460, 475, 495, 513])])
#bins = {"EE":np.array(range(0, L_MAX_SCALARS+2)), "BB":bins_BB}
bins = {"EE":np.array(range(0, L_MAX_SCALARS+2)), "BB":np.array(range(0, L_MAX_SCALARS+2))}
#bins = {"EE":np.array(range(0, L_MAX_SCALARS+2)), "BB":np.array(range(0, L_MAX_SCALARS+2))}

#The two next lines define the blocking scheme use for the non centered power spectrum sampling step.
blocks_EE = [2, len(bins["EE"])]
blocks_BB = np.concatenate([[2, 279], np.arange(280, len(bins["BB"]), 1)])
#blocks_BB = [2, len(bins["EE"])]

blocks = {"EE":blocks_EE, "BB":blocks_BB}


#Defining the blocking schemes for every algorithms.
metropolis_blocks_gibbs_nc = blocks
metropolis_blocks_gibbs_asis = blocks
metropolis_blocks_pncp = blocks


# Number of iterations of each algorithm.
N_gibbs = 100
N_nc_gibbs = 10000
N_rescale = 1000
N_ASIS = 10000



rescaling_map2alm = (Npix / (4 * np.pi)) # float, the rescaling factor to get the adjoint synthesis matrix instead of the analysis matrix.
w = 4 * np.pi / Npix # Size of a pixel.

def generate_var_cl(cls_):
    """

    :param cls_: array of float of size (L_max + 1,). C_\ell
    :return: The diagonal of the covariance matrix C of the alms exxpressed in real convention, see paper.
    """
    var_cl_full = np.concatenate([cls_,
                                  np.array(
                                      [cl for m in range(1, L_MAX_SCALARS + 1) for cl in cls_[m:] for _ in range(2)])])
    return var_cl_full


beam_fwhm = 0.5 #Beam fwhm in degrees.
fwhm_radians = (np.pi / 180) * beam_fwhm # Beam fwhm in radians
bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=L_MAX_SCALARS) #array of size (L_max +1,) of the b_\ell coefficients.
bl_map = generate_var_cl(bl_gauss) # Diagonal of the diagonal beam matrix B, see paper.


def compute_init_values_pol(unbinned_vars, pol):
    """

    :param unbinned_vars: array of size (L_max +1,) of the variances on the D_\ell
    :param pol: string "EE" or "BB", the power spectrum to bin
    :return: array of size (number of bins,)
    """
    vals = []
    for i, l_start in enumerate(bins[pol][:-1]):
        l_end = bins[pol][i+1]
        length = l_end - l_start
        vals.append(np.mean(unbinned_vars[l_start:l_end])/length)

    return np.array(vals)


def compute_init_values(unbinned_vars):
    vals = []
    for i, l_start in enumerate(bins[:-1]):
        l_end = bins[i+1]
        length = l_end - l_start
        vals.append(np.mean(unbinned_vars[l_start:l_end])/length)

    return np.array(vals)


scale = [((l*(l+1))**2*2/(4*np.pi**2*(2*l+1))) for l in range(0, L_MAX_SCALARS+1)] # Scaling factor to go from C_\ell to D_\ell
if mask_path is None:
    # If not mask is applied, compute the proposal covariances for the non centered move. See paper.
    unbinned_variances = (w*noise_covar_temp/bl_gauss**2)**2*scale
    unbinned_variances_pol = (w*noise_covar_pol/bl_gauss**2)**2*scale
else:
    # If mask is applied, we compute the same proposal covariances with 1/fraction_of_the_sky_we_observe factor.
    mask = hp.ud_grade(hp.read_map(mask_path), NSIDE)
    unbinned_variances_pol = (w * noise_covar_pol / bl_gauss ** 2) ** 2 * scale* 1/np.mean(mask)
    unbinned_variances = (w * noise_covar_temp / bl_gauss ** 2) ** 2 * scale * 1 / np.mean(mask)


#Create the variances for the proposal of the non centered move for the binned power spectrum.
binned_variances_pol = {"EE":compute_init_values_pol(unbinned_variances_pol, "EE"), "BB":compute_init_values_pol(unbinned_variances_pol, "BB")}



def get_proposal_variances_preliminary(path):
    """
    computing an approximation of the posterior variances and mean of D_\ell for "TT" only.

    :param path: string, path to the data of a preliminary run.
    :return: arrays of floats, of size (number of bins,). The first one is the estimated variances of the posterior for each D_\ell and the
            second one is the estimated mean of the posterior for each D_\ell.
    """
    list_files = os.listdir(path)
    chains = []

    for i, name in enumerate(list_files):
        if name not in [".ipynb_checkpoints", "Untitled.ipynb", "preliminary_runs"]:
            data = np.load(path + name, allow_pickle=True)
            data = data.item()
            chains.append(data["h_cls"])

        chains = np.array(chains)
        variances = np.var(chains[:, 200:, :], axis=(0, 1))
        means = np.mean(chains[:, 200:, :], axis=(0, 1))

        return variances, means



def get_proposal_variances_preliminary_pol(path):
    """
    computing an approximation of the posterior variances and mean of D_\ell for "EE" and "BB" only. Same function
    as above but for polarization.

    """
    list_files = os.listdir(path)
    chains = []

    print(list_files)
    for i, name in enumerate(list_files):
        if name not in [".ipynb_checkpoints",  '.ipynb_checkpoints', "Untitled.ipynb", "preliminary_runs"]:
            data = np.load(path + name, allow_pickle=True)
            data = data.item()
            chains.append(data["h_cls"])

    all_paths = {"EE":[], "BB":[]}
    for pol in ["EE", "BB"]:
        for i, chain in enumerate(chains):
            all_paths[pol].append(chain[pol][:, :])

    all_paths["EE"] = np.array(all_paths["EE"])
    all_paths["BB"] = np.array(all_paths["BB"])
    variances = {"EE":np.var(all_paths["EE"][:, :, :], axis=(0, 1)), "BB":np.var(all_paths["BB"][:, :, :], axis=(0, 1))}
    means = {"EE": np.mean(all_paths["EE"][:, :, :], axis=(0, 1)),
                    "BB": np.mean(all_paths["BB"][:, :, :], axis=(0, 1))}

    return variances, means



preliminary_run = True # If the run is preliminary tuning run or not
if preliminary_run:
    #If it is, we juste use some variances we set.
    proposal_variances_nc_polarized = {}
    proposal_variances_nc_polarized["EE"] = binned_variances_pol["EE"][2:]
    proposal_variances_nc_polarized["BB"] = binned_variances_pol["BB"][2:]

else:
    #If a preliminary run has already been done, we download the results, estimate the mean and variance of the
    # posterior. This will serve as proposal variances for the non cnetered move. We rescale it by an arbitrary factor to achieve acceptance rates
    # near 25%.
#for planck 80% skymask:
    path_vars = scratch_path + "/data/polarization_runs/cut_sky/asis/preliminary_run/"
    empirical_variances, starting_point = get_proposal_variances_preliminary_pol(path_vars)
    starting_point["EE"][:2] = 0
    starting_point["BB"][:2] = 0

    bl = blocks["BB"][:-1]
    proposal_variances_nc_polarized = {}
    proposal_variances_nc_polarized["EE"] = empirical_variances["EE"]*3
    proposal_variances_nc_polarized["BB"] = empirical_variances["BB"]

    proposal_variances_nc_polarized["BB"][bl[-3]:] *= 1.5
    proposal_variances_nc_polarized["BB"][bl[-11]:bl[-4]] *= 2
    proposal_variances_nc_polarized["BB"][bl[-133]:bl[-11]] *= 3.5
    proposal_variances_nc_polarized["BB"][bl[-134]] *= 0.00001

    proposal_variances_nc_polarized["BB"] *= 1.8
    proposal_variances_nc_polarized["EE"] *= 0.0001



    proposal_variances_nc_polarized["EE"] = proposal_variances_nc_polarized["EE"][2:]
    proposal_variances_nc_polarized["BB"] = proposal_variances_nc_polarized["BB"][2:]

