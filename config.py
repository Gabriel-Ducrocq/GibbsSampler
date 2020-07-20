import numpy as np
import os
import healpy as hp


#scratch_path = os.environ['SCRATCH']
#slurm_task_id = os.environ["SLURM_ARRAY_TASK_ID"]

def compute_observed_spectrum(d):
    observed_cls = []
    for l in range(2, L_MAX_SCALARS + 1):
        piece_d = d[l ** 2 - 4:(l + 1) ** 2 - 4]
        observed_cl = (np.abs(piece_d[0]) ** 2 + 2 * np.sum(piece_d[1:] ** 2)) / (2 * l + 1)
        observed_cls.append(observed_cl)

    return observed_cls


# COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
COSMO_PARAMS_SIGMA_PRIOR_UNIF_inf = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047]) \
                                    - np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014]) * 10
COSMO_PARAMS_SIGMA_PRIOR_UNIF_sup = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047]) \
                                    + np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014]) * 10
COSMO_PARAMS_SIGMA_PROP = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]) * 5
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])
COSMO_PARAMS_PLANCK = np.array([0.9635, 0.02213, 0.12068, 1.04096, 3.0413, 0.0523])

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "Omega_Lambda", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_WMAP = [0.961, 0.02268, 0.1081, 0.751, 2.41, 0.089]
COSMO_WAMP_MEAN = [0.963, 0.02273, 0.1099, 0.742, 2.41, 0.087]

test_COSMO = [0.972, 0.02264, 0.1138, 1.04101, 3.047, 0.088]

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

N_MAX_PROCESS = 40

N_Stoke = 1
NSIDE = 4
Npix = 12 * NSIDE ** 2
L_MAX_SCALARS=int(2*NSIDE)
#L_MAX_SCALARS = 1000
dimension_sph = int((L_MAX_SCALARS * (L_MAX_SCALARS + 1) / 2) + L_MAX_SCALARS + 1)
dimension_h = (L_MAX_SCALARS + 1) ** 2

mask_inversion = np.ones((L_MAX_SCALARS + 1) ** 2) == 1
mask_inversion[[0, 1, L_MAX_SCALARS + 1, L_MAX_SCALARS + 2]] = False

mask_inv_temp = np.ones(len(mask_inversion)) == 0


def noise_covariance_in_freq(nside):
    ##Prendre les plus basses fréquences pour le bruit (là où il est le plus petit)
    cov = LiteBIRD_sensitivities ** 2 / hp.nside2resol(nside, arcmin=True) ** 2
    return cov


noise_covar_one_pix = noise_covariance_in_freq(NSIDE)
# Les tests sur NSIDE = 4 ont été effectués avec un bruit multiplié par 100*10000 pour avoir un SNR de 1 à environ l = 4
# Les tests sur NSIDE = 32 ont été effectués avec un bruit multiplié par 1000 pour avoir un SNR de 1 à environ l = 18
# noise_covar = noise_covar_one_pix[7]*1000
# noise_covar = noise_covar_one_pix[7]*100*10000
# Pour nside = 8 et L_cut = 10: noise_covar = noise_covar_one_pix[7]*50000
# noise_covar = noise_covar_one_pix[7]*1000000
#noise_covar = noise_covar_one_pix[7]*100*10000*20
#noise_covar_temp = 40**2
noise_covar_temp = 4**2
noise_covar_pol = 0.00044**2
#noise_covar_temp = 100**2
#noise_covar_pol = 0.44**2
var_noise_temp = np.ones(Npix) * noise_covar_temp
var_noise_pol = np.ones(Npix) * noise_covar_pol
inv_var_noise = np.ones(Npix) / noise_covar_temp

l_cut = 5
print("L_CUT")
bins = np.array([636, 638, 640, 642, 644, 646, 648, 650, 653, 656, 660,664, 669, 675,682, 690, 695, 705, 720, 730, 750, 770,
          800, 850, 1001])
bins = np.concatenate([np.arange(600, 636, 2), bins])
bins = np.concatenate([range(600), bins])
#bins = np.array(range(L_MAX_SCALARS+1+1))
#blocks = np.concatenate([np.arange(0, len(bins), 10), [len(bins)+1]])
blocks = np.concatenate([np.arange(0, 557, 20),np.arange(557, 600+1, 10), range(601, len(bins))])
blocks[0] = 2


#bins = np.array([0, 1, 2, 3, 4, 5, 7, 9])
#bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#bins = np.array([850, 851])

metropolis_blocks_gibbs_nc = blocks
#metropolis_blocks_gibbs_nc = np.array([2, 3, 4, 5, 6, 7, 8, 9])
metropolis_blocks_gibbs_asis = blocks
#metropolis_blocks_gibbs_pncp = [4, 5, 7]
metropolis_blocks_gibbs_pncp = blocks
metropolis_blocks_pncp = blocks



N_gibbs = 100
N_nc_gibbs = 10000
N_rescale = 1000
N_ASIS = 10000
N_PNCP = 300
N_direct = 100
N_default_gibbs = 10000
N_exchange = 100
N_hmc = 20000

mask_non_centered_int = np.concatenate([[1 if l >= l_cut else 0 for l in range(0, L_MAX_SCALARS + 1)],
                                        np.concatenate([np.concatenate(
                                            [(1, 1) if l >= l_cut else (0, 0) for l in range(m, L_MAX_SCALARS + 1)])
                                                        for m in range(1, L_MAX_SCALARS + 1)])])

mask_non_centered = mask_non_centered_int == 1
mask_centered = mask_non_centered_int == 0
print(mask_non_centered)

N_metropolis = 1

rescaling_map2alm = (Npix / (4 * np.pi))

print("L_cut")
print(l_cut)


def generate_var_cl(cls_):
    var_cl_full = np.concatenate([cls_,
                                  np.array(
                                      [cl for m in range(1, L_MAX_SCALARS + 1) for cl in cls_[m:] for _ in range(2)])])
    return var_cl_full


import matplotlib.pyplot as plt

# fwhm_arcmin = 180
# fwhm_radians = (np.pi/(180*60))*fwhm_arcmin
beam_fwhm = 0.35
fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm=fwhm_radians, lmax=L_MAX_SCALARS)
bl_map = generate_var_cl(bl_gauss)

# alpha_l = np.array([(2*l-1)/2 for l in range(2, L_MAX_SCALARS+1)])
# proposal_variances_nc = np.abs((((noise_covar*w/(bl_gauss[2:]**2))**2)/((alpha_l-1)**2*(alpha_l-2))))

# data = np.load(scratch_path +"/data/skymap/map.npy", allow_pickle=True)
# data = data.item()
# d_ = data["d_"]
w = 4 * np.pi / Npix

def compute_init_values(unbinned_vars):
    vals = []
    for i, l_start in enumerate(bins[:-1]):
        l_end = bins[i+1]
        length = l_end - l_start
        vals.append(np.mean(unbinned_vars[l_start:l_end])/length)

    return np.array(vals)


scale = [((l*(l+1))**2*2/(4*np.pi**2*(2*l+1))) for l in range(0, L_MAX_SCALARS+1)]
unbinned_variances = (w*noise_covar_temp/bl_gauss**2)**2*scale
binned_variances = compute_init_values(unbinned_variances)
binned_variances[600:-2] *= 4
binned_variances[-2:-1] *= 1
binned_variances[-1] *= 0.1


unbinned_variances_pol = (w*noise_covar_pol/bl_gauss**2)**2*scale
binned_variances_pol = compute_init_values(unbinned_variances_pol)
binned_variances_pol[600:-2] *= 4
binned_variances_pol[-2:-1] *= 1
binned_variances_pol[-1] *= 0.1

binned_variance_polarization = np.stack([unbinned_variances, unbinned_variances_pol, unbinned_variances_pol], axis = 1)
preliminary_run =True

if preliminary_run:
    proposal_variances_nc = binned_variances[2:]
    proposal_variances_nc_polarized = {}
    proposal_variances_nc_polarized["TT"] = np.ones(len(unbinned_variances)) * 0.05
    proposal_variances_nc_polarized["EE"] = np.ones(len(unbinned_variances)) * 0.05
    proposal_variances_nc_polarized["BB"] = np.ones(len(unbinned_variances)) * 0.05
    proposal_variances_nc_polarized["TE"] = np.ones(len(unbinned_variances)) * 0.05
    proposal_variances_asis = binned_variances[2:]
    proposal_variances_pncp = binned_variances[2:]
else:
    def get_proposal_variances_preliminary(path):
        list_files = os.listdir(path)
        chains = []
        times = []
        accept_rate = []

        for i, name in enumerate(list_files):
            if name not in [".ipynb_checkpoints", "Untitled.ipynb", "preliminary_runs"]:
                data = np.load(path + name, allow_pickle=True)
                data = data.item()
                chains.append(data["chain"])


        chains = np.array(chains)
        variances = np.var(chains, axis=(0, 1))

        return variances

    """
    path_nc = scratch_path +"/data/isotropic_runs/non_centered_gibbs/preliminary_runs/SNR_550/"
    path_asis = scratch_path +"/data/isotropic_runs/asis/preliminary_runs/SNR_550/"
    path_pncp = scratch_path + "/data/isotropic_runs/pncp/preliminary_runs/SNR_550/"
    proposal_variances_pncp = get_proposal_variances_preliminary(path_asis)
    proposal_variances_pncp[-13:-3] *= 3
    proposal_variances_pncp[600:-13] *= 7
    proposal_variances_pncp[-4:-2] *= 2
    proposal_variances_pncp[-2:-1] *= 1.5
    proposal_variances_pncp[-1:] *= 0.7
    proposal_variances_pncp = proposal_variances_pncp[2:]
    """

    proposal_variances_nc = 6*unbinned_variances[2:]
    proposal_variances_pncp = 10 * unbinned_variances[2:]
    proposal_variances_asis = 6*unbinned_variances[2:]