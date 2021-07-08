import numpy as np
import config
from CenteredGibbs import PolarizedCenteredConstrainedRealization
import main_polarization
import matplotlib.pyplot as plt
import healpy as hp
import utils
from main_polarization import generate_dataset, generate_cls
from CenteredGibbs import CenteredGibbs
import copy

noise_temp = np.ones(config.Npix) * config.noise_covar_temp
noise_pol = np.ones(config.Npix) * config.noise_covar_pol

theta_, cls_ = generate_cls()
cls_ = np.array([cls for cls in cls_])
cls_[0] = np.zeros(len(cls_[0]))
cls_[3] = np.zeros(len(cls_[0]))

snr = cls_[1] * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
plt.plot(snr)
plt.axhline(y=1)
plt.title("EE")
plt.show()

snr = cls_[2] * (config.bl_gauss ** 2) / (config.noise_covar_pol * 4 * np.pi / config.Npix)
plt.plot(snr)
plt.axhline(y=1)
plt.title("BB")
plt.show()

s_true, pix_map = generate_dataset(cls_, polarization=True, mask_path=config.mask_path)
centered_gibbs = CenteredGibbs(pix_map, noise_temp, noise_pol, config.beam_fwhm, config.NSIDE, config.L_MAX_SCALARS,
                               config.Npix,
                               mask_path=config.mask_path, polarization=True, bins=config.bins, n_iter=1000,
                               rj_step=False)

_, cls_EE, cls_BB, _ = utils.generate_cls(config.COSMO_PARAMS_PLANCK, pol=True)
scale = np.array([l * (l + 1) / (2 * np.pi) for l in range(config.L_MAX_SCALARS + 1)])
dls_EE = scale * cls_EE
dls_BB = scale * cls_BB
all_dls = {"EE": dls_EE, "BB": dls_BB}

init_s = {"EE": np.zeros((config.L_MAX_SCALARS+1)**2), "BB": np.zeros((config.L_MAX_SCALARS+1)**2)}


var_cls_E = utils.generate_var_cl(all_dls["EE"])
var_cls_B = utils.generate_var_cl(all_dls["BB"])

inv_var_cls_E = np.zeros(len(var_cls_E))
inv_var_cls_E[var_cls_E != 0] = 1 / var_cls_E[var_cls_E != 0]

inv_var_cls_B = np.zeros(len(var_cls_B))
inv_var_cls_B[var_cls_B != 0] = 1 / var_cls_B[var_cls_B != 0]

sigma_E = 1 / ((config.Npix / (config.noise_covar_pol * 4 * np.pi)) * config.bl_map ** 2 + inv_var_cls_E)
sigma_B = 1 / ((config.Npix / (config.noise_covar_pol * 4 * np.pi)) * config.bl_map ** 2 + inv_var_cls_B)


_, pix_map_E, pix_map_B = hp.map2alm([np.zeros(len(pix_map["Q"])), pix_map["Q"], pix_map["U"]],
                                     lmax=config.L_MAX_SCALARS, pol=True, iter = 0)

pix_map_E = utils.complex_to_real(pix_map_E)
pix_map_B = utils.complex_to_real(pix_map_B)

r_E = (config.Npix * (1/config.noise_covar_pol) / (4 * np.pi)) * pix_map_E
r_B = (config.Npix * (1/config.noise_covar_pol) / (4 * np.pi)) * pix_map_B

r_E = config.bl_map * r_E
r_B = config.bl_map * r_B

mean_E = sigma_E * r_E
mean_B = sigma_B * r_B

init_s = {"EE":mean_E, "BB":mean_B}


s = copy.deepcopy(init_s)
s_over = copy.deepcopy(init_s)
acceptance = 0
N_iter = 10
all_s_E = []
all_s_B = []
all_s_E_over = []
all_s_B_over = []
import time
start = time.time()
for i in range(N_iter):
    print(i)
    s, accept = centered_gibbs.constrained_sampler.sample_gibbs_change_variable(all_dls,s)
    s_over, _ = centered_gibbs.constrained_sampler.overrelaxation(all_dls, s_over)
    all_s_E.append(s["EE"][100])
    all_s_B.append(s["BB"][100])
    all_s_E_over.append(s_over["EE"][100])
    all_s_B_over.append(s_over["BB"][100])
    acceptance += accept

end = time.time()
print("Duration:", end - start)
print(acceptance/N_iter)


interest = 10
all_s_E = np.array(all_s_E)
all_s_B = np.array(all_s_B)
all_s_E_over = np.array(all_s_E_over)
all_s_B_over = np.array(all_s_B_over)

results = {"all_s_E":all_s_E, "all_s_B":all_s_B, "all_s_E_exact": all_s_E_over, "all_s_B_exact":all_s_B_over}
np.save("results.npy", results, allow_pickle=True)

plt.hist(all_s_E, alpha = 0.5, label="Aux grad E", density = True, bins=30)
plt.hist(all_s_E_over, alpha = 0.5, label="True", density=True, bins=30)
plt.legend(loc="upper right")
plt.show()


plt.plot(all_s_E, label="aux var")
plt.plot(all_s_E_over, label="over")
plt.legend(loc="upper right")
plt.show()



