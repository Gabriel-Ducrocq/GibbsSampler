import numpy as np
from CenteredGibbs import PolarizedCenteredClsSampler, PolarizedCenteredConstrainedRealization
import config
from main_polarization import generate_dataset
import healpy as hp
import utils
import matplotlib.pyplot as plt
from NonCenteredGibbs import PolarizedNonCenteredConstrainedRealization, PolarizationNonCenteredClsSampler
from scipy.stats import truncnorm

theta_, cls_, s_true, pix_map = generate_dataset(polarization=True, mask_path=config.mask_path)

cls_TT = np.zeros(len(cls_[0]))
alms = hp.synalm([cls_TT, cls_[1], cls_[2], cls_TT, cls_TT, cls_TT], lmax=config.L_MAX_SCALARS, new=True)
map = hp.alm2map(alms, pol=True, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
alms_back = hp.map2alm(map, lmax=config.L_MAX_SCALARS, pol=True)

#pix_map["Q"] = np.zeros(len(pix_map["Q"]))
#pix_map["U"] = np.zeros(len(pix_map["U"]))

lmax = config.L_MAX_SCALARS
nside = config.NSIDE
bins = config.bins
bl_map = config.bl_map
noise = config.noise_covar_pol
_, s_alm_EE, s_alm_BB, = hp.map2alm(s_true, lmax=lmax, pol=True)

cls_sampler_nc = PolarizationNonCenteredClsSampler(pix_map, lmax, nside, bins, bl_map, config.noise_covar_pol, noise,
                                                   config.metropolis_blocks_gibbs_nc, config.proposal_variances_nc_polarized)
cls_sampler = PolarizedCenteredClsSampler(pix_map, lmax, nside, bins, bl_map, noise)
map_sampler_nc = PolarizedNonCenteredConstrainedRealization(pix_map, config.noise_covar_temp*np.ones(config.Npix),
                                                            config.noise_covar_pol*np.ones(config.Npix),config.bl_map,
                                                            lmax, config.Npix, config.beam_fwhm)
map_sampler = PolarizedCenteredConstrainedRealization(pix_map, config.noise_covar_temp*np.ones(config.Npix),
                                                      config.noise_covar_pol*np.ones(config.Npix),config.bl_map,
                                                            lmax, config.Npix, config.beam_fwhm)


rescale = np.array([l*(l+1)/(2*np.pi) for l in range(lmax+1)])
#all_dls = {"EE":rescale*cls_[1], "BB":rescale*cls_[2]}
all_dls = {"EE":rescale*cls_[1], "BB":rescale*cls_[1]}

var_EE = utils.generate_var_cl(all_dls["EE"])
var_BB = utils.generate_var_cl(all_dls["BB"])
inv_var_EE = np.zeros(len(var_EE))
inv_var_BB = np.zeros(len(var_BB))
inv_var_EE[var_EE != 0] = 1/var_EE[var_EE != 0]
inv_var_BB[var_BB != 0] = 1/var_EE[var_BB != 0]
s_nonCentered = {"EE":np.sqrt(inv_var_EE)*utils.complex_to_real(s_alm_EE),
                 "BB": np.sqrt(inv_var_BB)*utils.complex_to_real(s_alm_BB)}


"""
###Testing generation of Dls
old_dls = {"EE":np.random.uniform(0, 1000, lmax+1), "BB":np.random.uniform(0, 1000, lmax+1)}
h_dls = {"EE":[], "BB":[]}
for i in range(100000):
    dls_new = cls_sampler_nc.propose_dl(old_dls)
    h_dls["EE"].append(dls_new["EE"])
    h_dls["BB"].append(dls_new["BB"])


h_dls["EE"] = np.array(h_dls["EE"])
h_dls["BB"] = np.array(h_dls["BB"])

for _, pol in enumerate(["EE", "BB"]):
    for l in range(2, config.L_MAX_SCALARS + 1):
        xs =np.arange(0, np.max(h_dls[pol][:, l]), 0.01)
        y = truncnorm.pdf(x = xs, a=-old_dls[pol][l]/np.sqrt(config.proposal_variances_nc_polarized[pol][l-2]),
                          b=np.inf, loc=old_dls[pol][l],scale=np.sqrt(config.proposal_variances_nc_polarized[pol][l-2]))
        plt.hist(h_dls[pol][:, l], density=True, alpha=0.5, label="MCMC", bins=100)
        plt.plot(xs, y)
        plt.title(pol + " with l=" + str(l))
        plt.show()

"""
all_dls = {"EE":np.ones(lmax+1)*0.1, "BB":np.ones(lmax+1)*0.1}
h_dls = {"EE":[], "BB":[]}
h_dls["EE"].append(all_dls["EE"].copy())
h_dls["BB"].append(all_dls["BB"].copy())

h_dls_centered = {"EE":[], "BB":[]}
h_dls_centered["EE"].append(all_dls["EE"].copy())
h_dls_centered["BB"].append(all_dls["BB"].copy())
acceptions = []
for i in range(100000):
    print(i)
    all_dls, accept = cls_sampler_nc.sample(s_nonCentered.copy(), all_dls.copy())
    #all_dls_centered = cls_sampler.sample({"EE":utils.complex_to_real(s_alm_EE), "BB": utils.complex_to_real(s_alm_BB)})
    acceptions.append(accept)
    h_dls["EE"].append(all_dls["EE"].copy())
    h_dls["BB"].append(all_dls["BB"].copy())
    #h_dls_centered["EE"].append(all_dls_centered["EE"].copy())
    #h_dls_centered["BB"].append(all_dls_centered["BB"].copy())


h_dls["EE"] = np.array(h_dls["EE"])
h_dls["BB"] = np.array(h_dls["BB"])
h_dls_centered["EE"] = np.array(h_dls_centered["EE"])
h_dls_centered["BB"] = np.array(h_dls_centered["BB"])
acceptions = np.array(acceptions)
print("Non centered acceptance rate:")
print(np.mean(acceptions, axis=0))
from scipy.stats import norm
x = np.arange(-4, 4, 0.001)
y = truncnorm.pdf(x, loc = 1, a=-1, b= np.inf, scale = 1)
for _, pol in enumerate(["EE", "BB"]):
    for l in range(2, config.L_MAX_SCALARS + 1):
        #y, xs, norm = utils.trace_likelihood_pol_binned(h_dls[pol], {"EE":utils.complex_to_real(s_alm_EE), "BB":utils.complex_to_real(s_alm_BB)}, l,
        #                                                maximum=np.max(h_dls[pol][:, l]), pol=pol)

        y, x, norm = utils.conditionnal_nc(s_nonCentered, l, h_dls[pol][:, l], pol, pix_map)
        plt.plot(h_dls[pol][:, l])
        plt.show()

        plt.hist(h_dls[pol][100:, l], density=True, alpha=0.5, label="Gibbs NC", bins=30)
        #plt.hist(h_dls_centered[pol][100:, l], density=True, alpha=0.5, label="Gibbs Centered", bins=30)
        print("Norm:", norm)
        #plt.plot(xs, y / norm)
        plt.plot(x, y/norm)
        plt.title(pol + " with l=" + str(l))
        plt.legend(loc="upper right")
        plt.show()



rescale = np.array([l*(l+1)/(2*np.pi) for l in range(lmax+1)])
all_dls = {"EE":rescale*cls_[1], "BB":rescale*cls_[2]}
var_cls_EE = utils.generate_var_cl(all_dls["EE"])
var_cls_BB = utils.generate_var_cl(all_dls["BB"])



h_nc = {"EE":[], "BB":[]}
h_cent = {"EE":[], "BB":[]}
for i in range(100000):
    nc, _ = map_sampler_nc.sample(all_dls)
    cent, _ = map_sampler.sample(all_dls)
    h_cent["EE"].append(cent["EE"])
    h_cent["BB"].append(cent["BB"])

    h_nc["EE"].append(np.sqrt(var_cls_EE)*nc["EE"])
    h_nc["BB"].append(np.sqrt(var_cls_BB)*nc["BB"])



h_nc["EE"] = np.array(h_nc["EE"])
h_nc["BB"] = np.array(h_nc["BB"])

h_cent["EE"] = np.array(h_cent["EE"])
h_cent["BB"] = np.array(h_cent["BB"])
for l in [2, 10, 15, 24]:
    for pol in ["EE", "BB"]:
        plt.hist(h_cent[pol][:, l], alpha = 0.5, density=True, label="Cenered", bins = 50)
        plt.hist(h_nc[pol][:, l], alpha=0.5, density=True, label="NonCentered", bins = 50)
        plt.legend(loc="upper right")
        plt.title(pol + " for l="+str(l))
        plt.show()


"""
d = {"EE":pix_map_E, "BB":pix_map_B}
h_dls = {"EE":[], "BB":[]}
for k in range(100000):
    if k % 1000:
        print(k)

    sampled_dls = cls_sampler.sample(d.copy())
    h_dls["EE"].append(sampled_dls["EE"])
    h_dls["BB"].append(sampled_dls["BB"])


h_dls["EE"] = np.array(h_dls["EE"])
h_dls["BB"] = np.array(h_dls["BB"])
for pol in ["EE", "BB"]:
    for l in range(2, lmax+1):
        y, xs, norm = utils.trace_likelihood_pol_binned_bis(h_dls, d.copy(), l, np.max(h_dls[pol][:, l]), pol)

        print(norm)
        plt.hist(h_dls[pol][:, l], density=True, alpha = 0.5, bins = 50)
        if norm < 1e-9:
            plt.plot(xs, y)
        else:
            plt.plot(xs, y/norm)

        plt.title(pol + " with l="+str(l))
        plt.show()
"""