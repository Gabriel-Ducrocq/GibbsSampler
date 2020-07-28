import numpy as np
import healpy as hp
import qcinv
import config
import time



import sys
import qcinv

import imp, os, sys, getopt

import numpy  as np
import healpy as hp
import utils
import matplotlib.pyplot as plt

#mask=hp.ud_grade(hp.read_map('/global/cscratch1/sd/dbeck/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',4),512)
mask = np.ones(config.Npix)
np.mean(mask)

out_dir="/global/homes/d/dbeck/wienerfiltering/qcinvtest"
inp_dir="/global/homes/d/dbeck/wienerfiltering/"
tr_cg   = qcinv.cd_solve.tr_cg

lmax=config.L_MAX_SCALARS
nside=config.NSIDE
#chain_descr=[	  [  2,    ["split(dense,     32, diag_cl)"],    128,     32,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
#                  [  1,    ["split(stage(2), 128, diag_cl)"],    256,    128,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()],
#                  [  0,    ["split(stage(1), 256, diag_cl)"],    512,    256,        3,      0.0,    tr_cg,   qcinv.cd_solve.cache_mem()] ]
chain_descr  =  [ [  0,    ["diag_cl"],    lmax,   nside,  np.inf,  1.0e-6,   qcinv.cd_solve.tr_cg,    qcinv.cd_solve.cache_mem()] ]

theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale=config.COSMO_PARAMS_SIGMA_PRIOR)
cls_ = utils.generate_cls(theta_, False)
map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.beam_fwhm, new=True)
d = map_true
d += np.random.normal(scale=np.sqrt(config.var_noise_temp))

# restore simulation.
#cltt,clee,clbb,clte  = np.loadtxt(inp_dir + "wmap_pol_C_l.dat",unpack=1)
#beam=np.loadtxt(inp_dir + "wmap_pol_b_l.dat")
beam = config.bl_gauss
"""
class cl(object):
    pass
s_cls = cl
#s_cls.clee = clee[:lmax+1]
#s_cls.clbb = clbb[:lmax+1]
s_cls.cltt = cls_

#ninv  = hp.read_map(inp_dir + "wmap_band_iqumap_r9_9yr_V_v5.fits",[1,2,3],hdu=2)/(5e1*3.324)**2

#ninv[0,:]=1./(20./hp.nside2resol(nside,arcmin=True))**2
#ninv[1,:]=0#*mask
#ninv[2,:]=1./(20./hp.nside2resol(nside,arcmin=True))**2
ninv = 1/config.var_noise_temp

#nside = hp.npix2nside(len(ninv[0]))
nside = config.NSIDE

#dmap  = hp.read_map(inp_dir + "wmap_pol_cmb+noise_512.fits",[1,2])
dmap = d


s_cls.lmax=lmax
# construct the chain.
# note: just-in-time instantiation here is useful e.g.
#       if n_inv_filt and chain are defined in a parameter
#       file and don't necessarily need to be loaded.
n_inv_filt =qcinv.opfilt_tt.alm_filter_ninv(ninv, beam)#, marge_maps=[])
chain =qcinv.multigrid.multigrid_chain(qcinv.opfilt_tt, chain_descr, s_cls, n_inv_filt, debug_log_prefix=('log_') )

#soltn =  qcinv.opfilt_tt.eblm(np.zeros( (2,qcinv.util_alm.lmax2nlm(lmax)), dtype=np.complex ))

soltn = np.zeros(int(qcinv.util_alm.lmax2nlm(lmax)), dtype=np.complex )
chain.solve(soltn, dmap)
print(soltn)
#soltnm=hp.alm2map([soltn.elm*0,hp.almpix_map, noise, bl_map, fwhm_radians, lmax, Npix, mask_path, isotropic=True)xfl(soltn.elm,s_cls.clee*beam),hp.almxfl(soltn.blm,s_cls.clbb*beam)],512)

"""


from CenteredGibbs import CenteredConstrainedRealization

weiner_filtering = CenteredConstrainedRealization(d, config.var_noise_temp, config.bl_map, config.fwhm_radians
                                                  , config.L_MAX_SCALARS, config.Npix, None, isotropic=True)

weiner_filtering_diag = CenteredConstrainedRealization(d, config.var_noise_temp, config.bl_map, config.fwhm_radians
                                                  , config.L_MAX_SCALARS, config.Npix, None, isotropic=True)

weiner_filtering.mask_path = True
h_sol = []
h_sol_diag = []
start = time.time()
for i in range(1):
    if i % 1000 == 0:
        print(i)

    solution, _, _ = weiner_filtering.sample(cls_, utils.generate_var_cl(cls_))
    solution_diag, _, _, weiner_diag = weiner_filtering_diag.sample(cls_, utils.generate_var_cl(cls_))
    print("\n\n")
    h_sol.append(solution[50])
    h_sol_diag.append(solution_diag[50])

end = time.time()

print(end-start)


print("DIFFS")
solution_pix = hp.alm2map(utils.real_to_complex(solution), nside = config.NSIDE)
solution_pix_diag = hp.alm2map(utils.real_to_complex(weiner_diag), nside= config.NSIDE)
print(np.nanmax(np.abs((solution_pix - solution_pix_diag)/solution_pix_diag)))
print(np.nanargmax(np.abs((solution_pix - solution_pix_diag)/solution_pix_diag)))
print(np.abs((solution_pix - solution_pix_diag)/solution_pix_diag))

relative_errors = np.abs((solution_pix - solution_pix_diag)/solution_pix_diag)
hp.mollview(relative_errors)
plt.show()
plt.close()

plt.boxplot(relative_errors)
plt.title("Relative error distribution")
plt.show()

plt.boxplot(relative_errors, showfliers=False)
plt.title("Relative error distribution")
plt.show()

print(np.mean(h_sol))
print(np.mean(h_sol_diag))
print(np.var(h_sol))
print(np.var(h_sol_diag))
plt.hist(h_sol, density=True, alpha=0.5, label="PCG")
plt.hist(h_sol_diag, density=True, alpha=0.5, label="Diag")
plt.legend(loc="upper right")
plt.show()