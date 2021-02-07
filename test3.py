from CenteredGibbs import CenteredGibbs
import config
import numpy as np
import utils
import healpy as hp
import main
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from main_polarization import generate_dataset, generate_cls

theta_, cls_ = generate_cls()
map_true = hp.synfast(cls_, nside=config.NSIDE, lmax=config.L_MAX_SCALARS, fwhm=config.fwhm_radians, new=True)
d = map_true
s_T, s_E, s_B = hp.map2alm([d[0], d[1], d[2]], lmax=config.L_MAX_SCALARS, pol=True)

new_d_I, new_d_Q, new_U = hp.alm2map([s_T, s_E, s_B], lmax=config.L_MAX_SCALARS, nside=config.NSIDE)


print(np.abs((new_d_I - d[0])/d[0]))



