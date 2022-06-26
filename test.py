from classy import Class
import config
import matplotlib.pyplot as plt
import numpy as np
import camb
import time


cosmo = Class()

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "H0", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_CLASS = np.array([0.9665, 0.02242, 0.11933, 68.2, 3.047, 0.0561])
COSMO_PARAMS_CAMB = np.array([0.9665, 0.02242, 0.11933, 68.2, 3.047e-9, 0.0561])

L_MAX_SCALARS=2500
LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

def generate_cls_class(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': L_MAX_SCALARS,
              'lensing': LENSING,
              'format':'camb'}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    start = time.time()
    cosmo.compute()
    end = time.time()
    print("Computation time CLASS:")
    print(end-start)
    cls = cosmo.lensed_cl(L_MAX_SCALARS)
    #10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls["tt"] *= 1e12
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"]


def generate_cls_camb(theta):
    ns, omega_b, omega_cdm, H0, As, tau_reio = theta
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_cdm, tau=tau_reio)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(config.L_MAX_SCALARS)

    print("Computation time CAMB:")
    start = time.time()
    results = camb.get_results(pars)
    pow_spec = results.get_cmb_power_spectra(pars, CMB_unit='muK', )
    end = time.time()
    print(end-start)
    print(pow_spec["total"].shape)
    pow_spec = pow_spec["total"][: L_MAX_SCALARS+1, 0]
    return pow_spec


#cls_class = generate_cls_class(COSMO_PARAMS_CLASS)
cls_camb = generate_cls_camb(COSMO_PARAMS_CAMB)
scale = np.array([l*(l+1)/(2*np.pi) for l in range(L_MAX_SCALARS+1)])

#plt.plot(cls_class*scale, label="CLASS")
plt.plot(cls_camb, label="CAMB")
plt.legend(loc="upper right")
plt.title("C_ell")
plt.show()