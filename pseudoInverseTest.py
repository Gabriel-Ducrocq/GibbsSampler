import config
import numpy as np
import healpy as hp
from main_polarization import generate_cls
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import time

mask = hp.ud_grade(hp.read_map(config.mask_path, 0), config.NSIDE)
beta = 0.9
index = 130000
#delta = (-(2*beta**2 - 8) + np.sqrt((2*beta**2 -8)**2 - 16*beta**4))/(2*beta**2)
#print(np.sqrt(8*delta/((2+delta)**2)))
print("\n")
_, cls = generate_cls(polarization=True)
cls_tt, cls_ee, cls_bb, cls_te = cls

plt.plot(cls_tt, color="red")
plt.plot(cls_ee, color = "blue")
plt.plot(cls_te, color = "green")
plt.show()

inv_cls = np.zeros(len(cls_tt))
inv_cls[cls_tt!=0] = 1/cls_tt[cls_tt!=0]

w = 4*np.pi/config.Npix
#mask = np.random.uniform(0.98, 1.02, size = config.Npix)
noise_var = config.noise_covar_temp*np.ones(config.Npix)
inv_noise_var = 1/noise_var
#inv_noise_var = mask*inv_noise_var

pseudo_inv_inv_noise_var = np.zeros(config.Npix)
pseudo_inv_inv_noise_var[inv_noise_var!=0] = 1/inv_noise_var[inv_noise_var!=0]

d = hp.synfast(cls_tt, lmax=config.L_MAX_SCALARS, nside=config.NSIDE)


def compute(alms, noise_term):
    map_pix = hp.alm2map(alms, lmax=config.L_MAX_SCALARS, nside=config.NSIDE)
    map_pix *= noise_term
    return hp.map2alm(map_pix, lmax=config.L_MAX_SCALARS, iter=3) * config.Npix / (4 * np.pi)

test = hp.map2alm(inv_noise_var*d, lmax=config.L_MAX_SCALARS, iter = 1)*(1/w)
res = compute(test, pseudo_inv_inv_noise_var)*w**2

print("Mean prior:", res)

ones = np.ones(len(res), dtype=complex)
cls_values = hp.almxfl(ones, cls_tt*config.bl_gauss**2)

plt.plot(res.real, color="blue", alpha=0.5)
plt.plot(cls_values, color="red", alpha=0.5)
plt.title("comparison")
plt.show()

plt.hist(res.real, bins = 200)
plt.title("test")
plt.show()

plt.plot(cls_tt*config.bl_gauss**2)
plt.title("Cls")
plt.show()

alms_T, alm_E, alms_B = hp.synalm([cls_tt, cls_ee, cls_bb, cls_te], lmax=config.L_MAX_SCALARS, new=True)

average = hp.map2alm(inv_noise_var*d, lmax=config.L_MAX_SCALARS)*(1/w)
average = hp.almxfl(compute(average, pseudo_inv_inv_noise_var), 1/config.bl_gauss, inplace=False)*w**2
average = np.zeros(len(average), dtype=complex)

all_cls = hp.almxfl(np.ones(len(average)), cls_tt)
plt.plot(average.real)
plt.title("Average")
plt.plot(all_cls)
plt.show()



def sample_from_likelihood(pseudo_inv_noise):
    sampled_alms = hp.map2alm(np.sqrt(pseudo_inv_noise)*np.random.normal(size=config.Npix), lmax=config.L_MAX_SCALARS, iter=1)
    hp.almxfl(sampled_alms, 1/config.bl_gauss, inplace = True)
    return sampled_alms

def compute_log_prior(inv_cls, alms, average):
    alms_multiplied = hp.almxfl(alms+average, inv_cls, inplace=False)
    return -(1/2)*np.sum(np.conjugate(alms+average)*alms_multiplied)


def run_cn(cls, inv_cls, pseudo_inv_noise, average):
    s = np.zeros(len(average), dtype=complex)
    print("S dimension:", len(s))
    accept = 0
    N = 1000
    h = []
    h.append(s[index])
    start = time.time()
    for i in range(N):
        print(i)
        Z = sample_from_likelihood(pseudo_inv_noise)
        proposed_s = np.sqrt(1-beta**2)*s + beta*Z
        log_ratio = compute_log_prior(inv_cls, proposed_s, average) - compute_log_prior(inv_cls, s, average)
        if np.log(np.random.uniform()) < log_ratio:
            s = proposed_s
            accept += 1

        h.append(s[index])

    end = time.time()
    print("Time needed:", end - start)
    print("Acceptance rate:", accept/N)
    return np.array(h)


h = run_cn(cls_tt, inv_cls, pseudo_inv_inv_noise_var, average)
plt.plot(h.real, color="red")
plt.show()

plt.plot(h.imag, color="blue")
plt.show()

plot_acf(h.real)
plt.show()

plot_acf(h.imag)
plt.show()


"""
def compute(alms, noise_term):
    map_pix = hp.alm2map(alms, lmax=config.L_MAX_SCALARS, nside=config.NSIDE)
    #map_pix *= noise_term
    return hp.map2alm(map_pix, lmax=config.L_MAX_SCALARS, iter=3) * config.Npix / (4 * np.pi)


w = 4*np.pi/config.Npix
_, cls_tt = generate_cls(polarization=False)

cls_tt[:2] = cls_tt[2:4]
print(cls_tt)

map_alms = hp.synalm(cls_tt, lmax=config.L_MAX_SCALARS)

alms_back1 = compute(map_alms, inv_noise_var)

alm_first = compute(map_alms, inv_noise_var)
alm_second = compute(alm_first, pseudo_inv_inv_noise_var)
alms_back2 = compute(alm_second, inv_noise_var)
alms_back2 *= w**2



print(alms_back1)
print(alms_back2)

print(np.abs((alms_back1.real-alms_back2.real)/alms_back2.real))
print(np.min(alms_back1.real/alms_back2.real))
plt.boxplot(np.abs((alms_back1.real-alms_back2.real)/alms_back2.real), showfliers=True)
plt.show()
plt.plot(alms_back1.real/alms_back2.real)
plt.show()

print(map_alms)

"""

