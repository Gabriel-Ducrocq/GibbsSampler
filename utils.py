import numpy as np
import config
import healpy as hp
from classy import Class
#from variance_expension import generate_var_cl_cython

cosmo = Class()

def generate_theta():
    """

    :return: array of float, random cosmological parameters, sampled from the prior distribution.
    """
    return np.random.normal(config.COSMO_PARAMS_MEAN, config.COSMO_PARAMS_SIGMA)


def generate_cls(theta, pol = False):
    """
    generates the power spectrum corresponding the input cosmological parameters.

    :param theta: array of float, 6 cosmological parameters.
    :param pol: boolean, whether to compute polarization power spectra.
    :return: arrays of float, size L_max +1, of the power spectra.
    """
    params = {'output': config.OUTPUT_CLASS,
              "modes":"s,t",
              "r":0.001,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': config.LENSING}
    d = {name:val for name, val in zip(config.COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls_tt = cls["tt"]*2.7255e6**2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"]*2.7255e6**2
        cls_bb = cls["bb"]*2.7255e6**2
        cls_te = cls["te"]*2.7255e6**2
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt, cls_ee, cls_bb, cls_te

def real_to_complex(alms):
    """
    This converts the alms expressed in real convention to alms expressed in imaginary alms.

    :param alms: array of floats, size ((L_max + 1)**2,) of alms coefficient expressed in real convention. The first
            L_max + 1 terms are the m = 0 real coefficients. The following terms are, in m major, the couples (sqrt(2)Realpart(alm), sqrt(2)Imagpart(alm))
    :return: array of floats, size (L_max +1)*(L_max+2)/2, of alms coefficients in complex convention, m major.
    """
    m_0 = alms[:config.L_MAX_SCALARS+1] + np.zeros(config.L_MAX_SCALARS+1)*1j
    m_pos = alms[config.L_MAX_SCALARS+1:]
    m_pos = (m_pos[::2] + 1j*m_pos[1::2])/np.sqrt(2)
    return np.concatenate([m_0, m_pos])


def complex_to_real(alms):
    """
    convert a set of alms from complex convention to real convention. See the function real_to_complex for the conventions.

    :param alms: array of floats, size (L_max +1)*(L_max+2)/2, of alms coefficients in complex convention, m major.
    :return: array of floats, size ((L_max + 1)**2,) of alms coefficient expressed in real convention. The first
            L_max + 1 terms are the m = 0 real coefficients. The following terms are, in m major, the couples (sqrt(2)Realpart(alm), sqrt(2)Imagpart(alm))
    """
    m_0 = alms[:config.L_MAX_SCALARS+1].real
    m_pos = alms[config.L_MAX_SCALARS + 1:]
    m_pos_real = m_pos.real*np.sqrt(2)
    m_pos_img = m_pos.imag*np.sqrt(2)
    m_pos = np.array([a for parts in zip(m_pos_real, m_pos_img) for a in parts])
    return np.concatenate([m_0, m_pos])


def adjoint_synthesis_hp(map, bl_map=None):
    """
    computes an adjoint synthesis, i.e A^{T} instead of an harmonic analysis expressed as (4*\pi/Npix) * A^{T}

    :param map: array of floats, of size Npix. A sky map expressed in pixel domain.
    :param bl_map: array of floats, size (L_max+1)**2. The diagonal of the diagonal beam matrix B, see paper.
    :return: either one array of 3 arrays, corresponding to the alms coefficient of the map, expressed in real convention.
    """
    if len(map) == 3:
        #If 3 maps, it means we are dealing with polarization.
        alms = hp.map2alm(map, lmax=config.L_MAX_SCALARS, iter=3, pol=True) # Do a spherical harmonic analysis.
        alms_T, alms_E, alms_B = alms
        alms_T = complex_to_real(alms_T)*config.rescaling_map2alm #Rescale by Npix/(4*\pi) to get an adjoint synthesis.
        alms_E = complex_to_real(alms_E)*config.rescaling_map2alm # Same
        alms_B = complex_to_real(alms_B)*config.rescaling_map2alm #Same
        if bl_map is not None:
            #If a beam is provided, multiply by the diagonal beam matrix.
            alms_T *= bl_map
            alms_E *= bl_map
            alms_B *= bl_map

        return alms_T, alms_E, alms_B

    else:
        #Otherwise it is temperature only.
        alms = hp.map2alm(map, lmax=config.L_MAX_SCALARS, iter=3) # Do a spherical harmonic analysis
        alms = complex_to_real(alms) # Turn to real convention
        alms *= config.rescaling_map2alm # Rescale by Npix/(4*\pi) to get an adjoint transform.
        if bl_map is not None:
            #If a beam is provided, multiply by the diagonal beam matrix.
            alms *= bl_map

        return alms


def generate_var_cl_cython(cls_):
    pi = np.pi
    L_max = len(cls_) - 1
    size_real = (L_max + 1)**2
    size_complex = int((L_max+1)*(L_max+2)/2)
    alms_shape = np.zeros(size_complex)
    variance = np.zeros(size_real)
    for l in range(L_max+1):
        for m in range(l+1):
            idx = m * (2 * L_max + 1 - m) // 2 + l
            if l == 0:
                alms_shape[idx] = cls_[l]
            else:
                alms_shape[idx] = cls_[l]*2*pi/(l*(l+1))

    for i in range(L_max+1):
        variance[i] = alms_shape[i]


    for i in range(L_max+1, size_complex):
        variance[2*i - (L_max+1)] = alms_shape[i]
        variance[2*i - (L_max+1) +1] = alms_shape[i]

    return variance

def generate_var_cl(cls_):
    """
    compute the diagonal of the C matrix, see paper.

    :param cls_: array of floats, size L_max + 1. Power spectrum
    :return: Diagonal of the diagonal C matrix.
    """
    var_cl_full = generate_var_cl_cython(cls_) # sending to cython code
    return np.asarray(var_cl_full)


def unfold_bins(binned_cls_, bins):
    """
    turn a binned power spectrum to an unbinned power spectrum, e.g if C_\ell is the value of the bin \ell made of 10 multipoles
    then we repeat C_\ell ten times.

    :param binned_cls_: array of floats, size number of bins. Binned power spectrum.
    :param bins: array of integers, size number of bins + 1. Each value is the start and end of a bin, e.g
                [1, 4, 10, 20] means we have 3 bins, comprising respectively the multipoles 1, 2, 3 and 4, 5, 6, 7, 8, 9, 10 and
                then all the \ell from 10 to 20.
    :return: array of floats, size L_max + 1. unbinned power spectrum.
    """
    n_per_bin = bins[1:] - bins[:-1]
    return np.repeat(binned_cls_, n_per_bin)



