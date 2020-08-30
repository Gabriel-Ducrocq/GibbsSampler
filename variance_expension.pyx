import numpy as np
from healpy._healpy_sph_transform_lib import _alm2map
cimport scipy.special.cython_special
from scipy.special import erfinv
cimport numpy as np


cpdef double[:] generate_var_cl_cython(double[:] cls_):
    cdef int L_max, size_real, size_complex, idx, l, m, i
    cdef double pi
    pi = np.pi
    L_max = len(cls_) - 1
    size_real = (L_max + 1)**2
    size_complex = (L_max+1)*(L_max+2)/2
    cdef double[:] alms_shape = np.zeros(size_complex)
    cdef double[:] variance = np.zeros(size_real)
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


cpdef double[::1, :, :] generate_polarization_var_cl_cython(double[::1, :, :] cls_):
    cdef int L_max, size_real, size_complex, idx, l, m, i
    cdef double pi
    pi = np.pi
    L_max = len(cls_) - 1
    size_real = (L_max + 1)**2
    size_complex = (L_max+1)*(L_max+2)/2
    cdef double[::1, :, :] alms_shape = np.zeros((size_complex, 3, 3), order="F")
    cdef double[::1, :, :] variance = np.zeros((size_real, 3, 3), order="F")
    for l in range(L_max+1):
        for m in range(l+1):
            idx = m * (2 * L_max + 1 - m) // 2 + l
            if l == 0:
                alms_shape.base[idx, :, :] = cls_.base[l, :, :]
            else:
                alms_shape.base[idx, :, :] = cls_.base[idx, :, :]*2*pi/(l*(l+1))

    for i in range(L_max+1):
        variance.base[i, :, :] = alms_shape.base[i, :, :]


    for i in range(L_max+1, size_complex):
        variance.base[2*i - (L_max+1), :, :] = alms_shape.base[i, :, :]
        variance.base[2*i - (L_max+1) +1, :, :] = alms_shape.base[i, :, :]

    return variance



cpdef double[:] complex_to_real(double complex[:] alms):
    cdef int len_alms, Lm, i, size_result
    cdef double sqrtdeux
    sqrtdeux = np.sqrt(2)
    len_alms = len(alms)
    Lm = (-3 + np.sqrt(9+8*(len_alms-1)))/2

    cdef double[:] result = np.zeros((Lm + 1)**2)

    for i in range(len(alms)):
        if i <= Lm:
            result[i] = alms[i].real
        else:
            result[2*i - (Lm+1)] = alms[i].real*sqrtdeux
            result[2*i - (Lm+1) +1] = alms[i].imag*sqrtdeux

    return result


cpdef double complex[:] real_to_complex(double[:] alms):
    cdef int L_MAX_SCALARS, len_alms, len_result, i
    cdef double inv_sqrtdeux
    inv_sqrtdeux = 1/np.sqrt(2)
    len_alms = len(alms)
    L_MAX_SCALARS = np.sqrt(len_alms) - 1
    len_result = (L_MAX_SCALARS + 1)*(L_MAX_SCALARS + 2)/2
    cdef double complex[:] results = np.zeros(int((L_MAX_SCALARS + 1)*(L_MAX_SCALARS + 2)/2), dtype=complex)


    for i in range(len_result):
        if i <= L_MAX_SCALARS:
            results[i] = alms[i] + 0j
        else:
            results[i] = alms[2*i - (L_MAX_SCALARS + 1)]*inv_sqrtdeux + 1j * alms[2*i - (L_MAX_SCALARS + 1) + 1]*inv_sqrtdeux

    return results


cpdef double[:] remove_monopole_dipole_contributions(double[:] alms):
    cdef int len_alms, Lm
    len_alms = len(alms)
    Lm = np.sqrt(len_alms) - 1
    alms[0] = 0.0
    alms[1] = 0.0
    alms[Lm+1] = 0.0
    alms[Lm+2] = 0.0
    return alms


cpdef double[:] synthesis_hp(double[:] alms, int nside):
    cdef int Lm
    cdef double complex[:] alms_complex
    cdef double[:] s

    Lm = np.sqrt(len(alms)) - 1
    alms_complex = real_to_complex(alms)
    s = _alm2map(np.asarray(alms_complex), nside, lmax=Lm, mmax=-1)

    return s


cdef api double[:] synthesis(np.ndarray[np.float64_t] alms, int nside):
    cdef int Lm
    cdef double complex[:] alms_complex
    cdef double[:] s

    Lm = np.sqrt(len(alms)) - 1
    alms_complex = real_to_complex(alms)
    s = _alm2map(np.asarray(alms_complex), nside, lmax=Lm, mmax=-1)

    return s


cdef api double erfinv_wrap(double x):
    return erfinv(x)


cdef api double[:] test(double[:] inp):
    return inp