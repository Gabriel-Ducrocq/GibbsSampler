import ctypes
from numba.extending import get_cython_function_address
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import truncnorm, norm
from variance_expension import synthesis_hp

addr = get_cython_function_address("variance_expension", "erfinv_wrap")
addr2 = get_cython_function_address("variance_expension", "synthesis")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
functype2 = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.Array, ctypes.c_int)
erfinv = functype(addr)
alm2map = functype2(addr2)

""""
@njit
def erfinv_wrap(u):
    return erfinv(u)

@njit(parallel=True)
def sample(n = 100):
    samples = np.zeros(n)
    u = np.random.uniform(0, 1, size=n)
    for i in prange(n):
        samples[i] = erfinv_wrap(2*u[i] - 1)*np.sqrt(2)

    return samples
"""
@njit
def alm2map_python(alms, nside):
    print(nside)
    return alm2map(alms, nside)

@njit
def synthesis(alms, nside):
    return synthesis_hp(alms, nside)


"""
test()
print("Done")

start = time.time()
samples = sample()
end = time.time()
print("First")
print(end - start)
start = time.time()
samples = sample()
end = time.time()
print("Second")
print(end - start)

samples2 = np.random.normal(size= 100000)
plt.hist(samples, density = True, bins = 50, alpha = 0.5)
plt.hist(samples2, density = True, bins = 50, alpha = 0.5)
plt.show()
"""

from numba import jit
from cffi import FFI

ffi = FFI()
ffi.cdef('double sin(double x);')

# loads the entire C namespace
C = ffi.dlopen(None)
c_sin = C._alm2map

@jit(nopython=True)
def cffi_sin_example(x):
    return c_sin(x)