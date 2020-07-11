import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import inv, spsolve
import time
#from linear_algebra import product_cls_inverse, compute_cholesky, compute_inverse_matrices
import config
import utils
import healpy as hp
from CenteredGibbs import PolarizedCenteredConstrainedRealization, PolarizedCenteredClsSampler

theta_ = config.COSMO_PARAMS_MEAN_PRIOR + np.random.normal(scale=config.COSMO_PARAMS_SIGMA_PRIOR)
cls_tt, cls_ee, cls_bb, cls_te = utils.generate_cls(theta_, pol = True)
alms = hp.synalm([cls_tt, cls_ee, cls_ee, cls_te, np.zeros(len(cls_tt)), np.zeros(len(cls_tt))],
                    lmax=config.L_MAX_SCALARS, new=True)

alms_TT = utils.complex_to_real(alms[0])
alms_EE = utils.complex_to_real(alms[1])
alms_BB = utils.complex_to_real(alms[2])
list_alms = np.stack([alms_TT, alms_EE, alms_BB], axis = 1)
#pol_sampler = PolarizedCenteredClsSampler(pix_map, config.L_MAX_SCALARS, config.bins, config.bl_map, config.noise_covar_temp)
#pol_sampler.sample(list_alms)

print("CLS TT")
print(cls_tt[1000:])
print("\n")
r = hp.alm2cl(alms[0, :], lmax=config.L_MAX_SCALARS)
print(r[1000:])


print(alms.shape)



"""
pol_sampler = PolarizedCenteredConstrainedRealization(pix_map, 40, 0.44, config.bl_map, config.L_MAX_SCALARS,
                                                      config.Npix, config.beam_fwhm, isotropic=True)

print("BB")
print(cls_bb)
cls_bb = cls_ee
list_matrices = []
for i in range(config.L_MAX_SCALARS+1):
    m = np.zeros((3, 3))
    m[0, 0] = cls_tt[i]
    m[1, 0] = m[0, 1] = cls_te[i]
    m[1, 1] = cls_ee[i]
    m[2, 2] = cls_bb[i]
    list_matrices.append(m)

input_cls = np.stack(list_matrices, axis = 0)
print(input_cls[2, :, :])
print("Compilation time")
solution, _, _ = pol_sampler.sample2(input_cls)
print("No compilation time")
solution, _, _ = pol_sampler.sample2(input_cls)

print(solution)


"""


"""
start = time.time()
list_blocks =[]
list_b = []
N = 2*512
for i in range(N):
    m = np.random.normal(size=(3,3))
    m = np.dot(m.T, m)
    list_blocks.append(m)
    bb = np.random.normal(size=3)
    list_b.append(bb)

end = time.time()
print(end-start)

b = np.stack(list_b, axis = 0)
b = np.ascontiguousarray(b)

#b_python = b.reshape(3*1000000, -1)

#start = time.time()
#block_diag_mat = block_diag(list_blocks, format="csc")
#result_python = spsolve(block_diag_mat, b_python)
#end = time.time()
#print("Time python")
#print(end-start)

mat = np.stack(list_blocks, axis= 0)
mat = np.ascontiguousarray(mat)

start_cython = time.time()
result_cython = product_cls_inverse(mat, b, N)
end_cython = time.time()
print("Time Cython:")
print(end_cython-start_cython)


start = time.time()
L, info = compute_cholesky(mat, N)
end = time.time()
print("Time cython:", end - start)

print("\n")
print(np.linalg.inv(list_blocks[10]))
print("\n")
print(np.array(L)[10, :, :])

mat_f = np.asfortranarray(np.stack(list_blocks, axis= 0))
start = time.time()
var_cls = generate_polarization_var_cl_cython(mat_f)
end = time.time()
print("Time variance expension")
print(end- start)

#r = np.abs(np.array(L)[0, :, :] - np.linalg.inv(list_blocks[0]))/np.linalg.inv(list_blocks[0])
#print(r)


size_alm = int((config.L_MAX_SCALARS+1)*(config.L_MAX_SCALARS+2)/2)
#alms_pol = [np.random.normal(size = size_alm) + 1j*np.random.normal(size = size_alm),
#            np.random.normal(size = size_alm) + 1j*np.random.normal(size = size_alm),
#            np.random.normal(size = size_alm) + 1j*np.random.normal(size = size_alm)]


#map = hp.alm2map(alms_pol, lmax=config.L_MAX_SCALARS, nside=config.NSIDE)
#alm_back = hp.map2alm(map, lmax=config.L_MAX_SCALARS)


#print(np.max(np.abs(alms_pol[0].imag - alm_back[0].imag)/alm_back[0].imag))
#print("\n")
#print(np.abs(alms_pol[1].imag - alm_back[1].imag)/alm_back[1].imag)
#print("\n")
#print(np.abs(alms_pol[2].imag - alm_back[2].imag)/alm_back[2].imag)

#pix_pol = [np.random.normal(size=config.Npix), np.random.normal(size=config.Npix), np.random.normal(size=config.Npix)]
#pix_pol = np.random.normal(size=config.Npix)
#alms_pol = hp.map2alm(pix_pol, lmax=config.L_MAX_SCALARS)*4*np.pi/config.Npix
#pix_back = hp.alm2map(alms_pol, lmax=config.L_MAX_SCALARS, nside=config.NSIDE)

#print(np.abs((pix_back[0] - pix_pol[0])/pix_pol[0]))
#print("\n")
#print(np.abs((pix_back[1] - pix_pol[1])/pix_pol[1]))
#print("\n")
#print(np.abs((pix_back[2] - pix_pol[2])/pix_pol[2]))
#print(np.abs((pix_back - pix_pol)/pix_pol))
"""
"""
offset = np.array([float(i) for i in range(1000)])
list_mat = []
list_inv = []
for i in range(1000000):
    mat = np.random.normal(size=(3,3))
    mat = np.dot(mat.T, mat)
    mat[0, 2] = mat[1, 2] = mat[2, 0] = mat[2, 1] = 0
    list_mat.append(mat)

start = time.time()
list_chol= []
for i in range(1000000):
    mat = list_mat[i]
    np.dot(mat, np.ones(3))
    #inv_mat = np.linalg.inv(mat) + np.eye(3)*offset[i]
    #inv_inv_mat = np.linalg.inv(inv_mat)
    #chol = np.linalg.cholesky(inv_inv_mat)
    #list_inv.append(inv_inv_mat)
    #list_chol.append(chol)

end = time.time()
print("PYTHON")
print(end-start)

mat = np.stack(list_mat, axis = 0)
mat = np.ascontiguousarray(mat)
start = time.time()
r, e = compute_inverse_matrices(mat, 1000, offset)
end = time.time()
print("CYTHON")
print(end-start)


print(np.asarray(r)[0, :, :])
print(list_inv[0])
print("\n")
for i in range(1):
    print(np.nanmax(np.abs(np.asarray(r)[i, :, :] - list_inv[i])/list_inv[i]))
    print("\n")



alms = hp.map2alm(np.random.normal(size=config.Npix), lmax=config.L_MAX_SCALARS)
print(alms)
a = np.ones(len(alms)) + 1j*np.ones(len(alms))
r1, r2, r3 = hp.sphtfunc.smoothalm([a, a, a], fwhm=config.beam_fwhm)

print(r1-r3)
"""
