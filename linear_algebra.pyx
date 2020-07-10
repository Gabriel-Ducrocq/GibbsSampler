import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport *
from scipy.linalg.cython_lapack cimport *
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
cimport cython
from variance_expension import generate_polarization_var_cl_cython
from libc.math cimport sqrt, pow

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def product_cls_inverse(double[:,:,:] sigmas_symm, double[:,::1] b, int l):

    cdef:
        int n = sigmas_symm.shape[1]
        int nrhs = 1
        int lda = sigmas_symm.shape[1]
        int ldb = sigmas_symm.shape[1]
        int info = 0
        int inc = 1
        double[::1,:] result = np.zeros((l,3), order="F")
        int[::1] pivot = np.zeros(sigmas_symm.shape[1], dtype = np.intc, order = "F")
        double[::1] current = np.zeros(sigmas_symm.shape[1], order = "F")
        double[::1, :] sigm_current = np.zeros((sigmas_symm.shape[1], sigmas_symm.shape[1]), order = "F")

    for i in range(l):
        current = b[i].copy_fortran()
        sigm_current = sigmas_symm[i,:,:].copy_fortran()
        dgesv(&n, &nrhs, &sigm_current[0, 0], &lda, &pivot[0], &current[0], &ldb, &info)
        result[i,:] = current.copy_fortran()

    return result




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef compute_inverse_matrices(double[:, :, :] sigmas_symm, int l, double[::1] offset):

    cdef:
        double[::1,:, :] intermediate_result = np.zeros((l, sigmas_symm.shape[1], sigmas_symm.shape[1]), order="F", dtype=float)
        double[::1,:, :] result = np.zeros((l, sigmas_symm.shape[1], sigmas_symm.shape[1]), order="F", dtype=float)
        double[::1, :] sigm_current = np.zeros((sigmas_symm.shape[1], sigmas_symm.shape[1]), order = "F", dtype=float)
        double[::1, :, :] cholesky = np.zeros((l, sigmas_symm.shape[1], sigmas_symm.shape[1]), order="F", dtype=float)
        double[:, :] cholesky_temp = np.zeros((sigmas_symm.shape[1], sigmas_symm.shape[1]), order="F", dtype=float)
        double det = 1
        char* type_output = 'L'
        int n = sigmas_symm.shape[1]
        int lda = sigmas_symm.shape[1]
        int info = 0

    for i in range(2, l):
         sigm_current = sigmas_symm[i,:,:].copy_fortran()
         det = sigm_current[0, 0]*sigm_current[1, 1] - sigm_current[1, 0]*sigm_current[0, 1]
         intermediate_result.base[i, 0, 0] = sigm_current[1, 1]/det + offset[i]
         intermediate_result.base[i, 0, 1] = -sigm_current[0, 1]/det
         intermediate_result.base[i, 1, 0] = -sigm_current[1, 0]/det
         intermediate_result.base[i, 1, 1] = sigm_current[0, 0]/det + offset[i]
         intermediate_result.base[i, 2, 2] = 1.0/sigm_current[2, 2] + offset[i]


         det = intermediate_result.base[i, 0, 0]*intermediate_result.base[i, 1, 1] - intermediate_result.base[i, 1, 0]*intermediate_result.base[i,0, 1]
         result.base[i, 0, 0] = intermediate_result.base[i, 1, 1]/det
         result.base[i, 0, 1] = -intermediate_result.base[i, 0, 1]/det
         result.base[i, 1, 0] = -intermediate_result.base[i, 1, 0]/det
         result.base[i, 1, 1] = intermediate_result.base[i, 0, 0]/det
         result.base[i, 2, 2] = 1.0/intermediate_result.base[i, 2, 2]

         cholesky.base[i, 0, 0] = sqrt(result.base[i, 0, 0])
         cholesky.base[i, 1, 0] = result.base[i, 0, 1]/sqrt(result.base[i, 0, 0])
         cholesky.base[i, 1, 1] = sqrt(result.base[i, 1, 1] - (pow(result.base[i, 0, 1],2)/result.base[i, 0, 0]))
         cholesky.base[i, 2, 2] = sqrt(result.base[i, 2, 2])

    return result, cholesky



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
cpdef compute_matrix_product(double[:, :, :] dls_, double[:, :] solutions):
    cdef int L_max, size_real, size_complex, idx, l, m, i
    cdef double pi
    pi = np.pi
    L_max = len(dls_) - 1
    size_real = (L_max + 1)**2
    size_complex = (L_max+1)*(L_max+2)/2
    cdef double[::1, :, :] alms_shape = np.zeros((size_complex, 3, 3), order="F")
    cdef double[::1, :, :] variance = np.zeros((size_real, 3, 3), order="F")
    cdef double[::1, :] result = np.zeros((size_real, 3), order="F")
    for l in range(L_max+1):
        for m in range(l+1):
            idx = m * (2 * L_max + 1 - m) // 2 + l
            if l == 0:
                alms_shape.base[idx, :, :] = dls_.base[l, :, :]
            else:
                alms_shape.base[idx, :, :] = dls_.base[l, :, :]*2*pi/(l*(l+1))

    for i in range(L_max+1):
        result.base[i, 0] = alms_shape.base[i, 0, 0]*solutions[i, 0] + alms_shape.base[i, 0, 1]*solutions[i, 1]
        result.base[i, 1] = alms_shape.base[i, 1, 0]*solutions[i, 0] + alms_shape.base[i, 0, 1]*solutions[i, 1]
        result.base[i, 2] = alms_shape.base[i, 2, 2]*solutions[i, 2]

    for i in range(L_max+1, size_complex):
        result.base[2*i - (L_max+1), 0] = alms_shape.base[i, 0, 0]*solutions[2*i - (L_max+1), 0] + alms_shape.base[i, 0, 1]*solutions[2*i - (L_max+1), 1]
        result.base[2*i - (L_max+1), 1] = alms_shape.base[i, 1, 0]*solutions[2*i - (L_max+1), 0] + alms_shape.base[i, 0, 1]*solutions[2*i - (L_max+1), 1]
        result.base[2*i - (L_max+1), 2] = alms_shape.base[i, 2, 2]*solutions[2*i - (L_max+1), 2]

        result.base[2*i - (L_max+1) + 1, 0] = alms_shape.base[i, 0, 0]*solutions[2*i - (L_max+1) + 1, 0] + alms_shape.base[i, 0, 1]*solutions[2*i - (L_max+1) + 1, 1]
        result.base[2*i - (L_max+1) + 1, 1] = alms_shape.base[i, 1, 0]*solutions[2*i - (L_max+1) + 1, 0] + alms_shape.base[i, 0, 1]*solutions[2*i - (L_max+1) + 1, 1]
        result.base[2*i - (L_max+1) + 1, 2] = alms_shape.base[i, 2, 2]*solutions[2*i - (L_max+1) + 1, 2]

    return result






@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def compute_cholesky(double[::,:,:] sigmas_symm, int l):

    cdef:
        char* type_output = 'L'
        int n = sigmas_symm.shape[1]
        int lda = sigmas_symm.shape[1]
        int[::1] piv = np.zeros(sigmas_symm.shape[1], dtype = np.intc, order = "F")
        int rank = 0
        double tol = -1.0
        double[::1] work = np.zeros(2*n, order="F")
        int info = 0
        double[::1, :, :] solutions = np.zeros((l, sigmas_symm.shape[1], sigmas_symm.shape[1]), order="F")
        double[::1, :] sigm_current = np.zeros((sigmas_symm.shape[1], sigmas_symm.shape[1]), order = "F")
        int ia = 0
        int jia = 0

    for i in range(l):
        sigm_current = sigmas_symm[i,:,:].copy_fortran()
        dpotrf(type_output, &n, &sigm_current[0, 0], &lda, &info)
        sigm_current[0, 1] = 0.0
        sigm_current[0, 2] = 0.0
        sigm_current[1, 2] = 0.0

        dpotri(type_output, &n, &sigm_current[0, 0], &lda ,&info)
        sigm_current[0, 1] = sigm_current[1, 0]
        sigm_current[0, 2] = sigm_current[2, 0]
        sigm_current[1, 2] = sigm_current[2, 1]

        solutions.base[i, :, :] = sigm_current.copy_fortran()


    #solutions = generate_polarization_var_cl_cython(solutions)

    return solutions, info