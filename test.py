import numpy as np
from scipy.sparse import block_diag
from scipy.sparse.linalg import inv, spsolve
import time
from linear_algebra import product_cls_inverse, compute_cholesky


start = time.time()
list_blocks =[]
list_b = []
for i in range(1000000):
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
result_cython = product_cls_inverse(mat, b, 1000000)
end_cython = time.time()
print("Time Cython:")
print(end_cython-start_cython)


start = time.time()
L, info, sigm_current = compute_cholesky(mat, 1000000)
end = time.time()
print("Time cython:", end - start)

print("\n")
print(np.linalg.cholesky(list_blocks[0]))
print("\n")
print(np.array(L)[0, :, :])





