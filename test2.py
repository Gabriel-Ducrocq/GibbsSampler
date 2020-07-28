import healpy as hp
import numpy as np
import config
import matplotlib.pyplot as plt
import config
import utils
from CenteredGibbs import PolarizedCenteredConstrainedRealization
from scipy.stats import norm, invgamma, invwishart
from scipy.stats import t as student
import scipy
import time
from scipy.special import gammaln, multigammaln
import mpmath



def get_points(a, b, n):
    return np.array([(1/2)*(a+b) + (1/2)*(a-b)*np.cos(((2*k-1)/(2*n))*np.pi) for k in range(0, n)])


def compute_norm(l, scale_mat, cl_EE, cl_TE):
    k = scale_mat[1, 1]*cl_TE**2/cl_EE - 2*scale_mat[0, 1]*cl_TE + scale_mat[0, 0]*cl_EE

    return -(2*l-2)*np.log(2) - multigammaln((2*l-2)/2, 2) + ((2*l+1)/2)*np.log(2) - np.log(cl_EE) + ((2*l-2)/2)*np.log(np.linalg.det(scale_mat))\
    - ((2*l-3)/2)*np.log(2*l+1) - ((2*l-1)/2)*np.log(k) + gammaln((2*l-1)/2) - ((2*l+1)/2)*scale_mat[1, 1]/cl_EE



def compute_conditional_TT(x, l, scale_mat, cl_EE, cl_TE):
    param_mat = np.array([[x, cl_TE], [cl_TE, cl_EE]])
    if x <= cl_TE**2/cl_EE:
        return 0
    else:
        return invwishart.pdf(param_mat, df=2*l-2, scale=scale_mat)


def compute_conditional_TT_rescaled(x, l, scale_mat, cl_EE, cl_TE, maximum):
    return compute_conditional_TT(maximum*x, l, scale_mat, cl_EE, cl_TE)


def root_to_find(x, u, l, scale_mat, cl_EE, cl_TE, norm):
    low_bound = cl_TE**2/cl_EE
    integral, err = scipy.integrate.quad(compute_conditional_TT, a=low_bound, b=x, args=(l, scale_mat, cl_EE, cl_TE))
    return integral/norm - u


def sample(l, scale_mat):
    beta_EE = scale_mat[1, 1]
    cl_EE = invgamma.rvs(a = (2*l-3)/2, scale = beta_EE/2)
    student_TE = student.rvs(df = 2*l-2)
    determinant = np.linalg.det(scale_mat)
    cl_TE = (np.sqrt(determinant)*cl_EE*student_TE/np.sqrt(2*l-2) + scale_mat[0, 1]*cl_EE)/scale_mat[1, 1]
    u = np.random.uniform()
    ratio = cl_TE**2/cl_EE
    maximum = (cl_EE ** 2 * scale_mat[ 0, 0] + cl_TE ** 2 * scale_mat[1, 1] + cl_TE ** 2 * (
                2 * l + 1) * cl_EE - 2 * cl_TE * cl_EE * scale_mat[ 0, 1]) / ((2 * l + 1) * cl_EE ** 2)
    norm1, err = scipy.integrate.quad(compute_conditional_TT, a=ratio, b=maximum, args=(l, scale_mat, cl_EE, cl_TE))
    norm2, err = scipy.integrate.quad(compute_conditional_TT, a=maximum, b=np.inf, args=(l, scale_mat, cl_EE, cl_TE))
    norm = norm1 + norm2

    print("NORMS")
    print(norm)
    print(np.exp(norm2))
    #sol = scipy.optimize.bisect(root_to_find, a=ratio,b=1000, args=(u, l, scale_mat, cl_EE, cl_TE, norm))
    sol = scipy.optimize.root_scalar(root_to_find, x0=maximum -0.1, x1=maximum+0.1, args=(u, l, scale_mat, cl_EE, cl_TE, norm))
    print(sol)
    print(cl_TE**2/cl_EE)
    print("\n")
    return sol, cl_EE, cl_TE, 1



def trace_cdf(u, l, scale_mat, cl_EE, cl_TE):
    low_bound = cl_TE ** 2 / cl_EE
    #norm, err = scipy.integrate.quad(compute_conditional_TT, a=low_bound, b=np.inf, args=(l, scale_mat, cl_EE, cl_TE))
    log_norm = compute_norm(l, scale_mat, cl_EE, cl_TE)
    norm = np.exp(log_norm)
    y = []
    x = np.linspace(low_bound, 100, 1000)
    for i in x:
        print(i)

        integral, err = scipy.integrate.quad(compute_conditional_TT, a=low_bound, b=i,
                                             args=(l, scale_mat, cl_EE, cl_TE))
        y.append(integral/norm)
        print("ERROR")
        print(err)

    plt.plot(x,y)
    plt.axhline(y = 0)
    plt.show()


def trace_pdf(l, scale_mat, cl_EE, cl_TE):
    low_bound = cl_TE ** 2 / cl_EE
    norm, err = scipy.integrate.quad(compute_conditional_TT, a=low_bound, b=np.inf, args=(l, scale_mat, cl_EE, cl_TE))
    y = []
    y_norm = []
    maximum = (cl_EE ** 2 * scale_mat[ 0, 0] + cl_TE ** 2 * scale_mat[1, 1] + cl_TE ** 2 * (
                2 * l + 1) * cl_EE - 2 * cl_TE * cl_EE * scale_mat[ 0, 1]) / ((2 * l + 1) * cl_EE ** 2)

    max2 = (cl_EE ** 2 * scale_mat[0, 0] + cl_TE ** 2 * scale_mat[1, 1] + cl_TE ** 2 * cl_EE - 2 * cl_TE * cl_EE *
            scale_mat[0, 1]) / (cl_EE ** 2)

    precision = -compute_conditional_TT (maximum, l, scale_mat, cl_EE, cl_TE)*(1/(maximum*cl_EE - cl_TE**2 )**2)*(((2*l+1)/2)*cl_EE**2 - (scale_mat[1, 1]*cl_TE**2 + scale_mat[0, 0]*cl_EE**2 - 2* scale_mat[0, 1]*cl_TE*cl_EE)*cl_EE/(maximum*cl_EE - cl_TE**2 ))
    var = 1/precision
    stdd = np.sqrt(var)
    deg = 2**12 + 1
    #cheb_nodes = get_points(ratio/maximum, (10*stdd+maximum)/maximum, deg-1)
    #cheb_y = np.array([compute_conditional_TT_rescaled(node, l, scale_mat, cl_EE, cl_TE, maximum) for node in cheb_nodes])
    #cheb_coefs = np.polynomial.chebyshev.chebfit(cheb_nodes, cheb_y, deg)
    x = np.linspace(low_bound/maximum, (maximum+10*stdd)/maximum, 10000)
    #cheb_y = np.array([np.polynomial.chebyshev.chebval(l, cheb_coefs) for l in x])
    for i in x:
        print(i)
        normal = scipy.stats.norm.pdf(i, loc=maximum, scale=np.sqrt(var))
        y_norm.append(normal)
        integral = compute_conditional_TT_rescaled(i, l, scale_mat, cl_EE, cl_TE, maximum)
        y.append(integral)
        print("ERROR")
        print(err)

    print("MAXIMUM")
    print(maximum)
    print("NORM")
    print(norm)
    print("var")
    print(var)
    print("ERROR CHEB")
    #print(max(np.abs(cheb_coefs[-5:])))
    #plt.plot(cheb_coefs)
    #plt.show()
    print("All coefs")
    #print(np.sort(np.abs(cheb_coefs)))
    plt.plot(x,np.log(y), label="True function")
    #plt.plot(x, y_norm, label="Normal approximation")
    #plt.plot(x, cheb_y, label="Chebyshev")
    plt.axvline(x=maximum)
    plt.legend(loc="upper right")
    plt.show()



map = [np.random.normal(size = config.Npix), np.random.normal(size = config.Npix), np.random.normal(size = config.Npix)]
alms = hp.map2alm(map)
pow_spec_TT, pow_spec_EE, _, pow_spec_TE, _, _ = hp.alm2cl(alms, lmax=config.L_MAX_SCALARS)

l_interest = 7

scale_mat = np.zeros((2, 2))
scale_mat[0, 0] = pow_spec_TT[l_interest]
scale_mat[1, 1] = pow_spec_EE[l_interest]
scale_mat[1, 0] = scale_mat[0, 1] = pow_spec_TE[l_interest]
scale_mat *= (2*l_interest+1)

h_cond = []
h_successes = []
start = time.time()

cl_EE = 0.028135052007328482
cl_TE = -0.0088654407928456
u = 0.07687580214076828
ratio = cl_TE**2/cl_EE
norm, err = scipy.integrate.quad(compute_conditional_TT, a=ratio, b=np.inf, args=(l_interest, scale_mat, cl_EE, cl_TE))
b, err = scipy.integrate.quad(compute_conditional_TT, a=ratio, b=10000, args=(l_interest, scale_mat, cl_EE, cl_TE))
b = b / norm - u
print("B")
print(b)


#trace_cdf(0.07687580214076828, l_interest, scale_mat, 0.028135052007328482, -0.0088654407928456)
trace_pdf(l_interest, scale_mat, 0.028135052007328482, -0.0088654407928456)

for i in range(1000):
    if i % 100 == 0:
        print("Numerical inversion, iteration",i)

    cl_TT, cl_EE, cl_TE, success = sample(l_interest, scale_mat)
    h_cond.append(cl_TT)
    h_successes.append(success)

end = time.time()
print("Time numerical inversion:", end-start)

h_direct = []
for i in range(1000):
    mat_sample = invwishart.rvs(df=2*l_interest-2, scale=scale_mat)
    h_direct.append(mat_sample[0, 0])


d = {"h_cond":np.array(h_cond), "h_direct":np.array(h_direct), "h_successes":np.array(h_successes)}
np.save("numeric_inverse_test.npy", d, allow_pickle=True)

d = np.load("numeric_inverse_test.npy", allow_pickle=True)
d = d.item()
h_cond = d["h_cond"]
h_direct = d["h_direct"]
h_successes = d["h_successes"]


plt.hist(h_cond[h_successes==True], label="Cond", alpha=0.5, density=True, bins = 45)
plt.hist(h_direct, label="Direct", alpha = 0.5, density=True, bins = 45)
plt.legend(loc="upper right")
plt.show()


print(len(h_cond[h_successes==True]))