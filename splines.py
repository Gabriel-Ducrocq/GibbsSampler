import numpy as np
from numba import njit
from scipy import interpolate

gr = (1+np.sqrt(5))/2

@njit
def golden_search_ratio(minus_log_func, x1, x3, tol = 1e-8):
    y1, y3= minus_log_func(x1), minus_log_func(x3)
    if y1 < y3:
        a,b = x1, x3
        y1_new = y1
        while y1_new <= y1:
            x1_new = a - gr*np.abs(b-a)
            y1_new = minus_log_func(x1_new)
            a = x1_new
            b = x1

        x1 = a

    else:
        a,b = x1, x3
        y3_new = y3
        while y3_new <= y3:
            x3_new = b + gr*np.abs(b-a)
            y3_new = minus_log_func(x3_new)
            a = x3
            b = x3_new

        x3 = b


    err = 10
    while err >= tol:
        x2 = (x3 + x1 * gr) / (1 + gr)
        x4 = x1 + (x3 - x2)
        y2, y4 = minus_log_func(x2), minus_log_func(x4)
        err = 1
        if y2 < y4:
            x3 = x4
            y3 = y4
        else:
            x1 = x2
            y1 = y2

        err = x4 - x1
        print(err)

    return (x1+x3)/2

@njit
def find_upper_bound(log_func, mode, hard_upper_bound=np.inf):
    log_func_max = log_func(mode)
    x1 = mode + gr* np.abs(mode)
    x0 = mode
    while log_func_max - log_func(x1) < 12.5:
        x1_new = min(x1 + gr * np.abs((x1 - x0)), hard_upper_bound)
        x0 = x1
        x1 = x1_new

    return x1

@njit
def find_lower_bound(log_func, mode, hard_lower_bound=-np.inf):
    log_func_max = log_func(mode)
    x1 = max(mode - gr * np.abs(mode), hard_lower_bound)
    x0 = mode
    while log_func_max - log_func(x1) < 12.5:
        x1_new = max(x1 - gr * np.abs((x1 - x0)), hard_lower_bound)
        x0 = x1
        x1 = x1_new

    return x1

def spline_approximation(func, lower_bound, upper_bound, tol = 1e-10):
    i = 1
    max_err = tol + 0.1
    while max_err > tol:
        xx = np.linspace(lower_bound, upper_bound, (2**i) * 1000)
        y_cs = np.array([func(x) for x in xx])
        cs = interpolate.CubicSpline(xx, y_cs)

        new_x = (xx[:-1] + xx[1:])/2
        new_y = np.array([func(x) for x in new_x])
        cs_y_new = np.array([cs(x) for x in new_x])
        max_err = np.max(np.abs(cs_y_new - new_y))
        i += 1

    integs = np.array([cs.integrate(lower_bound, x) for x in xx])
    integs /= integs[-1]
    return cs, integs, xx

@njit
def sampling(integs, xx):
    u = np.random.uniform(0, 1, 1)
    position = np.searchsorted(integs, u)
    sample = (u - integs[position - 1]) * (xx[position] - xx[position - 1]) / (
                integs[position] - integs[position - 1]) + xx[position - 1]
    return sample


def sample_splines(log_func, hard_lower_bound=-np.inf, hard_upper_bound=np.inf, tol_gsr=1e-10):
    @njit
    def minus_log_funct(x):
        return - log_func(x)

    @njit
    def funct(x):
        return np.exp(log_func(x))

    mode = golden_search_ratio(minus_log_funct, tol_gsr, 100000)
    upper_bound = find_upper_bound(log_func, mode, hard_upper_bound = hard_upper_bound)
    lower_bound = find_lower_bound(log_func, mode, hard_lower_bound=hard_lower_bound)
    cs, integs, xx = spline_approximation(funct, lower_bound, upper_bound)
    sampled_point = sampling(integs, xx)
    return sampled_point

