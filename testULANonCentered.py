import numpy as np

dim = 10
alpha_star = 1
tau = 0.01
sig = 0.1

s_star = np.random.normal(size=dim)*np.sqrt(alpha_star)
d = s_star + np.random.normal(size=dim)


def posterior(x):
    return invgamma.pdf(x, a=(dim/2) - 1, loc = -1, scale =np.sum(d**2)/2)

def sample_param(s):
    return invgamma.rvs(a=(dim/2)-1, scale= np.sum(s**2)/2)


def compute_log_density(param, latent):
    return -0.5*param*np.sum(latent**2) + np.sqrt(param)*np.dot(latent, d)

def sample_param(old_param):
    new_param = old_param + np.random.normal(scale=sig)
    log_ratio = compute_log_density(new_param) - compute_log_density(old_param)
    if np.log(np.random.uniform()) < log_ratio:
        return new_param, 1

    return old_param, 0


def sample_latent(alpha):
    sigma = 1/(1+(1/alpha))
    mu = sigma*d
    return np.sqrt(sigma)*np.random.normal(size=dim) + mu


def ula(s, alpha):
    sigma = 1/(1+(1/alpha))
    mu = sigma*d
    tau = 0.9*(1/(2*sigma))
    grad = -(1/sigma)*(s-mu)
    return s + tau*grad + np.sqrt(2*tau)*np.random.normal()

