import numpy as np
import matplotlib.pyplot as plt

dim = 10
alpha_star = 0.1
tau = 0.1
sig = 0.01

s_star = np.random.normal(size=dim)*np.sqrt(alpha_star)
d = s_star + np.random.normal(size=dim)


def posterior(x):
    return invgamma.pdf(x, a=(dim/2) - 1, loc = -1, scale =np.sum(d**2)/2)

def compute_log_density(param, latent):
    return -0.5*param*np.sum(latent**2) + np.sqrt(param)*np.dot(latent, d)

def sample_param(old_param, latent):
    new_param = old_param + np.random.normal(scale=sig)
    log_ratio = compute_log_density(new_param, latent) - compute_log_density(old_param, latent)
    if np.log(np.random.uniform()) < log_ratio:
        return new_param, 1

    return old_param, 0


def sample_latent(alpha):
    sigma = 1/(1+(1/alpha))
    mu = sigma*d
    return (np.sqrt(sigma)*np.random.normal(size=dim) + mu)/np.sqrt(alpha)


def ula(s, alpha):
    sigma = 1/(1+(1/alpha))
    mu = sigma*d
    #tau = 0.9*(1/(2*sigma))
    grad = -(1/sigma)*(s-mu)
    sampled = s + tau*grad + np.sqrt(2*tau)*np.random.normal()
    sampled /=np.sqrt(alpha)
    return sampled



h_gibbs = []
h_accept = []
alpha = alpha_star
s = sample_latent(alpha)
for i in range(100000):
    print(alpha)
    alpha, acc = sample_param(alpha, s)
    h_accept.append(acc)
    #s = ula(s, alpha)
    s = sample_latent(alpha)
    h_gibbs.append(alpha)
    #h_gibbs.append(alpha)



print("Acc rate:", np.mean(h_accept))

plt.plot(h_gibbs[:])
plt.show()
mode = np.sum(d**2)/(dim/2)
print(mode)
x = np.linspace(np.max(mode-100, 0), mode+100, 10000)
y = [posterior(xx) for xx in x]
#plt.plot(x,y)
plt.hist(h_gibbs, density=True, alpha = 0.5, bins = 600)
plt.show()