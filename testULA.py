import numpy as np
import matplotlib.pyplot as plt

alpha_star = 2
mu_star = 1

s_star = np.random.normal(scale=np.sqrt(alpha_star)) + mu_star
d = s_star + np.random.normal()

def posterior(x):
    return np.exp(-0.5*(x-d)**2/(alpha_star+1))/np.sqrt(2*np.pi*(alpha_star+1))

def sample_posterior():
    return np.random.normal(size=100000)*np.sqrt(alpha_star+1) + d

def mean_sample(s):
    return np.random.normal(loc=s, scale=np.sqrt(alpha_star))

def latent_sample(mu):
    sigma = 1/(1+(1/alpha_star))
    return np.random.normal(loc = sigma*(d+mu/alpha_star), scale = np.sqrt(sigma))

def ula(mu, s_old):
    sigma = 1/(1+(1/alpha_star))
    tau =0.2
    mean = sigma*(d+(mu/alpha_star))
    grad = - (1/sigma)*(s_old - mean)
    s_new = s_old +tau*sigma*grad + np.sqrt(2*tau*sigma)*np.random.normal()
    return s_new

def cond(x):
    sigma = 1/(1+(1/alpha_star))
    mean = sigma*(d+mu_star/alpha_star)
    return np.exp(-0.5*(x-mean)**2/sigma)/np.sqrt(2*np.pi*sigma)

def sample_cond():
    sigma = 1/(1+(1/alpha_star))
    mean = sigma*(d+mu_star/alpha_star)
    return np.random.normal(size=100000)*np.sqrt(sigma) + mean


h_mu = []
mu = mu_star
h_mu.append(mu)
for i in range(100000):
    s = latent_sample(mu)
    mu = latent_sample(s)
    h_mu.append(mu)


h_ula = []
mu = mu_star
s = latent_sample(mu)
mu = latent_sample(s)
h_ula.append(mu)
for i in range(1000000):
    mu = mean_sample(s)
    s = ula(mu, s)
    h_ula.append(mu)


x = np.linspace(d-4, d+4, 100)
y = [posterior(xx) for xx in x]
#plt.hist(sample_posterior(), density=True, bins=50, alpha=0.5)
#plt.hist(h_mu, density=True, bins=50, alpha=0.5)
plt.hist(h_ula, density=True, bins=100, alpha=0.5)
plt.plot(x,y)
plt.show()

"""
h_ula = []
mu = mu_star
s = latent_sample(mu)
h_ula.append(s)
for i in range(100000):
    s = ula(mu, s)
    h_ula.append(s)



sigma = 1/(1+(1/alpha_star))
mean = sigma*(d+mu_star/alpha_star)
x = np.linspace(mean-4, mean+4, 100)
y = [cond(xx) for xx in x]
plt.hist(h_ula[:], density=True, bins=50, alpha=0.5)
#plt.hist(sample_cond(), density=True, bins=50, alpha=0.5)
plt.plot(x,y)
plt.show()
"""