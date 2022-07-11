import numpy as np
import matplotlib.pyplot as plt

sigma_prior = 0.01
sigma_likelihood = 1
mu = 1
beta = 0.1
z = np.random.normal(scale = np.sqrt(sigma_prior))
d = z + np.random.normal(scale=np.sqrt(sigma_likelihood))
d = 100

def computing_log_lik(s):
    return -0.5*(s)**2/sigma_likelihood


def CN(N):
    h = []
    h_accept = []
    s = 0
    h.append(s)
    for i in range(N):
        print(i)
        s_new = np.sqrt(1-beta**2)*(s+d) - d +beta*np.random.normal(scale=np.sqrt(sigma_prior))
        log_new = computing_log_lik(s_new)
        log_old = computing_log_lik(s)
        if np.log(np.random.uniform()) < log_new - log_old:
            h.append(s_new)
            h_accept.append(1)
            s = s_new
        else:
            h.append(s)
            h_accept.append(0)


    print("Acceptance rate:", np.mean(h_accept))
    return h


h = CN(100000)
plt.plot(h)
plt.show()
