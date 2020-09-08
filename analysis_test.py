import numpy as np
import matplotlib.pyplot as plt


d = np.load("test_gibbs_non_change_variable.npy", allow_pickle=True)
d = d.item()

h_cls = d["h_cls"]


print(h_cls.shape)

plt.plot(h_cls[:, 300])
plt.show()
