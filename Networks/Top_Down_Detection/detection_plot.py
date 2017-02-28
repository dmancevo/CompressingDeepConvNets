import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter as mf

with open("y_hat.pkl", "rb") as f:
	y_hat = pkl.load(f)


M = np.array([np.logical_and(block==mf(block,size=81),block>0.95) \
	for block in y_hat[:,:,:,1]])

j=1
for i in np.random.choice(range(y_hat.shape[0]), size=1, replace=False):

	plt.matshow(y_hat[i,:,:,1])
	plt.matshow(M[i,:,:])

	j+=1

plt.show()
