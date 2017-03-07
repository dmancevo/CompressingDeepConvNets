import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


with open("h_map.pkl", "rb") as f:
	counts, est, h_map = pkl.load(f)

for i in range(8):
	plt.matshow(h_map[i,:,:])
	plt.title("Actual: {0}, Estimate: {1}".format(counts[i], est[i]))
	plt.show()


# M = np.array([np.logical_and(block==mf(block,size=81),block>0.95) \
# 	for block in h_map[:,:,:,1]])

# j=1
# for i in np.random.choice(range(h_map.shape[0]), size=1, replace=False):

# 	plt.matshow(h_map[i,:,:,1])
# 	plt.matshow(M[i,:,:])

# 	j+=1

# plt.show()
