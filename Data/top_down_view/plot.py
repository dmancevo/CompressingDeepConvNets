import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from copy import deepcopy

N = 50
CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

def plot(imgs, labels):
	i = 1
	print imgs.shape
	for i in range(len(imgs)):
		plt.subplot(5,10,i)
		plt.imshow(imgs[i])
		plt.title(str(labels[i]))
		plt.axis('off')
		i+=1
	plt.show()


if __name__ == '__main__':
	import pickle as pkl

	with open("train.pkl", "rb") as f:
		labels, crops = pkl.load(f)

	I = np.random.choice(range(len(labels)), size=N, replace=False)

	# plot(imgs, labels)
	plot(crops[I], labels[I])