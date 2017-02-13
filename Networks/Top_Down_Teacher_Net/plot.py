import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from copy import deepcopy

CROP_HEIGHT, CROP_WIDTH = 60, 60
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

	with open("imgages.pkl", "rb") as f:
		images = pkl.load(f)

	imgs, crops, labels = images


	# plot(imgs, labels)
	plot(crops, labels)