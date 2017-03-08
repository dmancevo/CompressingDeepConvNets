import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from filtering_and_suppression import head_counting


with open("heat_maps.pkl", "rb") as f:
	labels, heat_maps = pkl.load(f)


for heat_map in heat_maps:
	heat_map = gf(heat_map, sigma=7)
	plt.matshow(heat_map)
	y_hat = head_counting("Estimate: {0}".format(heat_map))
	plt.title(y_hat)
	plt.show()
	plt.close()