import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from filtering_and_suppression import head_counting


# with open("heat_maps.pkl", "rb") as f:
# 	labels, heat_maps = pkl.load(f)


# for heat_map in heat_maps:
# 	heat_map = gf(heat_map, sigma=7)
# 	plt.matshow(heat_map)
# 	y_hat = head_counting("Estimate: {0}".format(heat_map))
# 	plt.title(y_hat)
# 	plt.show()
# 	plt.close()

with open("data.pkl","wb") as f:
	tst, n_boxes, scores = pkl.load(f)

df = pd.DataFrame({'counts':tst,'n_boxes':n_boxes,'scores':scores})
sns.lmplot(df)
