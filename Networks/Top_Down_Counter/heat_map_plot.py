import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# with open("heat_maps.pkl", "rb") as f:
# 	labels, heat_maps = pkl.load(f)


# for heat_map in heat_maps:
# 	heat_map = gf(heat_map, sigma=7)
# 	plt.matshow(heat_map)
# 	y_hat = head_counting("Estimate: {0}".format(heat_map))
# 	plt.title(y_hat)
# 	plt.show()
# 	plt.close()

with open("data.pkl","rb") as f:
	tst, n_boxes, scores = pkl.load(f)

tst = tst.astype(float)

df = pd.DataFrame({
	'counts':tst,
	'log counts':np.log(1+tst),
	'n_boxes':n_boxes,
	'scores':scores,
})

# sns.lmplot('n_boxes','counts',df, legend="hello")
# plt.show()
# sns.lmplot('scores','counts',df)
# plt.show()

sns.jointplot('n_boxes','counts',data=df, kind='reg')
plt.show()
sns.jointplot('n_boxes','log counts',data=df, kind='reg')
plt.show()
sns.jointplot('scores','counts',data=df, kind='reg')
plt.show()
sns.jointplot('scores','log counts',data=df, kind='reg')
plt.show()

# print("R2 Score: ")