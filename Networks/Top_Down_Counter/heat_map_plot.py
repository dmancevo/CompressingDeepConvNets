import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split



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
	'log_counts':np.log(1+tst),
	'n_boxes':n_boxes,
	'scores':scores,
})

# sns.jointplot('n_boxes','counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('n_boxes','log_counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('scores','counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('scores','log_counts',data=df, kind='reg')
# plt.show()

reg = ElasticNet(alpha=1.0, l1_ratio=0.5)

X_trn, X_tst, y_trn, y_tst = train_test_split(
	df[['n_boxes','scores']],
	df.log_counts,
	test_size=0.3,
)

reg.fit(X_trn, y_trn)
print "R2 Score: ", reg.score(X_tst, y_tst)