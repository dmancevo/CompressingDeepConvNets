import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



# with open("heat_maps.pkl", "rb") as f:
# 	labels, heat_maps = pkl.load(f)


# for heat_map in heat_maps:
# 	heat_map = gf(heat_map, sigma=7)
# 	plt.matshow(heat_map)
# 	y_hat = head_counting("Estimate: {0}".format(heat_map))
# 	plt.title(y_hat)
# 	plt.show()
# 	plt.close()

with open("test.pkl","rb") as f:
	tst, tst_n_boxes, tst_scores = pkl.load(f)

tst = tst.astype(float)

df = pd.DataFrame({
	'counts':tst,
	'log_counts':np.log(1+tst),
	'n_boxes':tst_n_boxes,
	'scores':tst_scores,
})

# sns.jointplot('n_boxes','counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('n_boxes','log_counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('scores','counts',data=df, kind='reg')
# plt.show()
# sns.jointplot('scores','log_counts',data=df, kind='reg')
# plt.show()

r2, est, act, = [], [], []
for _ in range(1000):
	reg = LinearRegression()

	X_trn, X_tst, y_trn, y_tst = train_test_split(
		df[['n_boxes','scores']],
		df.log_counts,
		test_size=0.5,
	)

	reg.fit(X_trn, y_trn)

	r2.append(reg.score(X_tst, y_tst))
	est.append(np.sum(reg.predict(X_tst)))
	act.append(np.sum(y_tst))

est, act = np.array(est), np.array(act)


print "R2 Score: ", np.mean(r2)
print "Mean Actual", np.mean(act)
print "MRSE: ", np.mean(np.sqrt(np.power(est-act,2)))