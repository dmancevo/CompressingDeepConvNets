import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PATH = "/notebooks/Networks/"
PATH = "../Networks/"

dir_reg = re.compile(r'.*Networks/(.+)_Student_Networks/saved/(student_\d)\/(.+)')

err_reg = re.compile(r'Err:  (\d\.\d+)')

series = {}

for dirpath, dirnames, filenames in os.walk(PATH):


	mtch = dir_reg.match(dirpath)

	if mtch:

		try:
			with open(dirpath+"/log.txt", "r") as f:
				raw_log = f.readlines()

			dataset, net, algo = mtch.group(1), mtch.group(2), mtch.group(3)

			if dataset not in series:
				series[dataset] = {}

			if net not in series[dataset]:
				series[dataset][net] = []

			err    = pd.Series(err_reg.findall(" ".join(raw_log)), dtype=float)
			m_avg  = pd.rolling_mean(err, 15)
			series[dataset][net].append((algo, m_avg))

		except IOError:
			print dirpath, "has no log file."

for dataset in series.keys():

	for net in series[dataset].keys():

		errs = series[dataset][net]

		for i in range(len(errs)):
			algo, m_avg = errs[i]

			if i==0:
				ax = m_avg.plot(kind='line', title=dataset+" "+net,
				label=algo, legend=True)
			else:
				m_avg.plot(kind='line',
				label=algo, legend=True, ax=ax)

		ax.set_ylabel("Err. Moving Avg. (15)")
		ax.set_xlabel("Epochs")

		plt.show()
		# plt.savefig(net+".png")
		# plt.close()
