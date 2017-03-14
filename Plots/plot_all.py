import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

PATH  = "../Networks/"
N_AVG = 15

dir_reg   = re.compile(r'.*Networks/(.+)_Student_Networks/saved/(student_[^\/]+)\/(.+)')
err_reg   = re.compile(r'Err:  (\d\.\d+)')

series = {}

teachers = {
	"CIFAR_10": (
		("VGG-16", 0.1407),
	),
	"Top_Down": (
		("", 0.0495833333333),
		("(8 rounds of model avg.)", 0.0422916666667),
	)
}

for dirpath, dirnames, filenames in os.walk(PATH):


	mtch    = dir_reg.match(dirpath)

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
			m_avg  = pd.rolling_mean(err, N_AVG)
			series[dataset][net].append((algo, m_avg))

		except IOError:
			print dirpath, "has no log file."

for dataset in series.keys():

	for net in series[dataset].keys():

		errs = series[dataset][net]

		for i in range(len(errs)):
			algo, m_avg = errs[i]

			if len(m_avg)==0: continue

			# m_std = pd.rolling_std(m_avg, N_AVG, ddof=0)

			if re.match(r'baseline',algo):
				marker = ','
			elif re.match(r'reg_logits',algo):
				marker='^'
			elif re.match(r'know_dist',algo):
				marker='*'

			if i==0:
				ax = m_avg.plot(kind='line', title="{0} {1}".format(dataset, net),
				marker=marker, label=algo, legend=True)#, yerr=m_std)
			else:
				m_avg.plot(kind='line', marker=marker,
				label=algo, legend=True)#, yerr=m_std)

		for teacher, err in teachers[dataset]:
			ax.plot(
				(0,len(m_avg)),(err,err),
				'--',
				label="Teacher {0}".format(teacher),
			)
			ax.legend()
			ax.set_ylabel("Err. Moving Avg. ({0})".format(N_AVG))
			ax.set_xlabel("Epochs")

		plt.show()
		# plt.savefig(dataset+" "+net+".png")
		# plt.close()
