import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

PATH  = "../Networks/"
N_AVG = 15

dir_reg   = re.compile(r'.*Networks/(.+)_Student_Networks/saved/(student_\d)_([^\/]+)\/(.+)')
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

colors = {
	"CIFAR_10": {
		"student_1":{
			"N1_H32": "#3498db",
			"N2_H32": "#34495e",
			"N2_H64": "#9b59b6",
		}
	},
	"Top_Down": {
		"student_1":{
			"20_50": "#3498db",
			"10_15": "#34495e",
			"7_10": "#9b59b6",
		},
		"student_2":{
			"25": "#3498db",
			"38": "#34495e",
		}
	}
}

for dirpath, dirnames, filenames in os.walk(PATH):


	mtch    = dir_reg.match(dirpath)

	if mtch:

		try:
			with open(dirpath+"/log.txt", "r") as f:
				raw_log = f.readlines()

			dataset, net, config, algo = mtch.group(1), mtch.group(2), mtch.group(3), mtch.group(4)

			if dataset not in series:
				series[dataset] = {}

			if net not in series[dataset]:
				series[dataset][net] = {}

			if config not in series[dataset][net]:
				series[dataset][net][config] = []

			err    = pd.Series(err_reg.findall(" ".join(raw_log)), dtype=float)
			m_avg  = pd.rolling_mean(err, N_AVG)
			series[dataset][net][config].append((algo, m_avg))

		except IOError:
			print dirpath, "has no log file."

for dataset in series.keys():

	for net in series[dataset].keys():
		ax=None
		for config in series[dataset][net].keys():

			for errs in series[dataset][net][config]:


				algo, m_avg = errs

				if len(m_avg)==0: continue

				# m_std = pd.rolling_std(m_avg, N_AVG, ddof=0)

				if re.match(r'baseline',algo):
					marker = ','
				elif re.match(r'reg_logits',algo):
					marker='*'
				elif re.match(r'know_dist',algo):
					marker='d'

				if not ax:
					ax = m_avg.plot(kind='line', title="{0} {1}".format(dataset, net), 
					marker=marker, label="{0} {1}".format(config, algo),
					legend=True, color=colors[dataset][net][config])#, yerr=m_std)
				else:
					m_avg.plot(kind='line', ax=ax, marker=marker,
					label="{0} {1}".format(config, algo),
					legend=True, color=colors[dataset][net][config])#, yerr=m_std)

		for teacher, err in teachers[dataset]:
			ax.plot(
				(0,100),(err,err),
				'--',
				label="Teacher {0}".format(teacher),
			)
			ax.legend()
			ax.set_ylabel("Err. Moving Avg. ({0})".format(N_AVG))
			ax.set_xlabel("Epochs")

		plt.show()
		# plt.savefig(dataset+" "+net+".png")
		# plt.close()
