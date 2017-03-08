import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import r2_score

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage.filters import maximum_filter as mf

import threading


FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1/\
know_dist_T10.0_beta0.05/"

DATA_PATH = "/notebooks/Data/top_down_view/images/"

def h_counter(heat_map, sigma=7, th=0.9, size=101):
	'''
	Count human subjects on heat_map.
	'''
	heat_map = gf(heat_map, sigma=sigma)
	y, x = np.where(np.logical_and(
		mf(heat_map,size=size)==heat_map,
		th<heat_map,
	))
	return len(x)



def head_counter(heat_map, N=30,
	sigma=(9.23, 1.5), th=(0.81, 0.039), size=(79, 40)):
	'''
	Apply and average h_counter.

	mu_param: mean parameter values
	std_param: std. dev parameter values.
	M: number of samples.
	'''
	counts  = np.empty(shape=(N,))
	class countingThread(threading.Thread):

		def __init__(self, threadID):
			threading.Thread.__init__(self)
			self.id = threadID

		def run(self):

			params = (
				np.random.normal(*sigma),
				np.clip(np.random.normal(*th), 0.50, 0.99),
				np.clip(int(np.round(np.random.normal(*size))), 15, 250)
			)

			counts[self.id] = h_counter(heat_map, *params)

	counting_threads = [countingThread(i) for i in range(N)]

	for t in counting_threads:
		t.start()

	for t in counting_threads:
		t.join()

	return np.mean(counts)

def score(heat_maps, head_counts, params):
	'''
	Evaluate head counter.

	return R2 score.
	'''
	y_hat = []
	for heat_map in heat_maps:
		subjects = head_counter(heat_map, *params)
		y_hat.append(subjects)
	y_hat = np.reshape(y_hat, newshape=(len(heat_maps),1))
	return r2_score(head_counts, y_hat)

def head_counter_optimize(heat_maps, head_counts, params, it=1000):
	'''
	Use simulated annealing to find optimal parameters.
	'''
	best_params = params
	best_score  = score(heat_maps, head_counts, best_params)

	print "R2: ", best_score, "; params: ", best_params

	curr_params = best_params
	curr_score  = best_score

	for i in range(1,it/4):

		T = i/float(it)
		prop_params = (
			int(np.clip(np.random.normal(30, 15), 15, 50)),
			(np.clip(np.random.normal(curr_params[1][0], 2), 0.5, 100),
				1.5),
			(np.clip(np.random.normal(curr_params[2][0], 0.03), 0.50, 0.99),
				0.039),
			(int(np.round(np.random.normal(curr_params[3][0], 15))),
				40),
		)
		prop_score = score(heat_maps, head_counts, prop_params)

		if curr_score < prop_score and \
		np.exp(-(prop_score-curr_score)/T)<np.random.uniform(0,1):

			curr_params = prop_params
			curr_score  = prop_score

		if best_score<prop_score:
			best_score  = prop_score
			best_params = prop_params

			print "R2: ", best_score, "; params: ", best_params

	return best_params

def sample_heat_maps(tt, M):
	'''
	Sample images and produce corresponding heat_maps.

	tt: "train" or "test"
	M: number of samples.

	returns labels and heatmaps.
	'''
	with open("{0}/meta.pkl".format(DATA_PATH), "rb") as f:
		meta = pkl.load(f)

	file_names  = np.array(meta[tt].keys())
	head_counts = np.array([len(meta[tt][fn]) for fn in file_names])

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("{0}student_1.meta".format(FOLDER))
		saver.restore(sess, tf.train.latest_checkpoint(FOLDER))
		layers = tf.get_collection('layers')

		labels, t_logits, images, keep_prob, training, augment,\
		prob, f_prob, train_step = layers

		h_map  = prob[0,:,:,1]

		J, heat_maps = [], []
		for _ in range(M):

			j         = np.random.choice(range(len(head_counts)))
			file_name = file_names[j]
			imgs      = [imread("{0}/{1}/{2}.jpg".format(DATA_PATH, tt, file_name))]

			heat_map = sess.run(h_map, feed_dict={
				images: imgs,
				keep_prob: 1.,
				augment: False,
				training: False
				})

			J.append(j)
			heat_maps.append(heat_map)

		return head_counts[J], heat_maps



if __name__ == '__main__':

	M = 500

	print "Sampling..."
	head_counts, heat_maps = sample_heat_maps("test", M)

	# print "Optimizing..."
	# params = head_counter_optimize(heat_maps, head_counts,
	# 	(30, (8.0587801716068483, 1.5), (0.84875240583571387, 0.039), (72, 40)))

	params = (30, (8.0587801716068483, 1.5), (0.84875240583571387, 0.039), (72, 40))

	print "R2: ", score(heat_maps, head_counts, params) # R2:  0.226591250172

	
