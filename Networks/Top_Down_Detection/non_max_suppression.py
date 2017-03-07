import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import r2_score
# from scipy.ndimage.filters import maximum_filter as mf

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.linear_model import ElasticNet

SAVE_OUTPUT = True

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

K = 100

FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1/\
know_dist_T10.0_beta0.05/"

DATA_PATH = "/notebooks/Data/top_down_view/images/"


if __name__ == '__main__':

	with open("{0}/meta.pkl".format(DATA_PATH), "rb") as f:
		meta = pkl.load(f)

	N_train, N_test = len(meta["train"]), len(meta["test"])

	trn_file_names = np.array(meta["train"].keys())
	trn_N          = np.array([len(meta["train"][fn]) for fn in trn_file_names])

	tst_file_names = np.array(meta["test"].keys())
	tst_N          = np.array([len(meta["test"][fn]) for fn in tst_file_names])


	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("{0}student_1.meta".format(FOLDER))
		saver.restore(sess, tf.train.latest_checkpoint(FOLDER))
		layers = tf.get_collection('layers')

		labels, t_logits, images, keep_prob, training, augment,\
		prob, f_prob, train_step = layers

		# Bounding boxes and scores
		f_prob         = tf.reshape(prob[:,:,:,1], shape=(-1, 181*241))
		scores, coords = tf.nn.top_k(f_prob, k=K)
		scores         = tf.reshape(scores, shape=(-1,))
		max_pxls       = tf.reshape(tf.to_int32(coords), shape=(-1,))
		boxes          = tf.transpose(tf.stack((
			tf.maximum(0, tf.floordiv(max_pxls, 241)-CROP_HEIGHT/2),
			tf.maximum(0, tf.mod(max_pxls, 241)-CROP_WIDTH/2),
			tf.minimum(180, tf.floordiv(max_pxls, 241)+CROP_HEIGHT/2),
			tf.minimum(240, tf.mod(max_pxls, 241)+CROP_WIDTH/2),
		)))

		# Non-Maximum Supression
		nmx = tf.image.non_max_suppression(
			tf.to_float(boxes),
			scores,
			max_output_size=10
		)

		N = 200
		x_hat = []
		I = []
		for _ in range(N):

			i         = np.random.choice(range(len(trn_N)))
			file_name = trn_file_names[i]
			imgs      = [imread("{0}/train/{1}.jpg".format(DATA_PATH, file_name))]

			x_hat.append(len(sess.run(nmx, feed_dict={
				images: imgs,
				keep_prob: 1.,
				augment: False,
				training: False
				})))

			I.append(i)


		x_hat = np.reshape(x_hat, newshape=(N,1))
		target = np.reshape(trn_N[I], newshape=(N,1))
		enet = ElasticNet(alpha=0.00001, l1_ratio=0.15)
		enet.fit(x_hat, trn_N[I])

		M = 50
		y_hat = []
		J = []
		for _ in range(M):

			j         = np.random.choice(range(len(tst_N)))
			file_name = tst_file_names[j]
			imgs      = [imread("{0}/test/{1}.jpg".format(DATA_PATH, file_name))]

			y_hat.append(len(sess.run(nmx, feed_dict={
				images: imgs,
				keep_prob: 1.,
				augment: False,
				training: False
				})))

			J.append(j)


	y_hat = np.reshape(y_hat, newshape=(M,1))
	print "R2: ", r2_score(tst_N[J], np.round(enet.predict(y_hat)))
	print "Mean: ", np.mean(tst_N)
	print "x_hat mean: ", np.mean(x_hat)

	# with open("h_map.pkl", "wb") as f:
	# 	pkl.dump((counts, est, h_map[:,:,:,1]), f)