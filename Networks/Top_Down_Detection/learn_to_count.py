import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.ndimage.filters import maximum_filter as mf

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True


EPOCHS     = 1
MINI_BATCH = 5

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320


FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1/\
know_dist_T10.0_beta0.05/"

DATA_PATH = "/notebooks/Data/top_down_view/images/"


def conv(current, height, width, chan_in, chan_out, padding="SAME"):
	'''
	Convolutional layer.
	'''
	W       = tf.Variable(
			tf.random_normal(
				shape=(height,width,chan_in,chan_out),
				mean=0.,
				stddev=0.01),
			dtype=tf.float32
		)
	current =tf.nn.conv2d(
			input  =current,
			filter =W,
			strides=(1,1,1,1),
			padding=padding,
		)

	return current


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

		heat_map = prob[:,:,:,1:]
		current  = heat_map

		for _ in range(3):
			current = conv(current, 60, 80, 1, 1, padding="VALID")

		current = tf.nn.relu(current)
		count   = tf.constant(np.mean(trn_N), dtype=tf.float32)+\
		conv(current, 4, 4, 1, 1, padding="VALID")[:,0,0,0]

		target      = tf.placeholder(dtype=tf.float32)
		loss        = tf.reduce_mean(tf.nn.l2_loss(count-target))
		train_step2 = tf.train.AdamOptimizer(name="adamame").minimize(loss)


		init_op    = tf.global_variables_initializer()
		sess.run(init_op)

		for epoch in range(EPOCHS):
			for _ in range(10):

				I         = np.random.choice(range(len(trn_N)), size=MINI_BATCH)
				imgs      = [imread("{0}/train/{1}.jpg".format(DATA_PATH, trn_file_names[i])) \
					for i in I]

				sess.run(train_step2, feed_dict={
					target: trn_N[I],
					images: imgs,
					keep_prob: 1.,
					augment: False,
					training: False
					})


			N = 50
			y_hat = []
			I = []
			for _ in range(N):

				i         = np.random.choice(range(len(tst_N)))
				file_name = tst_file_names[i]
				imgs      = [imread("{0}/test/{1}.jpg".format(DATA_PATH, file_name))]

				y_hat.append(sess.run(count, feed_dict={
					images: imgs,
					keep_prob: 1.,
					augment: False,
					training: False
					}))

				I.append(i)


			print "R2: ", r2_score(tst_N[I], y_hat)
			print "Test Mean: ", np.mean(tst_N)
			print "Y hat mean: ", np.mean(y_hat)

