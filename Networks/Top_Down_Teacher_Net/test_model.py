import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from scipy.ndimage.filters import maximum_filter


N_avg = 8

DEPTH = 8

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

HEIGHT_OFFSETS = [30*i for i in range(16)]
WIDTH_OFFSETS  = []

DATA_PATH               = "/notebooks/Data/top_down_view"


if __name__ == '__main__':

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_test = len(test_labels)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, logits, prob, loss, train_step =\
			layers

		for n in range(N_avg):

			temp = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):

				temp = np.concatenate((
					temp, sess.run(prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			if n == 0:
				y_hat = temp
			else:
				y_hat = (n*y_hat + temp)/(n+1.)

			print "No. Tests: ", n+1
			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Err: ", err
			print "F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))






