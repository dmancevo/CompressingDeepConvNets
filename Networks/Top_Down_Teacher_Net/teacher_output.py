import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.ndimage.filters import maximum_filter


SAVE_OUTPUT = True
N_avg       = 8
DEPTH       = 9

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

DATA_PATH = "/notebooks/Data/top_down_view"


if __name__ == '__main__':


	if SAVE_OUTPUT:

		with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
			train_labels, train_crops = pkl.load(f)

		train_labels, train_crops = train_labels[:16300], train_crops[:16300]

		N_train = len(train_labels)

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_test = len(test_labels)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, logits, prob, train_step = layers

		for n in range(N_avg):

			if SAVE_OUTPUT:

				trn_temp  = np.empty(shape=(0,1,1,2))
				for I in np.array_split(range(N_train),  163):

					trn_temp = np.concatenate((
						trn_temp, sess.run(logits, feed_dict={
						labels: train_labels[I],
						images: train_crops[I],
						keep_prob: [1. for i in range(DEPTH)],
						augment: False,
					})))

				if n == 0:
					lgts = trn_temp
				else:
					lgts = (n*lgts + trn_temp)/(n+1.)

			tst_temp = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):

				tst_temp = np.concatenate((
					tst_temp, sess.run(prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			if n == 0:
				y_hat = tst_temp
			else:
				y_hat = (n*y_hat + tst_temp)/(n+1.)


			err = 1-accuracy_score(
				test_labels,
				np.argmax(y_hat,axis=1)
			)
			precision, recall, f1, support = precision_recall_fscore_support(
				test_labels,
				np.argmax(y_hat,axis=1)
			)

			precision, recall, f1 = precision[1], recall[1], f1[1]

			print "Tests: ", n
			print "Crp Err: ", err
			print "Crp Precision: ", precision
			print "Crp Recall: ", recall
			print "Crp F1 Score: ", f1

		if SAVE_OUTPUT:
			with open(DATA_PATH+"/teacher_logits.pkl","wb") as f:
				pkl.dump(lgts, f)
