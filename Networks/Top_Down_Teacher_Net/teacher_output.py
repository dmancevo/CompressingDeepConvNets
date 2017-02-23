import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from scipy.ndimage.filters import maximum_filter

from top_down_net import pxl_img_lab


CRPS = True
PXLS  = False

SAVE_OUTPUT = False

N_avg = 8

DEPTH = 8

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

HEIGHT_OFFSETS = [30*i for i in range(16)]
WIDTH_OFFSETS  = []

DATA_PATH = "/notebooks/Data/top_down_view"


if __name__ == '__main__':

	if CRPS:

		with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
			test_labels, test_crops = pkl.load(f)

		test_labels, test_crops = test_labels[:4800], test_crops[:4800]

		N_test = len(test_labels)

	if PXLS:

		with open(DATA_PATH+"/images/meta.pkl", "rb") as f:
			meta = pkl.load(f)

		trn_str_queue = ["{0}/{1}/{2}.jpg".format(DATA_PATH, "images/train", img) \
		for img in meta["train"].keys()]

		tst_str_queue = ["{0}/{1}/{2}.jpg".format(DATA_PATH, "images/test", img) \
		for img in meta["test"].keys()]

		trn_file_queue = tf.train.string_input_producer(trn_str_queue, shuffle=True)
		tst_file_queue = tf.train.string_input_producer(tst_str_queue, shuffle=True)

		image_reader = tf.WholeFileReader()

		trn_file, trn_image_file = image_reader.read(trn_file_queue)
		tst_file, tst_image_file = image_reader.read(tst_file_queue)

		trn_image = tf.image.decode_jpeg(trn_image_file)
		tst_image = tf.image.decode_jpeg(tst_image_file)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = layers


		if CRPS:

			for n in range(N_avg):

				if SAVE_OUTPUT:

					trn_temp  = np.empty(shape=(0,2))
					for I in np.array_split(range(N_train),  163):

						temp = np.concatenate((
							temp, sess.run(logits, feed_dict={
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
						is_crops: True
					})))

				if n == 0:
					y_hat = tst_temp
				else:
					y_hat = (n*y_hat + tst_temp)/(n+1.)

				print "Crp Err: ", 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
				print "Crp Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
				print "Crp Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
				print "Crp F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

			if SAVE_OUTPUT:
				with open("notebooks/Data/top_down_view/crp_teacher_logits.pkl","wb") as f:
					pkl.dump(lgts, f)

		if PXLS:

			test_labels, imgs = [], []
			for __ in range(len(meta["test"])):
				pxl_labs, imgs = pxl_img_lab(tst_file, tst_image)
				test_labels   += pxl_lab
				imgs          +=imgs

			for n in range(N_avg):

				temp = np.empty(shape=(0,2))
				for j in range(test_labels.shape[0]):
					imgs, pxl_labs = imgs[j], test_labels[j]
					y_hat = np.concatenate((
						y_hat, np.reshape(sess.run(prob, feed_dict={
						labels: pxl_labs,
						images: imgs,
						keep_prob: [1. for i in range(DEPTH)],
						augment: False,
						is_crops: False,
					}), newshape=(-1,2))))

				if n == 0:
					y_hat = temp
				else:
					y_hat = (n*y_hat + temp)/(n+1.)

				print "No. Tests: ", n+1
				print "Pxl Err: ", 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl F1 Score: ", curr_f1







