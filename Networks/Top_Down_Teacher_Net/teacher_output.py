import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

def save_logits(file_name, lgts):
	'''
	Save logits in the right folder.
	'''
	with open("{0}/teacher/pxl/{1}.pkl".format(DATA_PATH, file_name)) as f:
		pkl.dump(lgts)


if __name__ == '__main__':

	if CRPS:

		if SAVE_OUTPUT:

			with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
				train_labels, train_crops = pkl.load(f)

			train_labels, train_crops = train_labels[:16300], train_crops[:16300]

			N_train = len(train_labels)

		with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
			test_labels, test_crops = pkl.load(f)

		test_labels, test_crops = test_labels[:4800], test_crops[:4800]

		N_test = len(test_labels)

	if PXLS:

		with open(DATA_PATH+"/images/meta.pkl", "rb") as f:
			meta = pkl.load(f)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, temp, c8,\
		is_crops, logits, f_labels, prob, train_step = layers


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
						is_crops: True,
						temp: 10.0
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
				with open("notebooks/Data/top_down_view/teacher/crp_teacher_logits.pkl","wb") as f:
					pkl.dump(lgts, f)

		if PXLS:

			trn_img_names = meta["traing"].keys()
			tst_img_names = meta["test"].keys()

			test_labels = np.empty(shape=(0,2))
			for img_name in tst_img_names:
				pxl_labs, imgs = pxl_img_lab("test", image_name=img_name)
				test_labels = np.concatenate((
					test_labels, np.reshape(sess.run(f_labels, feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: False,
					temp: 10.0
				}), newshape=(-1,2))))


			for n in range(N_avg):

				if SAVE_OUTPUT:

					trn_temp  = np.empty(shape=(0,240, 360, 2))
					for img_name in trn_img_names:
						pxl_labs, imgs = pxl_img_lab("train", image_name=img_name)
						trn_temp = np.concatenate((
							trn_temp, sess.run(logits, feed_dict={
							labels: pxl_labs,
							images: imgs,
							keep_prob: [1. for i in range(DEPTH)],
							augment: False,
							is_crops: False,
							temp: 10.0
						})))

					if n == 0:
						lgts = trn_temp
					else:
						lgts = (n*lgts + trn_temp)/(n+1.)

				temp, = np.empty(shape=(0,2))
				for img_name in tst_img_names:
					pxl_labs, imgs = pxl_img_lab(tt, image_name=img_name)
					temp = np.concatenate((
						temp, np.reshape(sess.run(prob, feed_dict={
						labels: pxl_labs,
						images: imgs,
						keep_prob: [1. for i in range(DEPTH)],
						augment: False,
						is_crops: False,
						temp: 10.0
					}), newshape=(-1,2))))

				if n == 0:
					y_hat = temp
				else:
					y_hat = (n*y_hat + temp)/(n+1.)

				err = 1-accuracy_score(
					np.argmax(test_labels, axis=1),
					np.argmax(y_hat,axis=1)
				)
				precision, recall, f1, support = precision_recall_fscore_support(
					np.argmax(test_labels, axis=1),
					np.argmax(y_hat,axis=1)
				)

				precision, recall, f1 = precision[1], recall[1], f1[1]

				print "No. Tests: ", n+1
				print "Pxl Err: ", err
				print "Pxl Precision: ", precision
				print "Pxl Recall: ", recall
				print "Pxl F1 Score: ", f1


			if SAVE_OUTPUT:
				for i in range(len(trn_img_names)):
					img_name = trn_img_names[i]
					save_logits(img_name, lgts[i,:,:,:])







