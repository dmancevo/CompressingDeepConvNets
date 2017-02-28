import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score
from scipy.ndimage.filters import maximum_filter as mf

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

SAVE_OUTPUT = True
N_avg       = 8
DEPTH       = 9

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

NET_PATH  = "/notebooks/Networks/"+"Top_Down_Teacher_Net/saved/"
DATA_PATH = "/notebooks/Data/top_down_view/images/"

if __name__ == '__main__':

	with open("{0}/meta.pkl".format(DATA_PATH), "rb") as f:
		meta = pkl.load(f)

	N_train, N_test = len(meta["train"]), len(meta["test"])

	tst_file_names = np.array(meta["test"].keys())
	tst_file_names = tst_file_names[:630]
	tst_N          = [len(meta["test"][fn]) for fn in tst_file_names]
	N_test         = len(tst_file_names)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph(NET_PATH+"top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint(NET_PATH))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, c5, logits, prob, train_step = layers

		prob = tf.nn.softmax(logits, name="prob")

		for n in range(N_avg):

			tst_temp = np.empty(shape=(0,9,11,2))
			for J in np.array_split(range(N_test),  21):

				imgs = [imread("{0}/test/{1}.jpg".format(DATA_PATH, file_name)) \
				for file_name in tst_file_names[J]]

				tst_temp = np.concatenate((
					tst_temp, sess.run(prob, feed_dict={
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			if n == 0:
				y_hat = tst_temp
			else:
				y_hat = (n*y_hat + tst_temp)/(n+1.)

			count_hat = np.array([len(np.unique(block[np.logical_and(block==mf(block,size=81),block>0.95)])) \
				for block in y_hat[:,:,:,1]])

			# M = np.array([np.logical_and(block==mf(block,size=81),block>0.95) \
			# 	for block in y_hat[:,:,:,1]])

			err = 1-accuracy_score(
				tst_N,
				count_hat
			)
			precision, recall, f1, support = precision_recall_fscore_support(
				tst_N,
				count_hat
			)
			r2 = r2_score(
				tst_N,
				count_hat
			)

			precision, recall, f1 = precision[1], recall[1], f1[1]

			print "Tests: ", n
			print "Err: ", err
			print "Precision: ", precision
			print "Recall: ", recall
			print "F1 Score: ", f1
			print "R2: ", r2

			break		