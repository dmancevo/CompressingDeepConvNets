import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score


SAVE_OUTPUT = True
N_avg       = 12

EPOCHS   = 50
FMP      = np.power(2,1./3.)
DEPTH    = 12
CHANNELS = 50
MAX_CHAN = DEPTH*CHANNELS
DATA_PATH= "/notebooks/Data/cifar10"


if __name__ == '__main__':

	saver = tf.train.import_meta_graph("saved/fmp_net.meta")
	saver.restore(sess, tf.train.latest_checkpoint("saved/"))
	layers = tf.get_collection('layers')

	print "Successfully loaded graph from file."


	row_images, labels, augment, keep_prob, c4, logits, prob, train_step = layers

	with open("/notebooks/Data/cifar10/test_batch","rb") as f:
			test_data = pkl.load(f)

	test_data["data"]   = np.array(test_data["data"])
	test_data["labels"] = np.array(test_data["labels"], dtype=int)

	N_test = len(test_data["labels"])


	with tf.Session() as sess:

		for _ in range(N_avg):

			if SAVE_OUTPUT:

				trn_temp  = np.empty(shape=(0,1,1,10))
				trn_temp2 = np.empty(shape=(0,7,7,10))
				for i in range(1,6):

					with open("/notebooks/Data/cifar10/data_batch_{0}".format(i),"rb") as f:
						train_data = pkl.load(f)


					train_data["data"]   = np.array(train_data["data"])
					train_data["labels"] = np.array(train_data["labels"], dtype=int)

					N_train = len(train_data["labels"])

					for I in np.array_split(range(N_train),  100):
						train_rows   = train_data["data"][I]

						trn_temp = np.concatenate((
							trn_temp, sess.run(logits, feed_dict={
							row_images: train_rows,
							augment: False,
							keep_prob:  [1. for _ in range(DEPTH)],
						})))

						trn_temp2 = np.concatenate((
							trn_temp2, sess.run(c4, feed_dict={
							row_images: train_rows,
							augment: False,
							keep_prob:  [1. for _ in range(DEPTH)],
						})))

				if n == 0:
					lgts   = trn_temp
					c4_out = trn_temp2
				else:
					lgts   = (n*lgts   + trn_temp)/(n+1.)
					c4_out = (n*c4_out + trn_temp2)/(n+1.)

			tst_temp  = np.empty(shape=(0,1,1,10))
			for J in np.array_split(range(N_test), 100):

				test_rows   = test_data["data"][J]
				test_labels = test_data["labels"][J]

				trn_temp = np.concatenate((
					tst_temp, sess.run(logits, feed_dict={
					row_images: train_rows,
					augment: False,
					keep_prob:  [1. for _ in range(DEPTH)],
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

		with open(DATA_PATH+"/teacher_c5.pkl","wb") as f:
			pkl.dump(c5_out, f)

