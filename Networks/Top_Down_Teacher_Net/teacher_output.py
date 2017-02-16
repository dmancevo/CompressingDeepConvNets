import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


N_avg = 12

if __name__ == '__main__':

	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
		train_labels, train_crops = pkl.load(f)

	train_labels, train_crops = train_labels[:16300], train_crops[:16300]

	N_train = len(train_labels)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, logits, prob, loss, train_step =\
			layers

		for n in range(N_avg):

			temp  = np.empty(shape=(0,2))
			for I in np.array_split(range(N_train),  163):

				temp = np.concatenate((
					temp, sess.run(logits, feed_dict={
					labels: train_labels[I],
					images: train_crops[I],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			if n == 0:
				lgts = temp
			else:
				lgts = (n*lgts + temp)/(n+1.)


	with open("notebooks/Data/top_down_view/teacher_logits.pkl","wb") as f:
		pkl.dump(lgts, f)