import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import r2_score

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1/\
know_dist_T10.0_beta0.05/"

DATA_PATH = "/notebooks/Data/top_down_view/images/"

if __name__ =='__main__':

	N, M = 500, 300

	with open("{0}/meta.pkl".format(DATA_PATH), "rb") as f:
		meta = pkl.load(f)

	file_names_train  = np.array(meta["train"].keys())
	head_counts_train = np.array([len(meta["train"][fn]) for fn in file_names_train])

	file_names_test   = np.array(meta["test"].keys())
	head_counts_test  = np.array([len(meta["test"][fn]) for fn in file_names_test])

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("{0}student_1.meta".format(FOLDER))
		saver.restore(sess, tf.train.latest_checkpoint(FOLDER))
		layers = tf.get_collection('layers')

		labels, t_logits, images, keep_prob, training, augment,\
		prob, f_prob, train_step = layers

		with tf.variable_score("area_to_counts"):

			th = tf.get_variable("th", (1,),
        	initializer=tf.random_normal_initializer(.8,0.01))

			alpha = tf.get_variable("alpha", (1,),
        	initializer=tf.random_normal_initializer(0,0.01))

        	beta = tf.get_variable("beta", (1,),
        	initializer=tf.random_normal_initializer(0,0.01))

		pxl_counts  = tf.reduce_sum(tf.to_float32(tf.greater(prob[0,:,:,1], th)))
		hat_counts  = alpha*pxl_counts + beta
		h_counts    = tf.placeholder(dtype=tf.float32)

		train_vars  = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES,
			"area_to_counts"
		)

		loss2       = tf.nn.l2_loss(hat_counts-h_counts)
		train_step2 = tf.train.AdamOptimizer(name="adamame").minimize(
			loss,
			var_list=train_vars
		)


		for _ in range(N):

			I = np.random.choice(range(len(head_counts_train)),
				size=10,
				replace=False
			)

			imgs      = [imread("{0}/{1}/{2}.jpg".format(DATA_PATH, "train", fn)) \
			for fn in file_names_train[I]]

			heat_map = sess.run(train_step2, feed_dict={
				images: imgs,
				keep_prob: .5,
				augment: False,
				training: False
			})

		hat = []
		for _ in range(M):

			J = np.random.choice(range(len(head_counts_test)),
				size=10,
				replace=False
			)

			imgs      = [imread("{0}/{1}/{2}.jpg".format(DATA_PATH, "test", fn)) \
			for fn in file_names_test[J]]

			hat.append(sess.run(hat_counts, feed_dict={
				images: imgs,
				keep_prob: 1.,
				augment: False,
				training: False
			}))

	print "R2: score", r2_score(head_counts_test, hat)