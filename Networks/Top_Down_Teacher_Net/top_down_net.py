import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


### An alternative to top_down_net where batch normalization is always
### performed using the mini-batch statistics.

# Network is trained on crops only

CRP_EPOCHS     = 50
CRP_MINI_BATCH = 100

FMP      = np.sqrt(2)
DEPTH    = 9
CHANNELS = 30
MAX_CHAN = DEPTH*CHANNELS

DATA_PATH                 = "/notebooks/Data/top_down_view"
IMAGE_HEIGHT, IMAGE_WIDTH = 240, 320
CROP_HEIGHT, CROP_WIDTH   = 60, 80

def data_aug(images):
	images = tf.map_fn(
		lambda img: tf.image.random_flip_left_right(img), images)
	images = tf.map_fn(
		lambda img: tf.image.random_flip_up_down(img), images)
	images = tf.map_fn(
		lambda img: tf.image.random_brightness(img, max_delta=63), images)
	images = tf.map_fn(
		lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), images)

	return images

def conv(current, height, width, chan_in, chan_out, padding="SAME"):
	'''
	Convolutional layer.
	'''
	W       = tf.Variable(
			np.random.normal(0,0.01,size=(height,width,chan_in,chan_out)),
			dtype=tf.float32
		)
	current =tf.nn.conv2d(
			input  =current,
			filter =W,
			strides=(1,1,1,1),
			padding=padding,
		)

	return current

def leaky_relu(current):
	'''
	Leaky ReLU activation with learnable parameter a.
	'''
	a = tf.Variable(np.random.uniform(0.07,0.13), dtype=tf.float32)
	return tf.maximum(current, a*current)

def batch_norm(current):
	'''
	Batch Normalization performed exclusively using mini-batch statistics.
	'''
	beta = tf.Variable(
		np.random.normal(0,0.01, current.get_shape()[-1]),
		dtype=tf.float32
	)
	bn_mean, bn_var = tf.nn.moments(current, axes=[0,1,2])
	current = tf.nn.batch_normalization(
		x=current,
		mean=bn_mean,
		variance=bn_var,
		offset=beta,
		scale=None,
		variance_epsilon=0.001
	)

	return current

def graph():
	'''
	Build convolution-fractional max pooling portion of graph.
	'''

	labels      = tf.placeholder(dtype=tf.int32, shape=(None,))
	images      = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))
	augment     = tf.placeholder(dtype=tf.bool, name="augment")

	# Data Augmentation
	current = tf.cond(augment, lambda: data_aug(images), lambda: images)

	chan_in, chan_out = 3, CHANNELS
	current = batch_norm(current)
	for i in range(1,DEPTH+1):
		chan_out = i*CHANNELS
		current  = conv(current, 2, 2, chan_in, chan_out)
		current  = leaky_relu(current)
		current  = tf.nn.dropout(current, keep_prob[i-1])
		current  = tf.nn.fractional_max_pool(
			value=current,
			pooling_ratio=(1.0, FMP, FMP, 1.0),
			pseudo_random=True,
			)[0]
		current = batch_norm(current)
		chan_in  = i*CHANNELS

	current = conv(current, 1, 2, MAX_CHAN, MAX_CHAN, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current = batch_norm(current)
	current = conv(current, 1, 1, MAX_CHAN, 2)

	logits = current
	f_log  = tf.reshape(logits, shape=(-1,2))
	prob   = tf.nn.softmax(f_log, name="crp_prob")
	loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		 logits=f_log, labels=labels), name="loss")

	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return labels, images, keep_prob, augment, logits, train_step


def test():
	'''
	Test Graph.
	'''

	labels, images, keep_prob, augment, logits, train_step = graph()

	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	labs = np.random.choice(range(1), size=100)
	imgs = np.random.normal(size=(10,240,320,3))
	# imgs = np.random.normal(size=(10,60,80,3))

	print sess.run(logits, feed_dict={
		labels: labs,
		images: imgs,
		keep_prob: [1., .9, .8, .7, .6, .5, .5, .5, .5],
		augment: True
		}).shape



if __name__ == '__main__':

	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
		train_labels, train_crops = pkl.load(f)

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)


	train_labels, train_crops = train_labels[:16300], train_crops[:16300]
	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_train, N_test = len(train_labels), len(test_labels)


	with tf.Session() as sess:

		try:
			saver = tf.train.import_meta_graph("saved/top_down_net.meta")
			saver.restore(sess, tf.train.latest_checkpoint("saved/"))
			layers = tf.get_collection('layers')

			print "Successfully loaded graph from file."

		except IOError:

			print "Building graph from scratch..."

			layers = graph()
			for layer in layers:
				tf.add_to_collection('layers', layer)

			init_op    = tf.global_variables_initializer()
			sess.run(init_op)

		labels, images, keep_prob, augment, logits, train_step = layers

		min_err = 0.05
		for epoch in range(CRP_EPOCHS):
			print "epoch: ", epoch+1
			for __ in range(N_train/CRP_MINI_BATCH):

				I = np.random.choice(range(N_train), size=100, replace=False)
				sess.run(train_step, feed_dict={
					labels: train_labels[I],
					images: train_crops[I],
					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5, .5],
					augment: True,
				})

			y_hat = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):
				y_hat = np.concatenate((
					y_hat, sess.run(prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Crp Err: ", err
			if err<min_err:
				min_err=err
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/top_down_net")

				precision, recall, f1, support = precision_recall_fscore_support(
					test_labels,
					np.argmax(y_hat,axis=1)
					)

				precision, recall, f1 = precision[1], recall[1], f1[1]

				print "Crp Precision: ", precision
				print "Crp Recall: ", recall
				print "Crp F1 Score: ", f1

	print "Done!"