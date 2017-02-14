import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

ARCH       = "arch1"
EPOCHS     = 1
MINI_BATCH = 100
FMP        = 1.414
DEPTH      = 8
CHANNELS   = 15

DATA_PATH               = "/notebooks/Data/top_down_view"
CROP_HEIGHT, CROP_WIDTH = 60, 80


def batch_norm(layer, dims):
	'''
	Batch normalization.
	'''
	b_mean, b_var = tf.nn.moments(layer,[0])
	scale         = tf.Variable(tf.ones(dims))
	beta          = tf.Variable(tf.zeros(dims))
	current       = tf.nn.batch_normalization(
		x                = layer,
		mean             = b_mean,
		variance         = b_var,
		offset           = beta,
		scale            = scale,
		variance_epsilon = 1e-3)

	return current

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

def graph():
	'''
	Build Graph.
	'''

	# Load the data...
	labels       = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
	images       = tf.placeholder(dtype=tf.float32,
		shape=(None,CROP_HEIGHT,CROP_WIDTH,3), name="images")
	keep_prob    = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))

	# Data Augmentation
	augment = tf.placeholder(tf.bool, name="augment")
	current = tf.cond(augment, lambda: data_aug(images), lambda: images)
	
	height, width = CROP_HEIGHT,CROP_WIDTH
	chan_in, chan_out = 3, CHANNELS
	for i in range(1,DEPTH+1):
		chan_out = i*CHANNELS
		current  = batch_norm(current, (height,width,chan_in))
		current  = conv(current, 2, 2, chan_in, chan_out)
		current  = leaky_relu(current)
		curent   = tf.nn.dropout(current, keep_prob[i-1])
		current  = tf.nn.fractional_max_pool(
			value=current,
			pooling_ratio=(1.0, FMP, FMP, 1.0),
			pseudo_random=True,
			)[0]
		chan_in       = i*CHANNELS
		height, width = int(height/FMP), int(width/FMP)

	current = batch_norm(current, (height,width,chan_in))
	current = conv(current, height, width, chan_in, chan_in, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])

	current = batch_norm(current, (1,1,chan_in))
	current = conv(current, 1, 1, chan_in, 2)

	logits     = current[:,0,0,:]
	prob       = tf.nn.softmax(logits, name="prob")
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels), name="loss")
	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return labels, images, keep_prob, augment, logits, prob, loss, train_step


if __name__ == '__main__':

	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
		train_labels, train_crops = pkl.load(f)

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_train, N_test = len(train_labels), len(test_labels)

	with tf.Session() as sess:

		try:
			saver = tf.train.import_meta_graph("saved/{0}.meta".format(ARCH))
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

		labels, images, keep_prob, augment, logits, prob, loss, train_step =\
			layers

		print "Training..."
		for epoch in range(EPOCHS):
			for __ in range(N_train/MINI_BATCH):

				I = np.random.choice(range(N_train), size=100, replace=False)
				sess.run(train_step, feed_dict={
					labels: train_labels[I],
					images: train_crops[I],
					keep_prob: [0., .1, .2, .3, .4, .5, .5, .5],
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

			print "Epoch: ", epoch+1
			print "Err: ", 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

		new_saver = tf.train.Saver(max_to_keep=2)
		new_saver.save(sess, "saved/{0}".format(ARCH))

print "Done!"