import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score

EPOCHS   = 50
FMP      = np.power(2,1./3.)
DEPTH    = 9
CHANNELS = 30
MAX_CHAN = DEPTH*CHANNELS
DATA_PATH= "/notebooks/Data/top_down_view"

TEMP = 100.
BETA = .15

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
		tf.zeros(current.get_shape()[-1]),
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
	Build graph.
	'''
	row_images = tf.placeholder(tf.float32, shape=(None, 3072), name="row_images")
	labels     = tf.placeholder(dtype=tf.int32, name="labels")
	t_logits   = tf.placeholder(dtype=tf.float32, shape=(None, 10))
	keep_prob  = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))
	images     = tf.transpose(tf.reshape(row_images, (-1, 3, 32, 32)), perm=[0,2,3,1])
	augment    = tf.placeholder(dtype=tf.bool, name="augment")

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
		chan_in = i*CHANNELS

		if i==4:
			c4 = current

	current = conv(current, 2, 2, MAX_CHAN, MAX_CHAN, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current = batch_norm(current)
	current = conv(current, 1, 1, MAX_CHAN, 10)

	logits = current[:,0,0,:]
	prob   = tf.nn.softmax(logits, name="prob")

	loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits, labels=labels))

	train_step = tf.train.AdamOptimizer().minimize(loss)

	kd_loss = BETA*loss+(1.-BETA)*tf.pow(TEMP,2.)*\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=TEMP*logits, labels=tf.nn.softmax(TEMP*t_logits)))

	kd_train_step = tf.train.AdamOptimizer().minimize(kd_loss)

	return row_images, labels, t_logits, augment, keep_prob, c4, logits, prob,\
	kd_train_step, train_step


if __name__ == '__main__':

	with tf.Session() as sess:

		try:
			saver = tf.train.import_meta_graph("saved/fmp_net.meta")
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

		row_images, labels, t_logits, augment, keep_prob, c4, logits, prob,\
		kd_train_step, train_step = layers

		with open("/notebooks/Data/cifar10/test_batch","rb") as f:
			test_data = pkl.load(f)

		test_data["data"]   = np.array(test_data["data"])
		test_data["labels"] = np.array(test_data["labels"], dtype=int)

		N_test = len(test_data["labels"])

		with open("/notebooks/Data/cifar10/vgg16_logits.pkl", "rb") as f:
			vgg16_logits = pkl.load(f)

		min_err = 0.15
		for epoch in range(EPOCHS):

			for i in range(1,6):

				with open("/notebooks/Data/cifar10/data_batch_{0}".format(i),"rb") as f:
					train_data = pkl.load(f)


				train_data["data"]   = np.array(train_data["data"])
				train_data["labels"] = np.array(train_data["labels"], dtype=int)

				N_train = len(train_data["labels"])

				for I in np.array_split(range(N_train),  100):

					if min_err<.15:
						sess.run(train_step, feed_dict={
							row_images: train_data["data"][I],
							labels: train_data["labels"][I],
							augment: True,
							keep_prob:  [1., .9, .8, .7, .6, .5, .5, .5, .5],
						})
					else:
						sess.run(kd_train_step, feed_dict={
							row_images: train_data["data"][I],
							labels: train_data["labels"][I],
							t_logits: vgg16_logits[I],
							augment: True,
							keep_prob:  [1., .9, .8, .7, .6, .5, .5, .5, .5],
						})

			y_hat = np.empty(shape=(0,10))
			for J in np.array_split(range(N_test), 100):

				test_rows   = test_data["data"][J]
				test_labels = test_data["labels"][J]

				y_hat = np.concatenate((
					y_hat, sess.run(prob, feed_dict={
					row_images: test_rows,
					labels: test_labels,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			err = 1-accuracy_score(test_data["labels"],np.argmax(y_hat,axis=1))
			print "epoch: ",epoch+1
			print "Err: ", err
			if err<min_err:
				print "Saving..."
				min_err=err
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/fmp_net")
