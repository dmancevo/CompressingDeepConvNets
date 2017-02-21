import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

MIN_ERR    = 0.046

EPOCHS     = 1
MINI_BATCH = 100
FMP        = np.sqrt(2)
DEPTH      = 8
CHANNELS   = 30
PXL_CHAN   = 8

TOTAL_EPOCHS = 0
SAVED_EPOCHS = TOTAL_EPOCHS

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

def graph_conv(training, images, augment, keep_prob):
	'''
	Build convolution-fractional max pooling portion of graph.
	'''

	# Data Augmentation
	current = tf.cond(augment, lambda: data_aug(images), lambda: images)

	chan_in, chan_out = 3, CHANNELS
	current = tf.layers.batch_normalization(
		current,
		axis=-1,
		training=training
	)
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
		current = tf.layers.batch_normalization(
			current,
			axis=-1,
			training=training
		)
		chan_in  = i*CHANNELS

	return current

def graph_crop_class(training, labels, keep_prob, c8):
	'''
	Crop classification portion of the graph.
	'''
	current = conv(c8, 2, 4, 240, 240, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])

	current = tf.layers.batch_normalization(
		current,
		axis=-1,
		training=training
	)
	current = conv(current, 1, 1, 240, 2)

	crop_log   = current[:,0,0,:]
	crop_prob  = tf.nn.softmax(crop_log, name="crop_prob")
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=crop_log, labels=labels), name="loss")

	return crop_log, crop_prob, loss

def graph_pxl_class(training, pxl_labels, keep_prob, c8):
	'''
	Segmentation portion of the graph.
	'''

	fltr = (18,17,PXL_CHAN, 240)

	batch_size = tf.shape(c8)[0]
	W = tf.Variable(
		np.random.normal(0,0.01,size=fltr),
		dtype=tf.float32
	)
	current = tf.nn.conv2d_transpose(
		value=c8,
		filter=W,
		output_shape=tf.stack((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, fltr[2])),
		strides=(1,fltr[0],fltr[1],1),
		padding="VALID"
	) 

	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current.set_shape((None,IMAGE_HEIGHT,IMAGE_WIDTH,PXL_CHAN))
	current = tf.layers.batch_normalization(
		current,
		axis=-1,
		training=training
	)
	current = conv(current, 1, 1, PXL_CHAN, 2, padding="VALID")

	pxl_log      = current
	pxl_prob     = tf.nn.softmax(pxl_log, name="pxl_prob")
	f_pxl_log    = tf.reshape(pxl_log, shape=(-1,2))
	f_pxl_labels = tf.reshape(pxl_labels, shape=(-1,))
	pxl_loss     = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=f_pxl_log, labels=f_pxl_labels), name="pxl_loss")

	return pxl_log, pxl_prob, pxl_loss

def graph():
	'''
	Assemble the graph.
	'''

	# Placeholders
	training = tf.placeholder(dtype=tf.bool, name="training")
	crp_labels  = tf.placeholder(dtype=tf.int32, shape=(None,), name="crop_labels")
	pxl_labels  = tf.placeholder(dtype=tf.int32,
		   shape=(None, IMAGE_HEIGHT,IMAGE_WIDTH), name="pixel_labels")
	images      = tf.placeholder(dtype=tf.float32,
		   shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))
	augment = tf.placeholder(dtype=tf.bool, name="augment")

	# Convolution / Fractional Max Pooling
	c8 = graph_conv(training, images, augment, keep_prob)

	is_crops = tf.placeholder(dtype=tf.bool, name="is_crops")

	# Process crop or image
	logits, prob, loss=\
	tf.cond(
		is_crops,
		lambda: graph_crop_class(training, crp_labels, keep_prob, c8),
		lambda: graph_pxl_class(training, pxl_labels, keep_prob, c8)
	)

	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return training, crp_labels, pxl_labels, images, keep_prob, augment, c8,\
	is_crops, logits, prob, train_step

def test(trn, aug, crop):
	'''
	Test with generated data.
	'''

	crp_labs = np.ones(shape=(100,), dtype=int)
	pxl_labs = np.ones(shape=(2,240,320), dtype=int)
	imgs     = np.random.uniform(size=(2,240,320,3))

	with tf.Session() as sess:

		training, crp_labels, pxl_labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = graph()

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		print sess.run(prob, feed_dict={
			training: trn,
			crp_labels: crp_labs,
			pxl_labels: pxl_labs,
			images: imgs,
			keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
			augment: aug,
			is_crops: crop,
		}).shape



if __name__ == '__main__':

	test(trn=True, aug=True, crop=False)


	# with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
	# 	train_labels, train_crops = pkl.load(f)

	# with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
	# 	test_labels, test_crops = pkl.load(f)


	# train_labels, train_crops = train_labels[:16300], train_crops[:16300]
	# test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	# N_train, N_test = len(train_labels), len(test_labels)


	# with tf.Session() as sess:

	# 	try:
	# 		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
	# 		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
	# 		layers = tf.get_collection('layers')

	# 		print "Successfully loaded graph from file."

	# 	except IOError:

	# 		print "Building graph from scratch..."

	# 		layers = graph()
	# 		for layer in layers:
	# 			tf.add_to_collection('layers', layer)

	# 		init_op    = tf.global_variables_initializer()
	# 		sess.run(init_op)

	# training, crp_labels, pxl_labels, images, keep_prob, augment, c8,\
	# is_crops, logits, prob, train_step = layers

	# 	pxl_labs = np.ones(shape=(1,240,320), dtype=int)

	# 	print "Crop Class Training..."
	# 	for epoch in range(EPOCHS):
	# 		for __ in range(N_train/MINI_BATCH):

	# 			I = np.random.choice(range(N_train), size=100, replace=False)
	# 			sess.run(train_step, feed_dict={
	# 				training: True,
	# 				crp_labels: train_labels[I],
	# 				pxl_labels: pxl_labs,
	# 				images: train_crops[I],
	# 				keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
	# 				augment: True,
	# 				is_crops: True,
	# 			})

	# 		y_hat = np.empty(shape=(0,2))
	# 		for J in np.array_split(range(N_test),  16):
	# 			y_hat = np.concatenate((
	# 				y_hat, sess.run(prob, feed_dict={
	# 				training: True,
	# 				crp_labels: test_labels[J],
	# 				pxl_labels: pxl_labs,
	# 				images: test_crops[J],
	# 				keep_prob: [1. for i in range(DEPTH)],
	# 				augment: False,
	# 				is_crops: True,
	# 			})))

	# 		print "Epoch: ", TOTAL_EPOCHS+epoch+1
	# 		err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
	# 		print "Err: ", err
	# 		print "F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

	# 		if err < MIN_ERR:
	# 			print "Saving..."
	# 			MIN_ERR = err
	# 			SAVED_EPOCHS = TOTAL_EPOCHS+epoch+1
	# 			new_saver = tf.train.Saver(max_to_keep=2)
	# 			new_saver.save(sess, "saved/top_down_net")

	# 	print "SAVED EPOCHS: ", SAVED_EPOCHS
	# 	print "MIN ERR: ", MIN_ERR
	# 	print "Crop Class Training Done!"