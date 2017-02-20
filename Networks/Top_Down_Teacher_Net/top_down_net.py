import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

MIN_ERR    = 0.046

EPOCHS     = 50
MINI_BATCH = 100
FMP        = 1.62
DEPTH      = 8
CHANNELS   = 30

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

def batch_norm(layer, channels):
	'''
	Batch normalization.
	'''
	b_mean, b_var = tf.nn.moments(layer,[0,1,2])
	scale         = tf.Variable(tf.ones(channels))
	beta          = tf.Variable(tf.zeros(channels))
	current       = tf.nn.batch_normalization(
		x                = layer,
		mean             = b_mean,
		variance         = b_var,
		offset           = beta,
		scale            = scale,
		variance_epsilon = 1e-3)

	return current

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

def segment(layers):
	'''
	Upsample and classify each pixel in picture.

	cx: output from convolutional layer at depth x
	'''

def graph():
	'''
	Build Graph.
	'''

	# Load the data...
	labels      = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
	pxl_labels  = tf.placeholder(dtype=tf.int32,
		   shape=(None, IMAGE_HEIGHT,IMAGE_WIDTH))
	is_crops    = tf.placeholder(dtype=tf.bool)
	images      = tf.placeholder(dtype=tf.float32,
		   shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	crops       = tf.placeholder(dtype=tf.float32,
		   shape=(None,CROP_HEIGHT,CROP_WIDTH,3), name="crops")
	keep_prob   = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))

	# Process crop or image
	current = tf.cond(is_crops, lambda: crops, lambda: images)

	# Data Augmentation
	augment = tf.placeholder(tf.bool, name="augment")
	# current = tf.cond(augment, lambda: data_aug(current), lambda: current)
	current = images
	
	crop_height, crop_width = CROP_HEIGHT,CROP_WIDTH

	chan_in, chan_out = 3, CHANNELS
	current  = batch_norm(current, chan_in)
	for i in range(1,DEPTH+1):
		chan_out = i*CHANNELS
		current  = conv(current, 2, 2, chan_in, chan_out)
		current  = leaky_relu(current)
		curent   = tf.nn.dropout(current, keep_prob[i-1])
		current  = tf.nn.fractional_max_pool(
			value=current,
			pooling_ratio=(1.0, FMP, FMP, 1.0),
			pseudo_random=True,
			)[0]
		current  = batch_norm(current, chan_out)
		chan_in  = i*CHANNELS
		crop_height, crop_width = int(crop_height/FMP), int(crop_width/FMP)

		if i==04:
			c4 = current
		if i==8:
			c8 = current

	return labels, pxl_labels, images, is_crops, crops, keep_prob, augment, c4, c8


def graph_class(labels, keep_prob, c8):
	'''
	Crop classification portion of the graph.
	'''
	current = conv(c8, 4, 5, 240, 240, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])

	current = batch_norm(current, (1,1,240))
	current = conv(current, 1, 1, 240, 2)

	logits     = current[:,0,0,:]
	prob       = tf.nn.softmax(logits, name="prob")
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels), name="loss")
	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return  logits, prob, train_step


def graph_loc(pxl_labels, keep_prob, c4, c8):
	'''
	Segmentation portion of the graph.
	'''
	layers = ([
		(c4, (7, 7, 4, 120)),
		(c8, (60,64,8,240))
	])

	values = []
	for layer, fltr in layers:
		batch_size = tf.shape(layer)[0]

		W = tf.Variable(
			np.random.normal(0,0.01,size=fltr),
			dtype=tf.float32
		)
		current = tf.nn.conv2d_transpose(
			value=layer,
			filter=W,
			output_shape=tf.pack((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, fltr[2])),
			strides=(1,fltr[0],fltr[1],1),
			padding="VALID"
		) 

		values.append(current)

	current    = tf.concat(3, values)
	current    = batch_norm(current, 12)
	W          = tf.Variable(
		            np.random.normal(0,0.01,
		            size=(IMAGE_HEIGHT,IMAGE_WIDTH,12,2)),
		            dtype=tf.float32
		         )
	seg_log      = tf.nn.conv2d(current, W, strides=(1,1,1,1), padding="VALID")
	seg          = tf.nn.softmax(seg_log, name="seg")

	f_seg_log    = tf.reshape(seg_log, shape=(-1,2))
	f_pxl_labels = tf.reshape(pxl_labels, shape=(-1,))
	pxl_loss2    = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		f_seg_log, f_pxl_labels), name="pxl_loss")
	# train_step2 = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return seg_log, seg


if __name__ == '__main__':

	labels, sprs_labels, images, is_crops, crops, keep_prob, augment, c4, c8\
	 = graph()

	logits, prob, train_step = graph_class(labels, keep_prob, c8)

	seg_log, seg = graph_loc(sprs_labels, keep_prob, c4, c8)

# 	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
# 		train_labels, train_crops = pkl.load(f)

# 	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
# 		test_labels, test_crops = pkl.load(f)


# 	train_labels, train_crops = train_labels[:16300], train_crops[:16300]
# 	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

# 	N_train, N_test = len(train_labels), len(test_labels)

# 	with tf.Session() as sess:

# 		try:
# 			saver = tf.train.import_meta_graph("saved/top_down_net.meta")
# 			saver.restore(sess, tf.train.latest_checkpoint("saved/"))
# 			layers = tf.get_collection('layers')

# 			print "Successfully loaded graph from file."

# 		except IOError:

# 			print "Building graph from scratch..."

# 			layers = graph()
# 			for layer in layers:
# 				tf.add_to_collection('layers', layer)

# 			init_op    = tf.global_variables_initializer()
# 			sess.run(init_op)

# 		labels, crops, keep_prob, augment, logits, prob, train_step =\
# 			layers

# 		print "Training..."
# 		for epoch in range(EPOCHS):
# 			for __ in range(N_train/MINI_BATCH):

# 				I = np.random.choice(range(N_train), size=100, replace=False)
# 				sess.run(train_step, feed_dict={
# 					labels: train_labels[I],
# 					crops: train_crops[I],
# 					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
# 					augment: True,
# 				})

# 			y_hat = np.empty(shape=(0,2))
# 			for J in np.array_split(range(N_test),  16):
# 				y_hat = np.concatenate((
# 					y_hat, sess.run(prob, feed_dict={
# 					labels: test_labels[J],
# 					crops: test_crops[J],
# 					keep_prob: [1. for i in range(DEPTH)],
# 					augment: False,
# 				})))

# 			print "Epoch: ", TOTAL_EPOCHS+epoch+1
# 			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
# 			print "Err: ", err
# 			print "F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

# 			if err < MIN_ERR:
# 				print "Saving..."
# 				MIN_ERR = err
# 				SAVED_EPOCHS = TOTAL_EPOCHS+epoch+1
# 				new_saver = tf.train.Saver(max_to_keep=2)
# 				new_saver.save(sess, "saved/top_down_net")

# print "SAVED EPOCHS: ", SAVED_EPOCHS
# print "MIN ERR: ", MIN_ERR
# print "Done!"