import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### An alternative to top_down_net where batch normalization is always
### performed using the mini-batch statistics.

CRP_EPOCHS     = 50
CRP_MINI_BATCH = 100

PXL_EPOCHS     = 0

FMP      = np.sqrt(2)
DEPTH    = 8
CHANNELS = 20
MAX_CHAN = DEPTH*CHANNELS
PXL_CHAN = 8

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

def pxl_dense_targets():
	'''
	Produce a dense target for localization training.
	'''

	coords = tf.placeholder(dtype=tf.float32, shape=(None, 2))
	tau  = tf.placeholder(dtype=tf.float32)

	rows = tf.meshgrid(
		tf.range(IMAGE_HEIGHT, dtype=tf.float32),
	 	tf.ones(IMAGE_WIDTH),
	 	indexing='ij'
	 )[0]
	cols = tf.meshgrid(
		tf.ones(IMAGE_HEIGHT),
		tf.range(IMAGE_WIDTH, dtype=tf.float32),
		indexing='ij'
	)[1]

	target = tf.zeros(shape=(IMAGE_HEIGHT,IMAGE_WIDTH), dtype=tf.float32)

	target_list = tf.map_fn(
		lambda coord: \
		tf.minimum(1., tf.exp(tau*(-tf.square(rows-coord[0])-\
					tf.square(cols-coord[1])))
		),
		coords
	)

	target = tf.minimum(1., tf.reduce_sum(target_list, axis=0))

	return coords, tau, target

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

def graph_conv(images, augment, keep_prob):
	'''
	Build convolution-fractional max pooling portion of graph.
	'''

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

	return current

def graph_crop_class(labels, keep_prob, c8):
	'''
	Crop classification portion of the graph.
	'''
	current = conv(c8, 2, 4, MAX_CHAN, MAX_CHAN, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current = batch_norm(current)
	current = conv(current, 1, 1, MAX_CHAN, 2)

	crop_log   = current[:,0,0,:]
	crop_prob  = tf.nn.softmax(crop_log, name="crop_prob")
	crp_loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=crop_log, labels=labels), name="loss")

	return crop_log, crop_prob, crp_loss

def graph_pxl_class(pxl_labels, keep_prob, c8):
	'''
	Segmentation portion of the graph.
	'''

	fltr = (18,17, PXL_CHAN, MAX_CHAN)

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
	current = batch_norm(current)
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
	is_crops    = tf.placeholder(dtype=tf.bool, name="is_crops")

	labels      = tf.cond(
		is_crops,
		lambda: tf.placeholder(dtype=tf.int32, shape=(None,)),
		lambda: tf.placeholder(dtype=tf.int32,
		   shape=(None, IMAGE_HEIGHT,IMAGE_WIDTH))
		)

	images      = tf.placeholder(dtype=tf.float32,
		   shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))
	augment = tf.placeholder(dtype=tf.bool, name="augment")

	# Convolution / Fractional Max Pooling
	c8 = graph_conv(images, augment, keep_prob)

	# Process crop or image
	logits, prob, loss =\
	tf.cond(
		is_crops,
		lambda: graph_crop_class(labels, keep_prob, c8),
		lambda: graph_pxl_class(labels, keep_prob, c8)
	)

	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return labels, images, keep_prob, augment, c8,\
	is_crops, logits, prob, train_step

def pxl_img_lab(file, image):
	'''
	Produce a batch of images and pixel labels.
	'''
	image_file, tensor_image = sess.run([file, image])
	image_name = re.match(r'.*\/([^\/]*)\.jpg',image_file).group(1)

	pxl_lab = np.zeros(shape=(IMAGE_HEIGHT,IMAGE_WIDTH), dtype=int)
	for coord in meta["train"][image_name]:
		pxl_lab[coord[0], coord[1]] = 1

	return [pxl_lab], [tensor_image]


def test(trn, aug, crop):
	'''
	Test with generated data.
	'''

	crp_labs = np.ones(shape=(100,), dtype=int)
	pxl_labs = np.ones(shape=(1,IMAGE_HEIGHT,IMAGE_WIDTH), dtype=int)
	imgs     = np.random.uniform(size=(2,IMAGE_HEIGHT,IMAGE_WIDTH,3))

	with tf.Session() as sess:

		labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = graph()

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		print sess.run(prob, feed_dict={
			labels: crp_labs if crp else pxl_labs,
			images: imgs,
			keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
			augment: aug,
			is_crops: crop,
		}).shape


if __name__ == '__main__':

	# test(trn=True, aug=True, crop=False)

	if CRP_EPOCHS:

		with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
			train_labels, train_crops = pkl.load(f)

		with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
			test_labels, test_crops = pkl.load(f)


		train_labels, train_crops = train_labels[:16300], train_crops[:16300]
		test_labels, test_crops = test_labels[:4800], test_crops[:4800]

		N_train, N_test = len(train_labels), len(test_labels)


	if PXL_EPOCHS:

		with open(DATA_PATH+"/images/meta.pkl", "rb") as f:
			meta = pkl.load(f)

		trn_str_queue = ["{0}/{1}/{2}.jpg".format(DATA_PATH, "images/train", img) \
		for img in meta["train"].keys()]

		tst_str_queue = ["{0}/{1}/{2}.jpg".format(DATA_PATH, "images/test", img) \
		for img in meta["test"].keys()]

		trn_file_queue = tf.train.string_input_producer(trn_str_queue, shuffle=True)
		tst_file_queue = tf.train.string_input_producer(tst_str_queue, shuffle=True)

		image_reader = tf.WholeFileReader()

		trn_file, trn_image_file = image_reader.read(trn_file_queue)
		tst_file, tst_image_file = image_reader.read(tst_file_queue)

		trn_image = tf.image.decode_jpeg(trn_image_file)
		tst_image = tf.image.decode_jpeg(tst_image_file)

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

		labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		min_err = 0.05
		for epoch in range(CRP_EPOCHS):
			print "epoch: ", epoch+1
			for __ in range(N_train/CRP_MINI_BATCH):

				I = np.random.choice(range(N_train), size=100, replace=False)
				sess.run(train_step, feed_dict={
					labels: train_labels[I],
					images: train_crops[I],
					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
					augment: True,
					is_crops: True,
				})

			y_hat = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):
				y_hat = np.concatenate((
					y_hat, sess.run(prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: True,
				})))

			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Crp Err: ", err
			if err<min_err:
				min_err=err
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/top_down_net")

				print "Crp Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
				print "Crp Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
				print "Crp F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

		print "\n\n\n"

		coord   = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		max_f1  = 0.9
		for epoch in range(PXL_EPOCHS):
			print "epoch: ", epoch+1
			for __ in range(len(meta["train"])):
				pxl_labs, imgs = pxl_img_lab(trn_file, trn_image)
				sess.run([train_step, bn_update], feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
					augment: True,
					is_crops: False,
				})

			y_hat, test_labels = np.empty(shape=(0,2)), np.empty(shape=(0,))
			for __ in range(len(meta["test"])):
				pxl_labs, imgs = pxl_img_lab(tst_file, tst_image)
				y_hat = np.concatenate((
					y_hat, np.reshape(sess.run(prob, feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: False,
				}), newshape=(-1,2))))

				test_labels = np.concatenate((
					test_labels,
					np.reshape(pxl_labs, newshape=(-1,))
				))
			y_hat = np.reshape(y_hat, newshape=(-1, 2))

			curr_f1 = f1_score(test_labels,np.argmax(y_hat,axis=1))

			if max_f1<curr_f1:
				max_f1=curr_f1
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/top_down_net")

			
				print "Pxl Err: ", 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
				print "Pxl F1 Score: ", curr_f1

		coord.request_stop()
		coord.join(threads)

print "Done!"