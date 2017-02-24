import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import ImageFile
from scipy.misc import imread

ImageFile.LOAD_TRUNCATED_IMAGES = True

### An alternative to top_down_net where batch normalization is always
### performed using the mini-batch statistics.

CRP_EPOCHS     = 0
CRP_MINI_BATCH = 100

PXL_EPOCHS     = 1

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

def pxl_dense_targets(temp, coords):
	'''
	Produce a dense target for localization training.
	'''

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
		tf.minimum(1., tf.exp(temp*(-tf.square(rows-coord[0])-\
					tf.square(cols-coord[1])))
		),
		tf.to_float(coords)
	)

	target = tf.minimum(1., tf.reduce_sum(target_list, axis=0))

	return tf.stack((target,1-target),axis=0)

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

def graph_crop_class(crp_labels, keep_prob, c8):
	'''
	Crop classification portion of the graph.
	'''
	current = conv(c8, 2, 4, MAX_CHAN, MAX_CHAN, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current = batch_norm(current)
	current = conv(current, 1, 1, MAX_CHAN, 2)

	crp_log   = current[:,0,0,:]
	crp_prob  = tf.nn.softmax(crp_log, name="crp_prob")
	crp_loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=crp_log, labels=crp_labels), name="loss")

	return crp_log, tf.to_float(crp_labels), crp_prob, crp_loss

def graph_pxl_class(temp, pxl_labels, keep_prob, c8):
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

	pxl_labels   = pxl_dense_targets(temp, pxl_labels)
	f_pxl_labels = tf.reshape(pxl_labels, shape=(-1,2))
	pxl_loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=f_pxl_log, labels=f_pxl_labels), name="pxl_loss")

	return pxl_log, pxl_labels, pxl_prob, pxl_loss

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
		   shape=(None, 2))
		)

	images      = tf.placeholder(dtype=tf.float32,
		   shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32, shape=(DEPTH,))
	augment     = tf.placeholder(dtype=tf.bool, name="augment")
	temp         = tf.placeholder(dtype=tf.float32)

	# Convolution / Fractional Max Pooling
	c8 = graph_conv(images, augment, keep_prob)

	# Process crop or image
	logits, f_labels, prob, loss =\
	tf.cond(
		is_crops,
		lambda: graph_crop_class(labels, keep_prob, c8),
		lambda: graph_pxl_class(temp, labels, keep_prob, c8)
	)

	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return labels, images, keep_prob, augment, temp, c8,\
	is_crops, logits, f_labels, prob, train_step

def pxl_img_lab(tt, image_name=None):
	'''
	Produce a batch of images and pixel labels.
	'''
	if not image_name:
		image_name = np.random.choice(meta[tt].keys())

	image = imread("{0}/images/{1}/{2}.jpg".format(DATA_PATH,tt,image_name))
	
	return meta[tt][image_name], [image]

def test(trn, aug, crop):
	'''
	Test with generated data.
	'''

	crp_labs = np.ones(shape=(100,), dtype=int)
	pxl_labs = np.ones(shape=(1,IMAGE_HEIGHT,IMAGE_WIDTH), dtype=int)
	imgs     = np.random.uniform(size=(2,IMAGE_HEIGHT,IMAGE_WIDTH,3))

	with tf.Session() as sess:

		labels, images, keep_prob, augment, temp, c8,\
		is_crops, logits, f_labels, prob, train_step = graph()

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

		labels, images, keep_prob, augment, temp, c8,\
		is_crops, logits, f_labels, prob, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		min_err = 0.06
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
					temp: 1.0
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
					temp: 1.0
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


		coord   = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		TEMP   = 1.0
		max_f1 = 0.95
		for epoch in range(PXL_EPOCHS):
			print "epoch: ", epoch+1
			for __ in range(len(meta["train"])):
				pxl_labs, imgs = pxl_img_lab("train")
				sess.run(train_step, feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
					augment: True,
					is_crops: False,
					temp: TEMP
				})

			y_hat, test_labels = np.empty(shape=(0,2)), np.empty(shape=(0,2))
			for __ in range(len(meta["test"])):
				pxl_labs, imgs = pxl_img_lab("test")
				y_hat = np.concatenate((
					y_hat, np.reshape(sess.run(prob, feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: False,
					temp: TEMP
				}), newshape=(-1,2))))

				test_labels = np.concatenate((
					test_labels, np.reshape(sess.run(f_labels, feed_dict={
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: False,
					temp: TEMP
				}), newshape=(-1,2))))

				TEMP *= 1.01

			precision, recall, f1, support = precision_recall_fscore_support(
				np.argmax(test_labels, axis=1),
				np.argmax(y_hat,axis=1)
			)

			precision, recall, f1 = precision[1], recall[1], f1[1]

			if max_f1<f1:
				max_f1=f1
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/top_down_net")

				print "Pxl Err: ", 1-accuracy_score(
					np.argmax(test_labels, axis=1),
					np.argmax(y_hat,axis=1)
					)
				print "Pxl Precision: ", precision
				print "Pxl Recall: ", recall
			print "Pxl F1 Score: ", f1

		coord.request_stop()
		coord.join(threads)

	print "Done!"