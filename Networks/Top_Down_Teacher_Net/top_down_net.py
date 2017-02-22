import numpy as np
import pickle as pkl
import tensorflow as tf
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CRP_ERR        = 0.048
CRP_EPOCHS     = 1
CRP_MINI_BATCH = 100

PXL_F1         = 0.98
PXL_EPOCHS     = 1

FMP      = np.sqrt(2)
DEPTH    = 8
CHANNELS = 15
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

def batch_norm(current, training):
	'''
	Batch Normalization wrapper.
	'''
	current = tf.layers.batch_normalization(
			current,
			axis=-1,
			training=training,
			momentum=0.9,
		)

	return current

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
		current = batch_norm(current, training)
		chan_in  = i*CHANNELS

	return current

def graph_crop_class(training, labels, keep_prob, c8):
	'''
	Crop classification portion of the graph.
	'''
	current = conv(c8, 2, 4, MAX_CHAN, MAX_CHAN, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob[-1])
	current = batch_norm(current, training)
	current = conv(current, 1, 1, MAX_CHAN, 2)

	crop_log   = current[:,0,0,:]
	crop_prob  = tf.nn.softmax(crop_log, name="crop_prob")
	crp_loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=crop_log, labels=labels), name="loss")

	return crop_log, crop_prob, crp_loss

def graph_pxl_class(training, pxl_labels, keep_prob, c8):
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
	current = batch_norm(current, training)
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
	training    = tf.placeholder(dtype=tf.bool, name="training")

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
	c8 = graph_conv(training, images, augment, keep_prob)

	# Process crop or image
	logits, prob, loss =\
	tf.cond(
		is_crops,
		lambda: graph_crop_class(training, labels, keep_prob, c8),
		lambda: graph_pxl_class(training, labels, keep_prob, c8)
	)

	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")

	return training, labels, images, keep_prob, augment, c8,\
	is_crops, logits, prob, train_step

def pxl_img_lab():
	'''
	Produce a batch of images and pixel labels.
	'''
	image_file, tensor_image = sess.run([trn_file, trn_image])
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

		training, labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = graph()

		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		print sess.run(prob, feed_dict={
			training: trn,
			labels: crp_labs if crp else pxl_labs,
			images: imgs,
			keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
			augment: aug,
			is_crops: crop,
		}).shape


if __name__ == '__main__':

	# test(trn=True, aug=True, crop=False)

	# with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
	# 	train_labels, train_crops = pkl.load(f)

	# with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
	# 	test_labels, test_crops = pkl.load(f)


	# train_labels, train_crops = train_labels[:16300], train_crops[:16300]
	# test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	# N_train, N_test = len(train_labels), len(test_labels)


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

		training, labels, images, keep_prob, augment, c8,\
		is_crops, logits, prob, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		# for epoch in range(CRP_EPOCHS):
		# 	for __ in range(N_train/CRP_MINI_BATCH):

		# 		I = np.random.choice(range(N_train), size=100, replace=False)
		# 		sess.run([train_step, bn_update], feed_dict={
		# 			training: True,
		# 			labels: train_labels[I],
		# 			# pxl_labels: pxl_labs,
		# 			images: train_crops[I],
		# 			keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
		# 			augment: True,
		# 			is_crops: True,
		# 		})

		# 	# Update batch normalization mean and std only.
		# 	for __ in range(150):
		# 		I = np.random.choice(range(N_train), size=100, replace=False)
		# 		sess.run(bn_update, feed_dict={
		# 			training: False,
		# 			labels: train_labels[I],
		# 			images: train_crops[I],
		# 			keep_prob: [1. for i in range(DEPTH)],
		# 			augment: False,
		# 			is_crops: True,
		# 		})

		# 	y_hat = np.empty(shape=(0,2))
		# 	for J in np.array_split(range(N_test),  16):
		# 		y_hat = np.concatenate((
		# 			y_hat, sess.run(prob, feed_dict={
		# 			training: False,
		# 			labels: test_labels[J],
		# 			images: test_crops[J],
		# 			keep_prob: [1. for i in range(DEPTH)],
		# 			augment: False,
		# 			is_crops: True,
		# 		})))

		# 	print "Crp Epoch: ", epoch+1
		# 	err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
		# 	print "Crp Err: ", err
		# 	print "Crp Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
		# 	print "Crp Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
		# 	print "Crp F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))

		# 	if err < CRP_ERR:
		# 		print "Crp Saving..."
		# 		CRP_ERR = err
		# 		saved_crop_epochs = epoch+1
		# 		new_saver = tf.train.Saver(max_to_keep=2)
		# 		new_saver.save(sess, "saved/top_down_net")

		# print "\n\n\n"

		coord   = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for epoch in range(PXL_EPOCHS):
			for __ in range(len(meta["train"])):
				pxl_labs, imgs = pxl_img_lab()
				sess.run([train_step, bn_update], feed_dict={
					training: True,
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1., .9, .8, .7, .6, .5, .5, .5],
					augment: True,
					is_crops: False,
				})

			# Update batch normalization mean and std only.
			for __ in range(150):
				pxl_labs, imgs = pxl_img_lab()
				sess.run(bn_update, feed_dict={
					training: False,
					labels: pxl_labs,
					images: imgs,
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
					is_crops: True,
				})

			y_hat, test_labels = np.empty(shape=(0,2)), np.empty(shape=(0,))
			for __ in range(len(meta["test"])):
				pxl_labs, imgs = pxl_img_lab()
				y_hat = np.concatenate((
					y_hat, np.reshape(sess.run(prob, feed_dict={
					training: False,
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

			print "Pxl Epoch: ", epoch+1
			print "Pxl Err: ", 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Pxl Precision: ", precision_score(test_labels,np.argmax(y_hat,axis=1))
			print "Pxl Recall: ", recall_score(test_labels,np.argmax(y_hat,axis=1))
			f1 = f1_score(test_labels,np.argmax(y_hat,axis=1))
			print "Pxl F1 Score: ", f1

			if PXL_F1<f1:
				print "Pxl Saving..."
				PXL_MIN_ERR = f1
				saved_crop_epochs = epoch+1
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/top_down_net")

		coord.request_stop()
		coord.join(threads)

print "Done!"