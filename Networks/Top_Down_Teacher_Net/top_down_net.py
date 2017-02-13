import numpy as np
import pickle as pkl
from cropping import batch_crop
import tensorflow as tf
from sklearn.metrics import accuracy_score

EPOCHS = 25

DATA_PATH = "/notebooks/Data/top_down_view"
CROP_HEIGHT, CROP_WIDTH = 60, 60
IMG_HEIGHT, IMG_WIDTH   = 240, 320

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

def graph(conv_depth):
	'''
	Build Graph.
	'''

	# Load the data...
	batch_size   = tf.placeholder(dtype=tf.int32)
	labels       = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
	crop_windows = tf.placeholder(dtype=tf.float32,
		shape=(None, CROP_HEIGHT, CROP_WIDTH,3))
	images       = tf.placeholder(dtype=tf.float32,
		shape=(None,IMG_HEIGHT,IMG_WIDTH,3), name="images")

	# Crop if crop_bool is set to True
	crop_bool      = tf.placeholder(tf.bool, name="crop_bool")
	current        = tf.cond(crop_bool,
		lambda: crop_windows,
		lambda: images
	)

	# Data Augmentation
	augment = tf.placeholder(tf.bool, name="augment")
	current = tf.cond(augment, lambda: data_aug(current), lambda: current)

	chan_in, chan_out = 3, 10
	for i in range(1,conv_depth+1):
		chan_out = i*10
		W        = tf.Variable(
			np.random.normal(0,0.01,size=(3,3,chan_in,chan_out)),
			dtype=tf.float32
		)
		current    =tf.nn.conv2d(
			input  =current,
			filter =W,
			strides=(1,1,1,1),
			padding="SAME",
		)
		b       = tf.Variable(
			np.random.normal(0,0.01,size=chan_out),
			dtype=tf.float32
			)
		current = tf.nn.bias_add(current, b)
		current = tf.nn.relu(current)
		current = tf.nn.fractional_max_pool(
			value=current,
			pooling_ratio=(1.0, 1.414, 1.414, 1.0),
			pseudo_random=True,
			)[0]
		chan_in = i*10

	return batch_size, labels, crop_windows, images, crop_bool, augment, current

batch_size, labels, crop_windows, images, crop_bool,\
 augment, current = graph(conv_depth=7)

init_op = tf.global_variables_initializer()

print current.get_shape(), current.dtype

with tf.Session() as sess:

	
	sess.run(init_op)

	batch, mini_batch = 0, 0
	
	with open("{0}/top_down_view_train_{1}.pkl".format(DATA_PATH,batch), "rb") as f:
		data = pkl.load(f)

	crop_boxes, true_labels = batch_crop(
		data[mini_batch]["data"],
		data[mini_batch]["labels"]
		)
	
	C = sess.run(current, feed_dict={
		batch_size: len(data[mini_batch]["bin_labels"]),
		labels: true_labels,
		crop_windows: crop_boxes,
		images: data[mini_batch]["data"],
		crop_bool: True,
		augment:True,
		})

	print C.shape


# Load labels into SparseTensors by passing indices (corresponding to "heads"),
# values (all ones) and shape (which is constant -  240x320)

# Might also want to assign 0.5 to pixels neighboring "head" pixels.