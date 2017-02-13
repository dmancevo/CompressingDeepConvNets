import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score

EPOCHS     = 25
MINI_BATCH = 100

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

def graph(conv_depth):
	'''
	Build Graph.
	'''

	# Load the data...
	labels       = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
	images       = tf.placeholder(dtype=tf.float32,
		shape=(None,CROP_HEIGHT,CROP_WIDTH,3), name="images")


	# Data Augmentation
	augment = tf.placeholder(tf.bool, name="augment")
	current = tf.cond(augment, lambda: data_aug(images), lambda: images)

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

	return labels, images, augment, current

labels, images, augment, current = graph(conv_depth=5)

init_op = tf.global_variables_initializer()

print current.get_shape(), current.dtype

# with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
# 	train_labels, train_crops = pkl.load(f)

with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
	test_labels, test_crops = pkl.load(f)

# N_train, N_test = len(train_labels), len(test_labels)

N_test = len(test_labels)

with tf.Session() as sess:
	
	sess.run(init_op)

	for mb in np.array_split(range(N_test), N_test/MINI_BATCH):
	
		C = sess.run(current, feed_dict={
			labels: test_labels[mb],
			images: test_crops[mb],
			augment:False,
			})

		print C.shape

		break