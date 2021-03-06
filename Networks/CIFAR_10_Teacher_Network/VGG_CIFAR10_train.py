import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score

from load import load

VGG        = 16
N_FC       = 1024 # Fully Connected Layers
EPOCHS     = 50

def batch_norm(layer, dim):
	'''
	Batch normalization.
	'''
	b_mean, b_var = tf.nn.moments(layer,[0])
	scale = tf.Variable(tf.ones([dim]))
	beta          = tf.Variable(tf.zeros([dim]))
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
		lambda img: tf.image.random_brightness(img, max_delta=63), images)
	images = tf.map_fn(
		lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), images)

	return images

def graph(VGG, N_FC):
	'''
	Build Graph.
	'''
	# Fetch images
	row_images = tf.placeholder(tf.float32, shape=(None, 3072), name="row_images")
	labels     = tf.placeholder(dtype=tf.int32, name="labels")
	images     = tf.transpose(tf.reshape(row_images, (-1, 3, 32, 32)), perm=[0,2,3,1])

	# Data Augmentation
	augment = tf.placeholder(tf.bool, name="augment")
	images  = tf.cond(augment, lambda: data_aug(images), lambda: images)

	# Convolution
	pool5      = load(VGG, images, layer="pool5")
	keep_prob  = tf.placeholder(dtype=tf.float32)

	current = batch_norm(pool5, 512)

	for k in [(1,1,512,N_FC), "drop", (1,1,N_FC,N_FC), "drop", (1,1,N_FC,10)]:

		if k=="drop":
			current = tf.nn.dropout(current, keep_prob)
		else:
			W       = tf.Variable(np.random.normal(0,0.01,size=k), dtype=tf.float32)
			conv    = tf.nn.conv2d(current, W, strides=(1,1,1,1), padding='SAME')
			current = tf.nn.relu(batch_norm(conv,k[-1]))

	logits     = current
	prob       = tf.nn.softmax(logits, name="prob")
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=logits, labels=labels), name="loss")
	train_step = tf.train.AdamOptimizer().minimize(loss, name="train_step")
	
	return row_images, labels, augment, keep_prob, logits, prob, train_step

if __name__ == '__main__':
	with tf.Session() as sess:

		try:
			saver = tf.train.import_meta_graph("saved/VGG-{0}_CIFAR10.meta".format(VGG))
			saver.restore(sess, tf.train.latest_checkpoint('saved/'))
			layers = tf.get_collection('layers')

			print "Successfully loaded graph from file."

		except IOError:

			print "Building graph from scratch..."

			layers = graph(VGG, N_FC)
			for layer in layers:
				tf.add_to_collection('layers', layer)

			init_op    = tf.global_variables_initializer()
			sess.run(init_op)

		row_images, labels, augment, keep_prob, logits, prob, train_step = layers

		with open("/notebooks/Data/cifar10/test_batch","rb") as f:
			test_data = pkl.load(f)

		test_data["data"]   = np.array(test_data["data"])
		test_data["labels"] = np.array(test_data["labels"], dtype=int)

		min_err = 0.15
		for epoch in range(EPOCHS):
			print"epoch: ", epoch+1
			for i in range(1,6):

				with open("/notebooks/Data/cifar10/data_batch_{0}".format(i),"rb") as f:
					data = pkl.load(f)

				data["data"]   = np.array(data["data"])
				data["labels"] = np.array(data["labels"], dtype=int)

				for I in np.array_split(range(10000), 100):
					sess.run(train_step, feed_dict={
						row_images: data["data"][I],
						labels: data["labels"][I],
						augment: True,
						keep_prob: 0.5,
						})

			y_hat = np.empty(shape=(0,10))
			for J in np.array_split(range(10000), 100):
				y_hat = np.concatenate((
					y_hat, sess.run(prob, feed_dict={
					row_images: test_data["data"][J],
					augment: False,
					keep_prob: 1.0,
					})[:,0,0,:]
				))

			test_labels = np.array(test_data["labels"])

			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Err: ", err
			if err<min_err:
				print "Saving..."
				min_err=err
				new_saver = tf.train.Saver(max_to_keep=2)
				new_saver.save(sess, "saved/VGG-{0}_CIFAR10".format(VGG))


	print "Done!"