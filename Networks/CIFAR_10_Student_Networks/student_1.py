import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score

# one of "baseline", "reg_logits" or "know_dist".
MODE = "baseline"
TEMP = 10. # Temperature while using knowledge distillation.
BETA = 0.05 # Weight given to true labels while using knowledge distillation.

# One of "vgg16" or "fmp"
TEACHER = "vgg16"

if MODE=="baseline" or MODE=="reg_logits":
	FOLDER = "saved/student_1/{0}/".format(MODE)
elif MODE=="know_dist":
	FOLDER = "saved/student_1/{0}_T{1}_beta{2}_{3}/".format(MODE,TEMP,BETA,TEACHER)

DATA_PATH  = "/notebooks/Data/cifar10"
EPOCHS     = 100
MINI_BATCH = 100
K, H       = 20, 50

def batch_norm(current, training):
	'''
	Batch Normalization wrapper.
	'''
	current = tf.layers.batch_normalization(
			current,
			axis=-1,
			training=training,
			momentum=0.99,
			center=True,
			scale=False
		)

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
	Student Network
	'''
	row_images = tf.placeholder(tf.float32, shape=(None, 3072), name="row_images")
	labels     = tf.placeholder(dtype=tf.int32, name="labels")
	images     = tf.transpose(tf.reshape(row_images, (-1, 3, 32, 32)), perm=[0,2,3,1])
	t_logits   = tf.placeholder(dtype=tf.float32, shape=(None, 10))

	current = tf.cond(augment, lambda: data_aug(images), lambda: images)
	current = batch_norm(current, training)
	current = conv(current, 3, 3, 3, K)
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob)
	current = batch_norm(current, training)

	current = conv(current, 32, 32, K, H, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob)
	current = batch_norm(current, training)

	current = conv(current, 1, 1, H, 10, padding="VALID")
	f_log   = current[:,0,0,:]

	return labels, t_logits, images, keep_prob, training, augment, f_log


def baseline():
	'''
	Train student network directly on the labels.
	'''
	labels, t_logits, images, keep_prob, training, augment, f_log = graph()

	prob       = tf.nn.softmax(f_log)
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		 logits=f_log, labels=labels))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, train_step

def regression_on_logits():
	'''
	Train the student network on a regression task to target the teacher logits.
	'''
	labels, t_logits, images, keep_prob, training, augment, f_log = graph()

	prob       = tf.nn.softmax(f_log)
	ft_log     = tf.reshape(t_logits, shape=(-1,2))
	f_prob     = tf.nn.softmax(f_log)
	loss       = tf.reduce_mean(tf.nn.l2_loss(f_log-ft_log))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, train_step

def knowledge_distillation(T, beta):
	'''
	Train the student network via knowledge distillation.

	T: temperature (float).
	beta: weight given to true labels (between 0 and 1)
	'''
	assert 0.<=beta and beta <=1.

	labels, t_logits, images, keep_prob, training, augment, f_log = graph()

	ft_log     = tf.reshape(t_logits, shape=(-1,2))
	prob       = tf.nn.softmax(f_log)
	loss       = tf.pow(T,2.)*(1.-beta)*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		 logits=f_log/T, labels=tf.nn.softmax(ft_log/T)))+\
		beta*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		 logits=f_log, labels=labels))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, train_step


if __name__ == '__main__':

	with open("/notebooks/Data/cifar10/test_batch","rb") as f:
			test_data = pkl.load(f)

	test_data["data"]   = np.array(test_data["data"])
	test_data["labels"] = np.array(test_data["labels"], dtype=int)

	with open("{0}/{1}_logits.pkl".format(DATA_PATH, TEACHER),"rb") as f:
		teacher_logits = pkl.load(f)

	with tf.Session() as sess:

		try:
			saver = tf.train.import_meta_graph("{0}student_1.meta".format(FOLDER))
			saver.restore(sess, tf.train.latest_checkpoint(FOLDER))
			layers = tf.get_collection('layers')

			print "Successfully loaded graph from file."

		except IOError:

			print "Building graph from scratch..."

			if MODE=="baseline":
				layers = baseline()
			elif MODE=="reg_logits":
				layers = regression_on_logits()
			elif MODE=="know_dist":
				layers = knowledge_distillation(TEMP, BETA)

			for layer in layers:
				tf.add_to_collection('layers', layer)

			init_op    = tf.global_variables_initializer()
			sess.run(init_op)

		labels, t_logits, images, keep_prob, training, augment,\
		prob, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		min_err = 0.15
		for epoch in range(EPOCHS):
			print"epoch: ", epoch+1
			for i in range(1,6):

				with open("/notebooks/Data/cifar10/data_batch_{0}".format(i),"rb") as f:
					data = pkl.load(f)

				data["data"]   = np.array(data["data"])
				data["labels"] = np.array(data["labels"], dtype=int)

				I = np.random.choice(range(N_train), size=100, replace=False)
				sess.run([train_step, bn_update], feed_dict={
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
				new_saver.save(sess, "{0}student_1")


	print "Done!"