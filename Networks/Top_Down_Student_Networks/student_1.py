import sys, getopt
import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if __name__ == '__main__':

	TEMP = 10. # Temperature while using knowledge distillation.
	BETA = 0.05 # Weight given to true labels while using knowledge distillation.

	optlist = getopt.getopt(sys.argv[1:], "K:H:M:")[0]
	for opt, arg in optlist:
		if   opt=='-K': K=int(arg)
		elif opt=='-H': H=int(arg)
		elif opt=='-M': MODE=arg # one of baseline, reg_logits or know_dist

	NAME = "student_1_{0}_{1}"
	print "###################### Training Top Down View", NAME, MODE

	if MODE=="baseline" or MODE=="reg_logits":
		FOLDER = "/notebooks/Networks/Top_Down_Student_Networks/saved/{0}/{1}/".format(NAME, MODE)
	elif MODE=="know_dist":
		FOLDER = "/notebooks/Networks/Top_Down_Student_Networks/saved/{0}/{1}_T{2}_beta{3}/".format(NAME,MODE,TEMP,BETA)

	DATA_PATH                 = "/notebooks/Data/top_down_view"
	CROP_HEIGHT, CROP_WIDTH   = 60, 80
	IMAGE_HEIGHT, IMAGE_WIDTH = 240, 320

	EPOCHS     = 100
	MINI_BATCH = 100

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
	labels      = tf.placeholder(dtype=tf.int32, shape=(None,))
	t_logits    = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1, 2))
	images      = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32)
	augment     = tf.placeholder(dtype=tf.bool, name="augment")
	training    = tf.placeholder(dtype=tf.bool)

	current = tf.cond(augment, lambda: data_aug(images), lambda: images)
	current = batch_norm(current, training)
	current = conv(current, 3, 3, 3, K)
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob)
	current = batch_norm(current, training)

	current = conv(current, 60, 80, K, H, padding="VALID")
	current = leaky_relu(current)
	current = tf.nn.dropout(current, keep_prob)
	current = batch_norm(current, training)

	current = conv(current, 1, 1, H, 2, padding="VALID")
	logits  = current
	f_log   = tf.reshape(logits, shape=(-1,2))

	return labels, t_logits, images, keep_prob, training, augment, logits, f_log


def baseline():
	'''
	Train student network directly on the labels.
	'''
	labels, t_logits, images, keep_prob, training, augment, logits, f_log = graph()

	prob       = tf.nn.softmax(logits)
	f_prob     = tf.nn.softmax(f_log)
	loss       = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		 logits=f_log, labels=labels))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, f_prob, train_step

def regression_on_logits():
	'''
	Train the student network on a regression task to target the teacher logits.
	'''
	labels, t_logits, images, keep_prob, training, augment, logits, f_log = graph()

	prob       = tf.nn.softmax(logits)
	f_prob     = tf.nn.softmax(f_log)
	ft_log     = tf.reshape(t_logits, shape=(-1,2))
	f_prob     = tf.nn.softmax(f_log)
	loss       = tf.reduce_mean(tf.nn.l2_loss(f_log-ft_log))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, f_prob, train_step

def knowledge_distillation(T, beta):
	'''
	Train the student network via knowledge distillation.

	T: temperature (float).
	beta: weight given to true labels (between 0 and 1)
	'''
	assert 0.<=beta and beta <=1.

	labels, t_logits, images, keep_prob, training, augment, logits, f_log = graph()

	prob       = tf.nn.softmax(logits)
	ft_log     = tf.reshape(t_logits, shape=(-1,2))
	f_prob     = tf.nn.softmax(f_log)
	loss       = tf.pow(T,2.)*(1.-beta)*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		 logits=f_log/T, labels=tf.nn.softmax(ft_log/T)))+\
		beta*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		 logits=f_log, labels=labels))
	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, images, keep_prob, training, augment,\
	prob, f_prob, train_step


if __name__ == '__main__':

	log = open("{0}log.txt".format(FOLDER),"w")

	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
		train_labels, train_crops = pkl.load(f)

	with open("{0}/teacher_logits.pkl".format(DATA_PATH),"rb") as f:
		teacher_logits = pkl.load(f)

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	train_labels, train_crops = train_labels[:16300], train_crops[:16300]
	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_train, N_test = len(train_labels), len(test_labels)


	with tf.Session() as sess:

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
		prob, f_prob, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		min_err = 0.08
		for epoch in range(EPOCHS):
			for __ in range(N_train/MINI_BATCH):

				I = np.random.choice(range(N_train), size=100, replace=False)
				sess.run([train_step, bn_update], feed_dict={
					labels: train_labels[I],
					t_logits: teacher_logits[I],
					images: train_crops[I],
					keep_prob: .5,
					augment: True,
					training: True
				})

			y_hat = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):
				y_hat = np.concatenate((
					y_hat, sess.run(f_prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: 1.,
					augment: False,
					training: False
				})))


			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "epoch: ", epoch+1
			print "Crp Err: ", err
			log.write("epoch:  {0}\nErr:  {1}\n".format(epoch+1,err))
			log.flush()
			if err<min_err:
				min_err=err
				new_saver = tf.train.Saver(max_to_keep=1)
				new_saver.save(sess, "{0}{1}".format(FOLDER, NAME))

				precision, recall, f1, support = precision_recall_fscore_support(
					test_labels,
					np.argmax(y_hat,axis=1)
				)

				precision, recall, f1 = precision[1], recall[1], f1[1]

				print "Crp Precision: ", precision
				print "Crp Recall: ", recall
				print "Crp F1 Score: ", f1

	log.close()
	print "Done!"