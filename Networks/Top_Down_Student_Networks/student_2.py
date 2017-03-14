import sys, getopt
import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from student_1 import batch_norm, data_aug, conv, leaky_relu

if __name__ == '__main__':

	optlist = getopt.getopt(sys.argv[1:], "C:H:")[0]
	for opt, arg in optlist:
		if   opt=='-C': CHANNELS=int(arg)
		elif opt=='-H': HINT_TRAINING=bool(arg)

	TEMP = 10. # Temperature while using knowledge distillation.
	BETA = 0.05 # Weight given to true labels while using knowledge distillation.

	NAME = "student_2_{0}".format(CHANNELS)
	print "Training Top Down View", NAME 

	if HINT_TRAINING:
		FOLDER = "/notebooks/Networks/Top_Down_Student_Networks/saved/{0}/hint_based/".format(NAME)
	else:
		FOLDER = "/notebooks/Networks/Top_Down_Student_Networks/saved/{0}/know_dist_only/".format(NAME)

	DATA_PATH                 = "/notebooks/Data/top_down_view"
	CROP_HEIGHT, CROP_WIDTH   = 60, 80
	IMAGE_HEIGHT, IMAGE_WIDTH = 240, 320

	HINT_EPOCHS = 30
	EPOCHS      = 10
	MINI_BATCH  = 100

def graph():
	'''
	Student 2.
	'''

	# Placeholders
	labels      = tf.placeholder(dtype=tf.int32, shape=(None,))
	t_logits    = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1, 2))
	hints       = tf.placeholder(dtype=tf.float32, shape=(None, 9, 13, 125))
	images      = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
	keep_prob   = tf.placeholder(dtype=tf.float32)
	augment     = tf.placeholder(dtype=tf.bool, name="augment")
	training    = tf.placeholder(dtype=tf.bool)

	current = tf.cond(augment, lambda: data_aug(images), lambda: images)
	current = batch_norm(current, training)

	chan_in = 3

	for c in range(1,6):
		chan_out = c*CHANNELS
		current  = conv(current, 3, 3, chan_in, chan_out)
		current  = leaky_relu(current)
		current  = tf.nn.dropout(current, keep_prob)
		current  = tf.nn.max_pool(current, (1,2,2,1), (1,2,2,1), "VALID")
		current  = batch_norm(current, training)
		chan_in  = chan_out

		if c==3:
			c3 = current

	# Logits
	current     = conv(
		current =current,
		height  =1,
		width   =2,
		chan_in =chan_out,
		chan_out=2, 
		padding ="VALID",
	)
	logits      = current
	f_log       = tf.reshape(logits, shape=(-1,2))

	# Hint-based training step
	Wr  = tf.Variable(
		np.random.normal(
			size=(3,4,125,CHANNELS*3)),
		dtype=tf.float32
	)
	reg = tf.nn.conv2d_transpose(
		value  =c3,
		filter =Wr,
		output_shape=(tf.shape(current)[0], 9, 13, 125),
		strides=(1,1,1,1),
		padding="VALID",
	)

	hint_loss       = tf.reduce_mean(tf.nn.l2_loss(hints-reg))
	hint_train_step = tf.train.AdamOptimizer(name="hint_train").minimize(hint_loss)


	# Knowledge distillation
	prob   = tf.nn.softmax(logits)
	ft_log = tf.reshape(t_logits, shape=(-1,2))
	f_prob = tf.nn.softmax(f_log)
	loss   = tf.pow(TEMP,2.)*(1.-BETA)*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		 logits=f_log/TEMP, labels=tf.nn.softmax(ft_log/TEMP)))+\
		 BETA*tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	     logits=f_log, labels=labels))

	train_step = tf.train.AdamOptimizer().minimize(loss)

	return labels, t_logits, hints, images, keep_prob, training, augment,\
	prob, f_prob, hint_loss, hint_train_step, train_step


if __name__ == '__main__':

	log = open("{0}log.txt".format(FOLDER),"w")

	with open("{0}/train.pkl".format(DATA_PATH), "rb") as f:
		train_labels, train_crops = pkl.load(f)

	with open("{0}/teacher_logits.pkl".format(DATA_PATH),"rb") as f:
		teacher_logits = pkl.load(f)

	if HINT_TRAINING:
		with open("{0}/teacher_c5.pkl".format(DATA_PATH), "rb") as f:
			teacher_hints = pkl.load(f)

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	train_labels, train_crops = train_labels[:16300], train_crops[:16300]
	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_train, N_test = len(train_labels), len(test_labels)


	with tf.Session() as sess:

		print "Building graph from scratch..."

		layers = graph()

		for layer in layers:
			tf.add_to_collection('layers', layer)

		init_op    = tf.global_variables_initializer()
		sess.run(init_op)

		labels, t_logits, hints, images, keep_prob, training, augment,\
		prob, f_prob, hint_loss, hint_train_step, train_step = layers

		bn_update = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

		print "Hint training..."
		if HINT_TRAINING:
			for epoch in range(HINT_EPOCHS):
				curr_hint_loss = []
				for __ in range(N_train/MINI_BATCH):

					I = np.random.choice(range(N_train), size=MINI_BATCH, replace=False)
					sess.run([hint_train_step, bn_update], feed_dict={
						hints: teacher_hints[I],
						images: train_crops[I],
						keep_prob: .5,
						augment: False,
						training: True
					})

					curr_hint_loss.append(sess.run(hint_loss, feed_dict={
						hints: teacher_hints[I],
						images: train_crops[I],
						keep_prob: 1.,
						augment: False,
						training: False
					}))

				m_hint_loss = np.mean(curr_hint_loss)
				print "Hint Loss: ", m_hint_loss
				log.write("Hint Loss:  {0}\n".format(m_hint_loss))
				log.flush()

		print "Training..."
		min_err = 0.08
		for epoch in range(EPOCHS):
			for __ in range(N_train/MINI_BATCH):

				I = np.random.choice(range(N_train), size=MINI_BATCH, replace=False)
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
				new_saver.save(sess, "{0}student_2".format(FOLDER))

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