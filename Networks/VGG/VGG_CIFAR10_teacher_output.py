import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score

VGG = 16

with tf.Session() as sess:

	saver = tf.train.import_meta_graph("saved/VGG-{0}_CIFAR10.meta".format(VGG))
	saver.restore(sess, tf.train.latest_checkpoint('saved/'))
	layers = tf.get_collection('layers')

	print "Successfully loaded graph from file."

	row_images, labels, augment, keep_prob, logits, prob, train_step = layers

	lgts = np.empty(shape=(0,10))
	for i in range(1,6):

		with open("/notebooks/Data/cifar10/data_batch_{0}".format(i),"rb") as f:
			data = pkl.load(f)

		data["data"] = np.array(data["data"])

		for I in np.array_split(range(10000), 100):

			lgts   = np.concatenate((
				lgts, sess.run(logits, feed_dict={
				row_images: data["data"][I],
				augment: False,
				keep_prob:  1.0,
				})[:,0,0,:]
			))

		print 1-accuracy_score(
			data["labels"],
			np.argmax(lgts[((i-1)*10000):(i*10000)], axis=1)
		)

with open("/notebooks/Data/cifar10/vgg{0}_logits.pkl".format(VGG),"wb") as f:
	pkl.dump(lgts, f)

print "Done!"