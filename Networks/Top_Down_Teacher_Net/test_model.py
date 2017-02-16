import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from scipy.ndimage.filters import maximum_filter


N_avg = 12

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

def crop(img, coord):
	'''
	Crop an image.
	img must have dimensions [batch, heigh, width, channels]
	coord=(y, x): center coordinates (y:height, x:width).
	h, w: height and width of the crop.
	'''
	x, y  = coord
	h, w = CROP_HEIGHT, CROP_WIDTH
	return img[:,int(y-h/2.):int(y+h/2.),int(x-w/2.):int(x+w/2.),:]

def scan_batch(img, step_frac=2.):
	'''
	Returns CROP_HEIGHT x CROP_WIDTH sized crops
	corresponding to the action of sliding a window across
	the corresponding image.

	Only returns valid crops.

	img: numpy array
	step_frac: 1 corresponds to no overlapping between crops.
	           2 corresponds to 50% overlapping.
	'''

	assert step_frac < min(CROP_HEIGHT, CROP_WIDTH)/2.

	crops = np.empty(shape=(0,CROP_HEIGHT, CROP_WIDTH))
	
	x, y = int(CROP_WIDTH/2. + 1), int(CROP_HEIGHT/2. + 1)

	while y < IMG_HEIGHT-1-CROP_HEIGHT/2.:
		while x < IMG_WIDTH-1-CROP_WIDTH/2.:

			crops = np.concatenate((
				crops, crop(img, (x,y))
				))

			x+=int(CROP_WIDTH/(2.*step_frac))
		y+=int(CROP_HEIGHT/(2.*step_frac))

	return crops

def head_count(resp):
	'''
	Count number of human subjects base of model's response
	by applying non-maximum suppression.

	resp are assumed to be row-major.
	'''
	mtrx = np.reshape(resp, newshape=(IMG_HEIGHT, IMG_WIDTH))
	return np.count_nonzero(
		mtrx==maximum_filter(
				mtrx, size=(CROP_HEIGHT, CROP_WIDTH)
		))


if __name__ == '__main__':

	with open("{0}/test.pkl".format(DATA_PATH), "rb") as f:
		test_labels, test_crops = pkl.load(f)

	test_labels, test_crops = test_labels[:4800], test_crops[:4800]

	N_test = len(test_labels)

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("saved/top_down_net.meta")
		saver.restore(sess, tf.train.latest_checkpoint("saved/"))
		layers = tf.get_collection('layers')

		labels, images, keep_prob, augment, logits, prob, loss, train_step =\
			layers

		for n in range(N_avg):

			temp = np.empty(shape=(0,2))
			for J in np.array_split(range(N_test),  16):

				temp = np.concatenate((
					temp, sess.run(prob, feed_dict={
					labels: test_labels[J],
					images: test_crops[J],
					keep_prob: [1. for i in range(DEPTH)],
					augment: False,
				})))

			if n == 0:
				y_hat = temp
			else:
				y_hat = (n*y_hat + temp)/(n+1.)

			print "No. Tests: ", n
			err = 1-accuracy_score(test_labels,np.argmax(y_hat,axis=1))
			print "Err: ", err
			print "F1 Score: ", f1_score(test_labels,np.argmax(y_hat,axis=1))


		# with open("/notebooks/Data/top_down_view/top_down_view_test_0.pkl", "rb") as f:
		# 	test = pkl.load(f)

