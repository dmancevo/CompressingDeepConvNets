import numpy as np
import scipy.io
import pickle as pkl
import tensorflow as tf

def load(L, current, layer):
	L = int(L)
	assert L==16 or L==19

	mat = scipy.io.loadmat("imagenet-vgg-verydeep-{0}.mat".format(L))

	mean_pixel = np.array([123.680, 116.779, 103.939], dtype=np.float32)

	current = tf.sub(current, tf.constant(mean_pixel, dtype=tf.float32))

	for i in range(mat["layers"][0].shape[0]):

		name = mat["layers"][0][i][0][0][0][0]
		kind = mat["layers"][0][i][0][0][1][0]

		if kind=="conv":
			kernel_dims = mat["layers"][0][i][0][0][3][0]

			print name, kernel_dims

			# matconvnet (W,H,C,K) where tensorflow (H,W,C,K)
			kernels = np.transpose(mat["layers"][0][i][0][0][2][0][0], axes=[1,0,2,3])
			bias    = mat["layers"][0][i][0][0][2][0][1].T[0]

			current = tf.nn.bias_add(
				tf.nn.conv2d(
					input  =current,
					filter =tf.Variable(kernels, dtype=tf.float32),
					strides=(1,1,1,1),
					padding="SAME"
				),
				tf.Variable(bias, dtype=tf.float32)
			)

		elif kind=="relu":
			current = tf.nn.relu(current)

		elif kind=="pool":
			current = tf.nn.max_pool(
				value  =current,
				ksize  =(1,2,2,1),
				strides=(1,2,2,1),
				padding="SAME"
			)


		if name==layer:
			break

	return current

if __name__ == '__main__':

	images = tf.placeholder(dtype=tf.float32, shape=(None,224,224,3))
	
	fc8 = load(16, images, layer="fc8")