import numpy as np
import pickle as pkl
import tensorflow as tf

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

# FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1_7_10/\
# know_dist_T5.0_beta0.05/"

FOLDER    = "../Top_Down_Student_Networks/saved/student_1_7_10/know_dist_T5.0_beta0.05/"

if __name__ == '__main__':

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("{0}student_1_7_10.meta".format(FOLDER))
		saver.restore(sess, tf.train.latest_checkpoint(FOLDER))
		layers = tf.get_collection('layers')

		labels, t_logits, images, keep_prob, training, augment,\
		prob, f_prob, train_step = layers

		# Bounding boxes and scores

		score  = tf.to_float(tf.count_nonzero(tf.to_int32(prob[:,:,:,1]>.8)))

		coords = tf.to_int32(tf.where(prob[:,:,:,1]>.8))

		boxes  = tf.map_fn(
			fn= lambda row: tf.stack((
				tf.maximum(0, row[0]-CROP_HEIGHT/2),
				tf.maximum(0, row[1]-CROP_WIDTH/2),
				tf.minimum(181, row[0]+CROP_HEIGHT/2),
				tf.minimum(241, row[1]+CROP_WIDTH/2)
			)),
			elems= coords
		)

		# Individual score inside every bounding box.
		scores = tf.map_fn(
			fn=lambda box: tf.to_float(tf.count_nonzero(
				tf.to_int32(tf.image.crop_to_bounding_box(
				prob[0,:,:,1:], box[0], box[1], CROP_HEIGHT, CROP_WIDTH
				)>.8))),
			elems=boxes,
			dtype=tf.float32
		)

		# Non-Maximum Supression
		nmx = tf.image.non_max_suppression(
			tf.to_float(boxes),
			scores,
			max_output_size=15
		)

		# img = imread("/notebooks/Data/top_down_view/images/img00053060.jpg")
		img = imread("../../Data/top_down_view/images/img00053060.jpg")

		heat_map = sess.run(prob, feed_dict={
			images: [img],
			keep_prob: 1,
			augment: False,
			training: False,
			})[0,:,:,1]

		print heat_map.shape

		with open("heat_map_example.pkl","wb") as f:
			pkl.dump(heat_map, f)

