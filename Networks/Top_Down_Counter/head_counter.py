import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import r2_score

from PIL import ImageFile
from scipy.misc import imread
ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320


FOLDER    = "/notebooks/Networks/Top_Down_Student_Networks/saved/student_1/\
know_dist_T10.0_beta0.05/"

DATA_PATH = "/notebooks/Data/top_down_view/images/"

if __name__ == '__main__':

	with open("{0}/meta.pkl".format(DATA_PATH), "rb") as f:
		meta = pkl.load(f)

	N_train, N_test = len(meta["train"]), len(meta["test"])

	trn_file_names = np.array(meta["train"].keys())
	trn_N          = np.array([len(meta["train"][fn]) for fn in trn_file_names])

	tst_file_names = np.array(meta["test"].keys())
	tst_N          = np.array([len(meta["test"][fn]) for fn in tst_file_names])

	with tf.Session() as sess:

		saver = tf.train.import_meta_graph("{0}student_1.meta".format(FOLDER))
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

		train_test = (
			(trn_N, trn_file_names, "train"),
			(tst_N, tst_file_names, "test")
		)

		for tt_N, tt_file_names, tt in train_test:

			J, n_boxes, S = [], [], []
			for j in range(len(tt_N)):

				file_name = tt_file_names[j]
				imgs      = [imread("{0}/{1}/{2}.jpg".format(DATA_PATH, tt, file_name))]

				n_boxes.append(len(sess.run(nmx, feed_dict={
					images: imgs,
					keep_prob: 1.,
					augment: False,
					training: False
				})))

				S.append(sess.run(score, feed_dict={
					images: imgs,
					keep_prob: 1.,
					augment: False,
					training: False
				}))

				J.append(j)

				if j%100==0: print j

			with open("{0}.pkl".format(tt),"wb") as f:
				pkl.dump((tt_N[J],  n_boxes, S), f)