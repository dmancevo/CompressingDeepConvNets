import numpy as np
import pickle as pkl
import tensorflow as tf

VGG        = 16
BATCH_SIZE = 100

batches = ["data_batch_{0}".format(i) for i in range(1,6)] + ["test_batch"]

teacher_output = {
	"train":{
		"logits": np.empty(shape=(0,1,1,10)),
		"labels": [],
		"data":   np.empty(shape=(0,3072)),
		},
	"test":{
		"logits": np.empty(shape=(0,1,1,10)),
		"labels": [],
		"data":   np.empty(shape=(0,3072)),
	}
}

with tf.Session() as sess:

	saver = tf.train.import_meta_graph("saved/VGG-{0}_CIFAR10.meta".format(VGG))
	saver.restore(sess, tf.train.latest_checkpoint('saved/'))
	layers = tf.get_collection('layers')

	print "Successfully loaded graph from file."

	row_images, labels, augment, keep_prob, logits, prob, loss, train_step = layers


	for batch in batches:

		print batch

		if batch[:4] == "data":
			tt = "train"
		else:
			tt = "test"

		with open("/notebooks/Data/cifar10/{0}".format(batch),"rb") as f:
			data = pkl.load(f)

		i,j = 0, BATCH_SIZE
		while j<= len(data["data"]):

			rows   = data["data"][i:j]
			labels = data["labels"][i:j]

			log    = sess.run(logits, feed_dict={
				row_images: rows,
				augment: False,
				keep_prob:  1.0,
				})

			teacher_output[tt]["labels"]+= labels
			teacher_output[tt]["logits"] = np.concatenate((teacher_output[tt]["logits"],
				log), axis=0)
			teacher_output[tt]["data"] = np.concatenate((teacher_output[tt]["data"],
				rows), axis=0)

			i += BATCH_SIZE
			j += BATCH_SIZE

with open("/notebooks/Data/cifar10/vgg{0}_cifar10.pkl".format(VGG),"wb") as f:
	pkl.dump(teacher_output, f)

print "Done!"