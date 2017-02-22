# -*- coding: utf-8 -*-

import os
import re
import json
import pickle as pkl
import numpy as np
import threading
from config import PATH, train_test
from produce_dataset import is_valid_crop

from PIL import ImageFile
from scipy.misc import imread, imsave
ImageFile.LOAD_TRUNCATED_IMAGES = True

N_THREADS = 100

meta = {
	"train": {},
	"test": {},
}

class copyThread(threading.Thread):

	def __init__(self, I):
		threading.Thread.__init__(self)
		self.I = I
	def run(self):

		for i in self.I:

			path  = "/".join([PATH,labeled[i]])

			with open("{0}.json".format(path), "rb") as f:
				label = json.load(f)

			coords = [coord[::-1] for coord in label["heads"] if is_valid_crop(coord)]

			if coords:
				img = imread("{0}.jpg".format(path))
				if img.shape == (240, 320, 3):
					file_name = labeled[i].split("/")[-1].strip()
					meta[tt][file_name] = coords
					imsave("./images/{0}/{1}.jpg".format(tt, file_name), img)


if __name__ == '__main__':

	for tt in train_test.keys():

		labeled = []
		for f_name in train_test[tt].keys():

			print tt, f_name

			for f_name2 in os.listdir("{0}/{1}".format(PATH,f_name)):

				labeled += ["{0}/{1}/{2}".format(
					f_name, f_name2, re.match(r'([^\.]+)\.json',f).group(1)) \
				for f in os.listdir("{0}/{1}/{2}".format(PATH,f_name,f_name2)) \
				if re.match(r'.+json',f)]

		I = np.random.choice(range(len(labeled)),
			 size=len(labeled), replace=False)

		chunks  = np.array_split(I, N_THREADS)

		threads = [copyThread(chunk) for chunk in chunks]

		for t in threads:
			t.start()
		for t in threads:
			t.join()


	with open("./images/meta.pkl","wb") as f:
		pkl.dump(meta,f)

	print "Done!"