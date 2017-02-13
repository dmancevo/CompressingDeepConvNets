# -*- coding: utf-8 -*-

import os
import re
import json
import pickle as pkl
import numpy as np
from PIL import ImageFile
from scipy.misc import imread
import threading

from config import PATH, train_test

ImageFile.LOAD_TRUNCATED_IMAGES = True

N_THREADS               = 32
CROP_HEIGHT, CROP_WIDTH = 60, 80
IMG_HEIGHT, IMG_WIDTH   = 240, 320

def is_valid_crop(coord):
	'''
	Check if it is possible to crop around the given coord.
	i.e. check that it is not too close to the margin.

	coord = (x, y)
	x: width pixel coord.
	y: hegith pixel coord.

	returns boolean
	'''
	x, y  = coord
	if  y > CROP_HEIGHT/2. \
	and y < IMG_HEIGHT-1-CROP_HEIGHT/2. \
	and x > CROP_WIDTH/2. \
	and x < IMG_WIDTH-1-CROP_WIDTH/2.:
		return True
	return False

def sample_random_coord():
	'''
	Sample a random (but valid) pair of coordinates to crop.

	returns x, y where y is the height coord and x is the width coord.
	'''
	y, x = np.random.uniform()*(IMG_HEIGHT-1), np.random.uniform()*(IMG_WIDTH-1)
	y, x = min(y, IMG_HEIGHT-1-(CROP_HEIGHT/2.)), min(x, IMG_WIDTH-1-(CROP_WIDTH/2.))
	y, x = max(y, CROP_HEIGHT/2.), max(x, CROP_WIDTH/2.)
	return (x,y)


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

class countingThread(threading.Thread):

	def __init__(self, I):
		threading.Thread.__init__(self)
		self.I = I
	def run(self):
		for i in self.I:
			path  = [PATH,labeled[i]]

			with open("{0}/{1}.json".format(*path), "rb") as f:
				label = json.load(f)

			valid=0
			for coord in label["heads"]:
				if is_valid_crop(coord):
					valid+=1

			C[i] = valid

class croppingThread(threading.Thread):

	def __init__(self, I, slots):
		threading.Thread.__init__(self)
		self.I     = I
		self.slots = slots
	def run(self):
		j = 0
		ones, zeros = 0, 0
		for i in self.I:
			path = [PATH,labeled[i]]

			with open("{0}/{1}.json".format(*path), "rb") as f:
				label = json.load(f)

			img = np.expand_dims(
				imread("{0}/{1}.jpg".format(*path)),
				axis=0
			)

			if label["heads"]:

				for coord in label["heads"]:

					if ones<=zeros and is_valid_crop(coord):
						labels[self.slots[j]] = 1
						crops[self.slots[j]]  = crop(img, coord)
						ones+=1
						j+=1

			elif zeros < ones:
				labels[self.slots[j]] = 0
				crops[self.slots[j]]  = crop(img, sample_random_coord())
				zeros+=1
				j+=1

			if len(self.slots)<=j: break

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

		threads = [countingThread(chunk) for chunk in chunks]
		C       = np.empty(shape=(len(I),))

		for t in threads:
			t.start()
		for t in threads:
			t.join()


		N       = int(2*np.sum(C))
		S       = np.array_split(range(N), N_THREADS)
		labels  = np.empty(shape=(N,), dtype=int)
		crops   = np.empty(shape=(N,CROP_HEIGHT,CROP_WIDTH,3))
		threads = [croppingThread(chunks[i], S[i]) for i in range(N_THREADS)]

		for t in threads:
			t.start()
		for t in threads:
			t.join()

		I      = np.any(crops,axis=(1,2,3))
		labels = labels[I]
		crops  = crops[I,:,:,:]

		print tt, crops.shape

		with open("{tt}.pkl".format(tt=tt),"wb") as f:
			pkl.dump((labels,crops),f)


	print "Done!"