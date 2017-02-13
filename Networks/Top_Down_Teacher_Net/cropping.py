import numpy as np
import threading

CROP_HEIGHT, CROP_WIDTH = 60, 60
IMG_HEIGHT, IMG_WIDTH   = 240, 320

def crop(img, y, x):
	'''
	Crop an image.
	y, x: crop center coordinates.
	h, w: height and width of the crop.
	'''
	h, w = CROP_HEIGHT, CROP_WIDTH
	return img[int(y-h/2.):int(y+h/2.),int(x-w/2.):int(x+w/2.),:]

def batch_crop(imgs, coords):
	'''
	Crop a batch of images.
	'''
	labels = np.empty(shape=(len(imgs),))
	crops = np.empty(shape=(len(imgs), CROP_HEIGHT, CROP_WIDTH, 3))

	class Crop_Thread(threading.Thread):
		def __init__(self, i, coords):
			threading.Thread.__init__(self)
			self.i      = i
			self.coords = coords
		def run(self):

			valid_crop = False
			if self.coords:
				N    = len(self.coords)
				I    = np.random.choice(range(N), size=N)
				for j in I:
					x, y  = self.coords[j]
					if  y > CROP_HEIGHT/2. \
					and y < IMG_HEIGHT-1-CROP_HEIGHT/2. \
					and x > CROP_WIDTH/2. \
					and x < IMG_WIDTH-1-CROP_WIDTH/2.:
						valid_crop = True
						break

			if not valid_crop:
				y, x = np.random.uniform()*(IMG_HEIGHT-1), np.random.uniform()*(IMG_WIDTH-1)
				y, x = min(y, IMG_HEIGHT-1-(CROP_HEIGHT/2.)), min(x, IMG_WIDTH-1-(CROP_WIDTH/2.))
				y, x = max(y, CROP_HEIGHT/2.), max(x, CROP_WIDTH/2.)
				labels[self.i] = 0
			else:
				labels[self.i] = 1
			
			crops[self.i] = crop(imgs[self.i], int(y), int(x))

	threads = []
	for i in range(len(coords)):
		t = Crop_Thread(i, coords[i])
		threads.append(t)
		t.start()

	for t in threads:
		t.join()

	return crops, labels