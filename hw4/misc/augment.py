import scipy.misc
import skimage
import skimage.io
import numpy as np
import glob

for filename in glob.glob('../../hw4_dataset/64_common_faces/*.jpg'): #assuming gif
    img = skimage.io.imread(filename)
    img = np.fliplr(img)
    filename = filename.split('/')[-1]
    scipy.misc.imsave('../../hw4_dataset/64_common_faces/_%s' % filename, img)
