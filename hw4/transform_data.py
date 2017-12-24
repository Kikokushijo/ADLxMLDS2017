import scipy.misc
import skimage
import skimage.io
import skimage.transform
import json

with open('my_tag/common.json', 'r') as f:
    common_dict = json.load(f)
# print(len(common_dict))
# face = misc.imread('../hw4_dataset/faces/1.jpg')
# print(face.shape, face.dtype)

for key in range(33431):
    # print(key)
    img = skimage.io.imread('../hw4_dataset/faces/%s.jpg' % key)
    img = skimage.transform.resize(img, (64, 64))
    # print(img.shape)
    scipy.misc.imsave('../hw4_dataset/64_faces/%s.jpg' % key, img)

