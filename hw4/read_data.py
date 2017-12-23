import json
import skimage.io
import numpy as np
with open('my_tag/common.json', 'r') as f:
    common_dict = json.load(f)

# print(len(common_dict))
imgs = []
for key, value in common_dict.items():
    # print(key, value)
    img = skimage.io.imread('../hw4_dataset/64_faces/%s.jpg' % key)
    imgs.append([img])

imgs = np.vstack(imgs)
print(imgs.shape)
