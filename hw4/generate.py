from model import get_gen_normal
import numpy as np
import scipy.misc
import sys
import json
import os

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)

def gen_noise(batch_size=5, noise_shape=(1, 1, 100)):
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

def gen_specified_tags(hair, eyes, batch_size=5, embed_shape=(1, 1, 23)):

    if hair is None:
        hair = np.random.randint(0, 12)
    if eyes is None:
        eyes = np.random.randint(12, 23)

    tags = np.zeros((batch_size,embed_shape[-1]))
    tags[np.arange(batch_size), hair] = 1.0
    tags[np.arange(batch_size), eyes] = 1.0
    return tags.reshape((batch_size,)+(embed_shape))

def gen_test(hair, eyes):
    noise = gen_noise()
    fake_data_Xt = gen_specified_tags(batch_size, hair, eyes)
    fake_data_X = generator.predict([noise, fake_data_Xt])
    step_num = str(tot_step).zfill(4)
    save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png",True)

if __name__ == '__main__':

    with open('dict/id2hair.json') as f:
        id2hair = json.load(f)
        
    with open('dict/id2eyes.json') as f:
        id2eyes = json.load(f)

    with open('dict/hair2id.json') as f:
        hair2id = json.load(f)
        
    with open('dict/eyes2id.json') as f:
        eyes2id = json.load(f)

    img_save_dir = 'samples/'
    generator = get_gen_normal()
    generator.load_weights('generator.hdf5')

    with open(sys.argv[1]) as f:
        for line in f:
            np.random.seed(1489)
            test_index, feature = line.split(',')
            feature = feature.strip('\n').split(' ')
            hair = hair2id[feature[feature.index('hair') - 1] + ' hair'] if 'hair' in feature else None
            eyes = eyes2id[feature[feature.index('eyes') - 1] + ' eyes'] if 'eyes' in feature else None
            noise = gen_noise()
            tags = gen_specified_tags(hair, eyes)
            pics = generator.predict([noise, tags])
            for pic_index, pic in enumerate(pics, 1):
                picname = os.path.join(img_save_dir, 'sample_%s_%d.jpg' % (test_index, pic_index))
                scipy.misc.imsave(picname, denorm_img(pic))