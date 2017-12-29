
# coding: utf-8

# In[1]:


import os
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
from keras.layers import Dense
from keras.layers import Reshape, Concatenate
from keras.layers.core import Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')
from collections import deque


# In[2]:


np.random.seed(4678)


# In[3]:


def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 


# In[4]:


import json
with open('dict/common.json') as f:
    common_dict = json.load(f)
print(len(common_dict))

with open('dict/hair2id.json') as f:
    hair2id = json.load(f)
print(len(hair2id))
    
with open('dict/eyes2id.json') as f:
    eyes2id = json.load(f)
print(len(eyes2id))

with open('dict/id2hair.json') as f:
    id2hair = json.load(f)
print(len(hair2id))
    
with open('dict/id2eyes.json') as f:
    id2eyes = json.load(f)


# In[5]:


def get_tag(filename, hair_size=12, eyes_size=11):
    tag = filename.split('/')[-1].strip('_')[:-4]
    embed = np.zeros(hair_size+eyes_size)
    hair, eyes = common_dict[tag]
    hair = hair2id[hair]
    eyes = eyes2id[eyes]
    embed[hair] = 1.0
    embed[eyes+hair_size] = 1.0
    return embed


# In[6]:


def sample_from_dataset(batch_size, image_shape, data_dir=None, data = None, embed_shape=(1, 1, 23)):
    sample_dim = (batch_size,) + image_shape
    embed_dim = (batch_size,) + embed_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    embed = np.empty(embed_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        #print(image.size)
        #image.thumbnail(image_shape[:-1], Image.ANTIALIAS) - this maintains aspect ratio ; we dont want that - we need m x m size
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        #print(image.size)
        #print('\n')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
        embed[index,...] = get_tag(img_filename)
    return sample, embed


# In[7]:


def gen_noise(batch_size, noise_shape):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)


# In[8]:


def gen_tags(batch_size, embed_shape=(1, 1, 23), hair_size=12, eyes_size=11):
    tags = np.zeros((batch_size,embed_shape[-1]))
    hair = np.random.randint(0, hair_size, size=batch_size)
    eyes = np.random.randint(hair_size, hair_size+eyes_size, size=batch_size)
    tags[np.arange(batch_size), hair] = 1.0
    tags[np.arange(batch_size), eyes] = 1.0
    return tags.reshape((batch_size,)+(embed_shape))


# In[9]:


def generate_images(generator, save_dir):
    noise = gen_noise(batch_size,noise_shape)
    #using noise produced by np.random.uniform - the generator seems to produce same image for ANY noise - 
    #but those images (even though they are the same) are very close to the actual image - experiment with it later.
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)
    plt.show()


# In[10]:


def save_img_batch(img_batch,img_save_dir,toShow):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)
    #print(rand_indices)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    plt.close('all')
    if toShow:
        plt.show()   


# In[11]:


def get_gen_normal(noise_shape, tag_shape):
    noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next
    tag_input = Input(shape = tag_shape)
    generator = Concatenate()([gen_input, tag_input])
    #gen_input = Input(shape = [noise_shape]) #if want to use with dense layer next
    
    generator = Conv2DTranspose(filters = 2048, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
        
    #generator = bilinear2x(generator,256,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 256, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 1024, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,128,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 128, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,64,kernel_size=(4,4))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 64, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)    
    generator = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    generator = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    
    #generator = bilinear2x(generator,3,kernel_size=(3,3))
    #generator = UpSampling2D(size=(2, 2))(generator)
    #generator = SubPixelUpscaling(scale_factor=2)(generator)
    #generator = Conv2D(filters = 3, kernel_size = (4,4), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Activation('tanh')(generator)
        
    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input = [gen_input, tag_input], output = generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    generator_model.summary()

    return generator_model
    
#------------------------------------------------------------------------------------------

def get_disc_normal(image_shape=(64,64,3), tag_shape=(1,1,23)):
    image_shape = image_shape
    
    dropout_prob = 0.4
    
    #kernel_init = RandomNormal(mean=0.0, stddev=0.01)
    kernel_init = 'glorot_uniform'
    
    dis_input = Input(shape = image_shape)
    
    discriminator = Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 1024, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    #discriminator = Dropout(dropout_prob)(discriminator)
    discriminator = Conv2D(filters = 2048, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    #discriminator = MaxPooling2D(pool_size=(2, 2))(discriminator)
    
    tag_input = Input(shape = tag_shape)
    tag_vec = Lambda(lambda x: K.tile(x, [1, 4, 4, 1]))(tag_input)
    discriminator = Concatenate()([discriminator, tag_vec])

    # discriminator = Flatten()(discriminator)

    discriminator = Conv2D(filters = 2048, kernel_size = (1,1), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    
    discriminator = Conv2D(filters = 1, kernel_size = (4,4), strides = (4,4), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    # discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    # discriminator = LeakyReLU(0.2)(discriminator)

    # discriminator = MinibatchDiscrimination(100,5)(discriminator)
    # discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)
    # discriminator = Lambda(lambda x: K.expand_dims(K.expand_dims(x, 0), 0))(discriminator)

    # discriminator = Conv2D(filters = 1, kernel_size = (4,4), strides = (4,4), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    # discriminator = BatchNormalization(momentum = 0.5)(discriminator)
    # discriminator = LeakyReLU(0.2)(discriminator)

    # discriminator = Activation('tanh')(discriminator)
    
    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input = [dis_input, tag_input], output = discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])
    discriminator_model.summary()
    return discriminator_model


# In[13]:


noise_shape = (1,1,100)
num_steps = 40000
batch_size = 64
image_shape = None
img_save_dir = "revise_huge/"
save_model = True


# In[14]:


image_shape = (64,64,3)
tag_shape = (1, 1, 23)
data_dir =  "../hw4_dataset/64_common_faces/*.jpg"
log_dir = img_save_dir
save_model_dir = img_save_dir
discriminator = get_disc_normal(image_shape, tag_shape)
generator = get_gen_normal(noise_shape, tag_shape)
# discriminator.load_weights('save5/39999_DISCRIMINATOR_weights_and_arch.hdf5')
# generator.load_weights('save5/39999_GENERATOR_weights_and_arch.hdf5')


# In[15]:


# generator.load_weights('save4/'+str(9999)+"_GENERATOR_weights_and_arch.hdf5")
# discriminator.load_weights('save4/'+str(9999)+"_DISCRIMINATOR_weights_and_arch.hdf5")


# In[16]:


discriminator.trainable = False
opt = Adam(lr=0.00015, beta_1=0.5) #same as gen
gen_inp = Input(shape=noise_shape)
tag_inp = Input(shape=tag_shape)
GAN_inp = generator([gen_inp, tag_inp])
GAN_opt = discriminator([GAN_inp, tag_inp])
gan = Model(input = [gen_inp, tag_inp], output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()


# In[17]:


def gen_specified_tags(batch_size, hair, eyes, embed_shape=(1, 1, 23), hair_size=12, eyes_size=11):
    tags = np.zeros((batch_size,embed_shape[-1]))
#     hair = np.random.randint(0, hair_size, size=batch_size)
#     eyes = np.random.randint(hair_size, hair_size+eyes_size, size=batch_size)
    tags[np.arange(batch_size), hair] = 1.0
    tags[np.arange(batch_size), eyes] = 1.0
    return tags.reshape((batch_size,)+(embed_shape))


def gen_test(hair, eyes, tot_step=0):
    noise = gen_noise(batch_size,noise_shape)
#     print(noise)
    print(id2hair[str(hair)], id2eyes[str(eyes)])
    fake_data_Xt = gen_specified_tags(batch_size, hair, eyes)
    fake_data_X = generator.predict([noise, fake_data_Xt])
    step_num = str(tot_step).zfill(4)
    save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png",True)

# In[18]:


avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)


# In[ ]:


for step in range(num_steps): 
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time() 
    
    real_data_X, real_data_Xt = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)
#     real_data_Xt = real_data_Xt.reshape((batch_size,) + (1, 1, 23) )
    
    noise = gen_noise(batch_size,noise_shape)
    fake_data_Xt = gen_tags(batch_size, tag_shape)
#     print(tags[0])
    
#     fake_data_Xt = np.zeros(23)
#     hair, eyes = rd.randint(0, 11), rd.randint(12, 22)
#     fake_data_Xt[hair] = 1.0
#     fake_data_Xt[eyes] = 1.0
    fake_data_X = generator.predict([noise, fake_data_Xt])
#     print(real_data_X.shape, fake_data_X.shape)
    
    if (tot_step % 10) == 0:
        step_num = str(tot_step).zfill(5)
        save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png",False)
    if (tot_step % 100) == 0:
        gen_test(0, 0, "test"+str(tot_step))

        
    data_X = np.concatenate([real_data_X,fake_data_X])
#     print(real_data_Xt.shape, fake_data_Xt.shape)
    data_Xt = np.concatenate([real_data_Xt,fake_data_Xt])
    
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    
    fake_data_Y = np.random.random_sample(batch_size)*0.2
    
     
    data_Y = np.concatenate((real_data_Y,fake_data_Y))
    
        
    discriminator.trainable = True
    generator.trainable = False
    real_data_Y = real_data_Y.reshape((-1, 1, 1, 1))
    fake_data_Y = fake_data_Y.reshape((-1, 1, 1, 1))
    
#     print(real_data_X.shape, real_data_Xt.shape, real_data_Y.shape)
    
    dis_metrics_real = discriminator.train_on_batch([real_data_X, real_data_Xt],real_data_Y)   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch([fake_data_X, real_data_Xt],fake_data_Y)   #training seperately on fake
    
    print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
    
    
    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])
    
    generator.trainable = True

    GAN_X = gen_noise(batch_size,noise_shape)
    GAN_Xt = gen_tags(batch_size, tag_shape)

    GAN_Y = real_data_Y
    
    discriminator.trainable = False
    
    gan_metrics = gan.train_on_batch([GAN_X, GAN_Xt], GAN_Y)
    print("GAN loss: %f" % (gan_metrics[0]))
    
    text_file = open(log_dir+"\\training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
    
        
    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))
    
    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        discriminator.trainable = True
        generator.trainable = True
        generator.save(save_model_dir+str(tot_step)+"_GENERATOR_weights_and_arch.hdf5")
        discriminator.save(save_model_dir+str(tot_step)+"_DISCRIMINATOR_weights_and_arch.hdf5")


# In[19]:





# In[20]:


def gen_random_test(tot_step=0):
    noise = gen_noise(batch_size,noise_shape)
    fake_data_Xt = gen_tags(batch_size)
    fake_data_X = generator.predict([noise, fake_data_Xt])
    step_num = str(tot_step).zfill(4)
    save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png",True)


# In[ ]:


# generator.load_weights('save3/'+str(9999)+"_GENERATOR_weights_and_arch.hdf5")


# In[22]:


# for i in range(1, 201):
#     generator.load_weights('save4/'+str(i*500-1)+"_GENERATOR_weights_and_arch.hdf5")
#     gen_test(11, 0, i)

