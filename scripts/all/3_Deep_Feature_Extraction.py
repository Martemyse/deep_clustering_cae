# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:20:20 2022

@author: Martin-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:39:44 2021

@author: Martin-PC
"""


from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from keras import optimizers
import numpy as np
import pandas as pd
from time import time
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.pyplot as plt
import os, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import csv
from keras.preprocessing.image import ImageDataGenerator

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline = '') as output_file:
        output_writer = csv.writer(output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(list_of_elem)
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
#%

# layers_depths=[36, 72, 256, 5] #nujno bottleneck = stevilo clustrov
layers_depths=[40, 80, 512, 8] #nujno bottleneck = stevilo clustrov
# layers_depths=[4, 8, 56, 4] #nujno bottleneck = stevilo clustrov
layer_depth_latent_cnn = 16

# layers_depths=[30, 60, 512, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12



# Output = r"C:\Users\Martin-PC\Biocomposite Analysis\Silhuette_clustering_analysis_re"
# Output_filename = Output + ver_str + ".csv"

def autoencoderConv2D_1(input_shape= (80,288)):
    conv2Dfilters = [(3,5), (3,5)]
    BatchesSliceHeight = 80
    BatchesLength = 288

    inputs_1 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    slice_height = int(BatchesSliceHeight/4)
    
    split_1 = Lambda(lambda x : x[:,0:slice_height,:,:])(inputs_1)
    split_2  = Lambda(lambda x : x[:,slice_height:slice_height*2,:,:])(inputs_1)
    split_3  = Lambda(lambda x : x[:,slice_height*2:slice_height*3,:,:])(inputs_1)
    split_4  = Lambda(lambda x : x[:,slice_height*3:,:,:])(inputs_1)
    
    conv_1_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_1)
    pool_1_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_1_1)
    conv_1_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_1_1)
    pool_1_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_1_2)
    
    inputs_2 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_2_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_2)
    pool_2_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_2_1)
    conv_2_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_2_1)
    pool_2_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_2_2)
    
    inputs_3 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_3_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_3)
    pool_3_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_3_1)
    conv_3_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_3_1)
    pool_3_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_3_2)
    
    inputs_4 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_4_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_4)
    pool_4_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_4_1)
    conv_4_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_4_1)
    pool_4_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_4_2)
    
    # concatenate both feature layers and define output layer after some dense layers
    concat = Concatenate(axis=1)([pool_1_2, pool_2_2, pool_3_2, pool_4_2])
    conv_latent_1 = Conv2D(layer_depth_latent_cnn, (3,3), activation = 'relu', padding = "SAME", strides=(1, 1))(concat)
    pool_latent_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_latent_1)
    flat_latent_1  = Flatten()(pool_latent_1)
    dense_num = reduce((lambda x, y: x * y), (pool_latent_1.get_shape().as_list())[1:])
    dense_latent_1 = Dense(dense_num, activation = 'relu')(flat_latent_1)
    dense_latent_2 = Dense(layers_depths[2], activation = 'relu')(dense_latent_1)
    dense_latent_bottleneck = Dense(layers_depths[3], activation = 'relu')(dense_latent_2)
    dense_latent_4 = Dense(layers_depths[2], activation = 'relu')(dense_latent_bottleneck)
    dense_latent_5 = Dense(dense_num, activation = 'relu')(dense_latent_4)
    reshape_latent_1 = Reshape(tuple(pool_latent_1.get_shape().as_list())[1:])(dense_latent_5) # v isto kot pool_latent_1
    upsampling_latent_1= UpSampling2D((2, 2))(reshape_latent_1)
    conv_latent_2 = Conv2DTranspose(layer_depth_latent_cnn, (3,3), activation = 'relu', padding = "SAME", strides=(1, 1))(upsampling_latent_1)
    remerge_size = (pool_2_2.get_shape().as_list())[1]
    split_1 = Lambda(lambda x : x[:,0:remerge_size,:,:])(conv_latent_2)
    split_2  = Lambda(lambda x : x[:,remerge_size:remerge_size*2,:,:])(conv_latent_2)
    split_3  = Lambda(lambda x : x[:,remerge_size*2:remerge_size*3,:,:])(conv_latent_2)
    split_4  = Lambda(lambda x : x[:,remerge_size*3:,:,:])(conv_latent_2)
    
    dec_upsampling_1_1 = UpSampling2D((2, 2))(split_1)
    dec_tconv_1_1 = Conv2DTranspose(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_1_1)
    dec_upsampling_1_2 = UpSampling2D((2, 2))(dec_tconv_1_1)
    dec_tconv_1_2 = Conv2DTranspose(1, (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_1_2)
    
    dec_upsampling_2_1 = UpSampling2D((2, 2))(split_2)
    dec_tconv_2_1 = Conv2DTranspose(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_2_1)
    dec_upsampling_2_2 = UpSampling2D((2, 2))(dec_tconv_2_1)
    dec_tconv_2_2 = Conv2DTranspose(1, (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_2_2)
    
    dec_upsampling_3_1 = UpSampling2D((2, 2))(split_3)
    dec_tconv_3_1 = Conv2DTranspose(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_3_1)
    dec_upsampling_3_2 = UpSampling2D((2, 2))(dec_tconv_3_1)
    dec_tconv_3_2 = Conv2DTranspose(1, (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_3_2)
    
    dec_upsampling_4_1 = UpSampling2D((2, 2))(split_4)
    dec_tconv_4_1 = Conv2DTranspose(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_4_1)
    dec_upsampling_4_2 = UpSampling2D((2, 2))(dec_tconv_4_1)
    dec_tconv_4_2 = Conv2DTranspose(1, (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(dec_upsampling_4_2)
    
    output_1 = Concatenate(axis=1)([dec_tconv_1_2, dec_tconv_2_2, dec_tconv_3_2, dec_tconv_4_2])

    return Model(inputs=inputs_1, outputs=output_1, name='AE'), Model(inputs=inputs_1, outputs=dense_latent_bottleneck, name='encoder')

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = 4
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    # def call(self, inputs, **kwargs):
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

#%%
cwtms_ch1_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_ch1_80_288.pkl")
cwtms_ch2_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_ch2_80_288.pkl")
cwtms_ch1_t80 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\cwtms_ch1_80_288.pkl")
cwtms_ch2_t80 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\cwtms_ch2_80_288.pkl")
cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_cfe_ch2_80_288.pkl")
cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_gfe_ch2_80_288.pkl")

BatchesSliceHeight = 80
BatchesLength = 288

cwtms_ch1_tok_input = cwtms_ch1_tok.reshape(cwtms_ch1_tok.shape + (1,))
cwtms_ch2_tok_input = cwtms_ch2_tok.reshape(cwtms_ch2_tok.shape + (1,))
cwtms_ch1_t80_input = cwtms_ch1_t80.reshape(cwtms_ch1_t80.shape + (1,))
cwtms_ch2_t80_input = cwtms_ch2_t80.reshape(cwtms_ch2_t80.shape + (1,))
#%
cwtms_cfe = cwtms_cfe.reshape(cwtms_cfe.shape + (1,))
cwtms_gfe = cwtms_gfe.reshape(cwtms_gfe.shape + (1,))


cwtms_tok = np.concatenate((cwtms_ch1_tok_input,cwtms_ch2_tok_input),axis=0)
cwtms_t80 = np.concatenate((cwtms_ch1_t80_input,cwtms_ch2_t80_input),axis=0)
# cwtms_cfe_gfe = np.concatenate((cwtms_cfe_input,cwtms_gfe_input),axis=0)

# batch_size_samples = 20000
# index = 0
# index_array = np.arange(cwtms_t80.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cwtms_t80.shape[0])]
# cwtms_t80 = cwtms_t80[idx]
x1_training_input_batch = np.concatenate((cwtms_tok,cwtms_t80,cwtms_gfe,cwtms_cfe),axis=0)


batch_size_samples = 1000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_validation_input_batch = x1_training_input_batch[idx]

batch_size_samples = 31000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_training_input_batch = x1_training_input_batch[idx]


x1_training_input_batch = np.concatenate((cwtms_tok,x1_training_input_batch),axis=0)
x1_training_input_batch = np.concatenate((cwtms_tok,x1_training_input_batch),axis=0)
x1_training_input_batch = np.concatenate((cwtms_tok,x1_training_input_batch),axis=0)
x1_training_input_batch = np.concatenate((cwtms_tok,x1_training_input_batch),axis=0)

#%%
   
autoencoder, encoder = autoencoderConv2D_1()
autoencoder.summary()

save_dir = r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
#%%
train_epochs = 40
# batch_size = 64
# batch_size = 256
batch_size = 128
autoencoder_train = autoencoder.fit(x1_training_input_batch, 
                                    x1_training_input_batch,
                                    batch_size=batch_size, epochs=train_epochs,shuffle=True,
                                    validation_data=(x1_validation_input_batch, x1_validation_input_batch), # data for validation
                                    # steps_per_epoch=len(x1_training_input_batch) // (batch_size*2)
                                    )
#%%
number_of_predicted = 8
start_number = 100
    
predicted_x2_filtered_input = cwtms_ch2_tok_input[start_number:start_number+number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs=952381
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    # freq = np.linspace(1, fs/5, BatchesSliceHeight)
    num_coef = 80
    # freq = np.linspace(1, fs/2, num_coef) #ch2
    freq = np.linspace(1,203000, num_coef) 
    plt.pcolormesh(previous_time_data, freq, cwtms_ch2_tok[start_number+img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Input'+str(img))
    plt.show()
    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Output'+str(img))
    plt.show()
#%%
# cwtms = cwtms_ch1_tok_input[111:,:,:]
# train_epochs = 30
# batch_size = 64
# datagen = ImageDataGenerator(
#                                 featurewise_center=True,
#                                 # rotation_range=(0-10),
#                                 width_shift_range=0.1,
#                                height_shift_range=0.1,
#                                shear_range=0.1,
#                                zoom_range=0.1,
#                               horizontal_flip=True, 
#                               vertical_flip=False,
#                               # fill_mode='nearest',
#                               brightness_range=[1,1]
#                               )
# datagen.fit(cwtms)

# it = datagen.flow(cwtms, batch_size=1)

# # generate samples and plot
# for i in range(6):
#  	# define subplot
#  	plt.subplot(330 + 1 + i)
#  	# generate batch of images
#  	batch = it.next()
#  	# convert to unsigned integers for viewing
#  	image = batch[0].astype('uint8')
#  	# plot raw pixel data
#  	plt.imshow(image)
     
# #%%    
# autoencoder.fit(datagen.flow(x1_training_input_batch, x1_training_input_batch, batch_size=2), 
#                             epochs=train_epochs, # one forward/backward pass of training data
#                             shuffle=True,
#                             batch_size = batch_size,
#                             steps_per_epoch=len(x1_training_input_batch) // batch_size
#                     # steps_per_epoch=x_train.shape[0]//batch_size, # number of images comprising of one epoch
#                     # validation_data=(x_test, y_test), # data for validation
#                     # validation_steps=x_test.shape[0]//batch_size
#                     )

# number_of_predicted = 10
# start_number = 100

# predicted_x2_filtered_input = cwtms_ch2_tok_input[start_number:start_number+number_of_predicted,:,:]
# for img in range(number_of_predicted):
#     predicted = autoencoder.predict([predicted_x2_filtered_input])
#     image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
#     fs=952381
#     dt = 1/fs
#     previous_time_data = np.arange(0,BatchesLength)*dt
#     # freq = np.linspace(1, fs/5, BatchesSliceHeight)
#     num_coef = 80
#     # freq = np.linspace(1, fs/2, num_coef) #ch2
#     freq = np.linspace(1,203000, num_coef) 
#     plt.pcolormesh(previous_time_data, freq, cwtms_ch2_tok[start_number+img,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Training Data, Input'+str(img))
#     plt.show()
#     plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Training Data, Output'+str(img))
#     plt.show()
#%% spectre data augmentation
# save_dir =  r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'

# # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.load_weights(save_dir+'\yUpsilon_16_32_14_256_6_100_256_data_aug.h5')
# train_epochs = 50

# datagen = ImageDataGenerator(
#                               featurewise_center=True,
#                               rotation_range=(0-10),
#                               width_shift_range=0.1,
#                              height_shift_range=0.1,
#                              shear_range=0.1,
#                              zoom_range=0.1,
#                             horizontal_flip=True, 
#                             vertical_flip=False,
#                             # fill_mode='nearest',
#                             brightness_range=[0.7,1.3]
#                               )

# # it = datagen.flow(cwtms, batch_size=1)

# # # generate samples and plot
# # for i in range(6):
# # 	# define subplot
# # 	plt.subplot(330 + 1 + i)
# # 	# generate batch of images
# # 	batch = it.next()
# # 	# convert to unsigned integers for viewing
# # 	image = batch[0].astype('uint8')
# # 	# plot raw pixel data
# # 	plt.imshow(image)
    
# autoencoder.fit_generator(datagen.flow(x1_training_input_batch, x1_training_input_batch, batch_size=batch_size), 
#                     epochs=train_epochs, # one forward/backward pass of training data
#                     # steps_per_epoch=x_train.shape[0]//batch_size, # number of images comprising of one epoch
#                     # validation_data=(x_test, y_test), # data for validation
#                     # validation_steps=x_test.shape[0]//batch_size
#                     )

#%%
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'
save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
autoencoder.save_weights(save_dir)

#% Training CAE

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs_range = range(train_epochs)
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% load weighs
autoencoder, encoder = autoencoderConv2D_1()
autoencoder.summary()

# save_dir = r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 25
batch_size = 64
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'
# save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained_40_80_16_512_8_25_64' + '.h5'
autoencoder.load_weights(save_dir)
#%% load data predict
cwtms_ch1_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_ch1_80_288.pkl")
cwtms_ch2_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_ch2_80_288.pkl")
cwtms_ch1_t80 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\cwtms_ch1_80_288.pkl")
cwtms_ch2_t80 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\cwtms_ch2_80_288.pkl")
cwtms_cfe2_t100 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_cfe-100_80_288.pkl")


BatchesSliceHeight = 80
BatchesLength = 288

cwtms_ch1_tok_input = cwtms_ch1_tok.reshape(cwtms_ch1_tok.shape + (1,))
cwtms_ch2_tok_input = cwtms_ch2_tok.reshape(cwtms_ch2_tok.shape + (1,))
cwtms_ch1_t80_input = cwtms_ch1_t80.reshape(cwtms_ch1_t80.shape + (1,))
cwtms_ch2_t80_input = cwtms_ch2_t80.reshape(cwtms_ch2_t80.shape + (1,))
cwtms_cfe2_t100_input = cwtms_cfe2_t100.reshape(cwtms_cfe2_t100.shape + (1,))

cwtms_tok = np.concatenate((cwtms_ch1_tok_input,cwtms_ch2_tok_input),axis=0)
cwtms_t80 = np.concatenate((cwtms_ch1_t80_input,cwtms_ch2_t80_input),axis=0)

# batch_size_samples = 3000
# index = 0
# index_array = np.arange(cwtms_t80.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cwtms_t80.shape[0])]
# cwtms_t80 = cwtms_t80[idx]


#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x_pure_pred_enc = encoder.predict(cwtms_tok)

classical_features_filtered_hits_ch1_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\classical_features_ch1.pkl")
classical_features_filtered_hits_ch2_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\classical_features_ch2.pkl")
classical_features_filtered_hits_tok = pd.concat([classical_features_filtered_hits_ch1_cfe, classical_features_filtered_hits_ch2_cfe], axis=0)

#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x_pure_pred_enc = encoder.predict(cwtms_t80)

classical_features_filtered_hits_ch1_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\classical_features_ch1.pkl")
classical_features_filtered_hits_ch2_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\Treshold_Trigger_temp_minus80\classical_features_ch2.pkl")
classical_features_filtered_hits_tok = pd.concat([classical_features_filtered_hits_ch1_cfe, classical_features_filtered_hits_ch2_cfe], axis=0)
# classical_features_filtered_hits_tok = classical_features_filtered_hits_tok.iloc[idx]
#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x_pure_pred_enc = encoder.predict(cwtms_cfe2_t100_input)

classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\classical_features_cfe-100AE_Features_CFE-100opt.csv.pkl")
#%%
number_of_predicted = 8
start_number = 10000
    
predicted_x2_filtered_input = cwtms_cfe2_t100_input[start_number:start_number+number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs=952381
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    # freq = np.linspace(1, fs/5, BatchesSliceHeight)
    num_coef = 80
    # freq = np.linspace(1, fs/2, num_coef) #ch2
    freq = np.linspace(1,203000, num_coef) 
    plt.pcolormesh(previous_time_data, freq, cwtms_cfe2_t100[start_number+img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Input'+str(img))
    plt.show()
    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Output'+str(img))
    plt.show()
#%%
columns_vec = []
for i in range(1,layers_depths[3]+1):
    columns_vec.append('Emb'+str(i))
x_pure_pred_enc_df = pd.DataFrame(x_pure_pred_enc,columns = columns_vec)
x_pure_pred_enc_df.reset_index(drop=True)
x_pure_pred_enc_df.astype('float32').dtypes
classical_features_filtered_hits_tok.reset_index(inplace=True, drop=True)
merged_features_embedded_classical = pd.concat([classical_features_filtered_hits_tok, x_pure_pred_enc_df], axis=1)
# merged_features_embedded_classical.astype('float32').dtypes

# merged_features_embedded_classical = merged_features_embedded_classical.drop(merged_features_embedded_classical[merged_features_embedded_classical.FMXA_ch1 < 0.1].index)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite Analysis'
save_dir = save_dir_root + '\merged_features_embedded_classical_t80' +  ver_str + '.pkl'
pd.to_pickle(merged_features_embedded_classical, save_dir)
save_dir = save_dir_root + '\merged_features_embedded_classical_t80' +  ver_str + '.xlsx'
merged_features_embedded_classical.to_excel(save_dir, index = False) 

# merged_features_embedded_classical = pd.read_excel(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.xlsx")
# save_dir = save_dir_root + '\merged_features_embedded_classical_ch2' +  ver_str + '.pkl'
# pd.to_pickle(merged_features_embedded_classical, save_dir)
