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
# lambda_cwtms_bio_cfe_gfe_zeta_lower_rev_w4_32x512_v1_16_32_14_256_6_100_128
# layers_depths=[80, 160, 56, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[40, 80, 512, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[4, 8, 56, 4] #nujno bottleneck = stevilo clustrov
layers_depths=[16, 32, 128, 2]
layer_depth_latent_cnn = 14

# layers_depths=[30, 60, 512, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12


# Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Silhuette_clustering_analysis_re"
# Output_filename = Output + ver_str + ".csv"

def autoencoderConv2D_1(input_shape= (32,512)):
    layers_depths=[16, 32, 128, 2]
    layer_depth_latent_cnn = 12
    conv2Dfilters = [(3,3), (3,3)]
    BatchesSliceHeight = 32
    BatchesLength = 512
    stride_pool_h = 2
    stride_pool_w = 4
    max_pool_filter_h = 2
    max_pool_filter_w = 4

    stride_norm_2dconv= 1 #vcasih 3
    stride_latent_2dconv = 1 #vcasih 1
    stride_latent_max_pooling = 2 #vcasih 2

    inputs_1 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    slice_height = int(BatchesSliceHeight/4)

    split_1 = Lambda(lambda x : x[:,0:slice_height,:,:])(inputs_1)
    split_2  = Lambda(lambda x : x[:,slice_height:slice_height*2,:,:])(inputs_1)
    split_3  = Lambda(lambda x : x[:,slice_height*2:slice_height*3,:,:])(inputs_1)
    split_4  = Lambda(lambda x : x[:,slice_height*3:,:,:])(inputs_1)

    conv_1_1 = Conv2D(layers_depths[0], conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(split_1)
    pool_1_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_1_1)
    conv_1_2 = Conv2D(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(pool_1_1)
    pool_1_2 = MaxPooling2D(pool_size = (max_pool_filter_h,max_pool_filter_w), strides = (stride_pool_h,stride_pool_w))(conv_1_2)

    inputs_2 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_2_1 = Conv2D(layers_depths[0], conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(split_2)
    pool_2_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_2_1)
    conv_2_2 = Conv2D(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(pool_2_1)
    pool_2_2 = MaxPooling2D(pool_size = (max_pool_filter_h,max_pool_filter_w), strides = (stride_pool_h,stride_pool_w))(conv_2_2)

    inputs_3 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_3_1 = Conv2D(layers_depths[0], conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(split_3)
    pool_3_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_3_1)
    conv_3_2 = Conv2D(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(pool_3_1)
    pool_3_2 = MaxPooling2D(pool_size = (max_pool_filter_h,max_pool_filter_w), strides = (stride_pool_h,stride_pool_w))(conv_3_2)

    inputs_4 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    conv_4_1 = Conv2D(layers_depths[0], conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(split_4)
    pool_4_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_4_1)
    conv_4_2 = Conv2D(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(pool_4_1)
    pool_4_2 = MaxPooling2D(pool_size = (max_pool_filter_h,max_pool_filter_w), strides = (stride_pool_h,stride_pool_w))(conv_4_2)

    # concatenate both feature layers and define output layer after some dense layers
    
    concat = Concatenate(axis=1)([pool_1_2, pool_2_2, pool_3_2, pool_4_2])
    conv_latent_1 = Conv2D(layer_depth_latent_cnn, ((pool_4_2.get_shape().as_list())[1:][0],(pool_4_2.get_shape().as_list())[1:][1]), activation = 'relu', padding = "SAME", strides=(1, stride_latent_2dconv))(concat)
    pool_latent_1 = MaxPooling2D(pool_size = (2,2), strides = stride_latent_max_pooling)(conv_latent_1)
    flat_latent_1  = Flatten()(pool_latent_1)
    dense_num = reduce((lambda x, y: x * y), (pool_latent_1.get_shape().as_list())[1:])
    dense_latent_1 = Dense(dense_num, activation = 'relu')(flat_latent_1)
    dense_latent_2 = Dense(layers_depths[2], activation = 'relu')(dense_latent_1)
    dense_latent_bottleneck = Dense(layers_depths[3], activation = 'relu')(dense_latent_2)
    # Decoder
    dense_latent_bottleneck_input = Input(shape=(layers_depths[3],))
    dense_latent_4 = Dense(layers_depths[2], activation = 'relu')(dense_latent_bottleneck_input)
    dense_latent_5 = Dense(dense_num, activation = 'relu')(dense_latent_4)
    reshape_latent_1 = Reshape(tuple(pool_latent_1.get_shape().as_list())[1:])(dense_latent_5) # v isto kot pool_latent_1
    upsampling_latent_1= UpSampling2D((2, 2))(reshape_latent_1)
    conv_latent_2 = Conv2DTranspose(layer_depth_latent_cnn, ((pool_4_2.get_shape().as_list())[1:][0],(pool_4_2.get_shape().as_list())[1:][1]), activation = 'relu', padding = "SAME", strides=(1, stride_latent_2dconv))(upsampling_latent_1)
    remerge_size = (pool_2_2.get_shape().as_list())[1]
    split_1 = Lambda(lambda x : x[:,0:remerge_size,:,:])(conv_latent_2)
    split_2  = Lambda(lambda x : x[:,remerge_size:remerge_size*2,:,:])(conv_latent_2)
    split_3  = Lambda(lambda x : x[:,remerge_size*2:remerge_size*3,:,:])(conv_latent_2)
    split_4  = Lambda(lambda x : x[:,remerge_size*3:,:,:])(conv_latent_2)

    dec_upsampling_1_1 = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(split_1)
    dec_tconv_1_1 = Conv2DTranspose(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_1_1)
    dec_upsampling_1_2 = UpSampling2D((2, 2))(dec_tconv_1_1)
    dec_tconv_1_2 = Conv2DTranspose(1, conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_1_2)

    dec_upsampling_2_1 = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(split_2)
    dec_tconv_2_1 = Conv2DTranspose(layers_depths[1], conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_2_1)
    dec_upsampling_2_2 = UpSampling2D((2, 2))(dec_tconv_2_1)
    dec_tconv_2_2 = Conv2DTranspose(1, conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_2_2)

    dec_upsampling_3_1 = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(split_3)
    dec_tconv_3_1 = Conv2DTranspose(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_3_1)
    dec_upsampling_3_2 = UpSampling2D((2, 2))(dec_tconv_3_1)
    dec_tconv_3_2 = Conv2DTranspose(1, conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_3_2)

    dec_upsampling_4_1 = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(split_4)
    dec_tconv_4_1 = Conv2DTranspose(layers_depths[1], conv2Dfilters[1], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_4_1)
    dec_upsampling_4_2 = UpSampling2D((2, 2))(dec_tconv_4_1)
    dec_tconv_4_2 = Conv2DTranspose(1, conv2Dfilters[0], activation = 'relu', padding = "SAME", strides=(1, stride_norm_2dconv))(dec_upsampling_4_2)

    output_1 = Concatenate(axis=1)([dec_tconv_1_2, dec_tconv_2_2, dec_tconv_3_2, dec_tconv_4_2])
    
    encoder = Model(inputs=inputs_1, outputs=dense_latent_bottleneck, name='encoder')
    encoder.summary()
    decoder = Model(inputs=dense_latent_bottleneck_input, outputs=output_1, name='decoder')
    decoder.summary()
    # AE =  Model(inputs=inputs_1, outputs=output_1, name='AE')
    AE = Model(inputs=inputs_1, outputs=decoder(encoder(inputs_1)), name='autoencoder')
    
    return AE, encoder, decoder
# decoder = Model(inputs=dense_latent_bottleneck, outputs=output_1, name='decoder')
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
#%
   
autoencoder, encoder, decoder = autoencoderConv2D_1()
autoencoder.summary()

save_dir = r'C:\Users\Martin-PC\\Biocomposite_analysis_isndcm\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') #mse različn način kako se penalizira razlikovanje vhodnega in izhodnega spektograma
#%%
filename = '45-02s-13'
# filename = '45-02s-11'
cwtms_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
# cwtms_bio_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_512.pkl")
# cwtms_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_512.pkl")
# cwtms_cfe_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_512.pkl")
# cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_cfe-100_80_288.pkl")
# cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_gfe-20_80_288.pkl")

BatchesSliceHeight = 32
BatchesLength = 512

cwtms_bio_cold_input = cwtms_bio_cold.reshape(cwtms_bio_cold.shape + (1,))
# cwtms_bio_tok_input = cwtms_bio_tok.reshape(cwtms_bio_tok.shape + (1,))
# cwtms_cfe_cold_input = cwtms_cfe_cold.reshape(cwtms_cfe_cold.shape + (1,))
# cwtms_cfe_tok_input = cwtms_cfe_tok.reshape(cwtms_cfe_tok.shape + (1,))
#%
# cwtms_cfe_input = cwtms_cfe.reshape(cwtms_cfe.shape + (1,))
# cwtms_gfe_input = cwtms_gfe.reshape(cwtms_gfe.shape + (1,))


# cwtms_bio = np.concatenate((cwtms_bio_cold_input,cwtms_bio_tok_input),axis=0)
# cwtms_cfe = np.concatenate((cwtms_cfe_cold_input,cwtms_cfe_tok_input),axis=0)
# cwtms_cfe_gfe = np.concatenate((cwtms_cfe_input,cwtms_gfe_input),axis=0)

# batch_size_samples = 20000
# index = 0
# index_array = np.arange(cwtms_t80.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cwtms_t80.shape[0])]
# cwtms_t80 = cwtms_t80[idx]
# x1_training_input_batch = np.concatenate((cwtms_bio,cwtms_cfe),axis=0)
x1_training_input_batch = cwtms_bio_cold_input

batch_size_samples = 1000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_validation_input_batch = x1_training_input_batch[idx]

batch_size_samples = 15000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_training_input_batch = x1_training_input_batch[idx]

#bio cold training
# batch_size_samples = 32000
# index = 0
# index_array = np.arange(cwtms_bio_cold_input.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cwtms_bio_cold_input.shape[0])]
# x1_training_input_batch = cwtms_bio_cold_input[idx]

#%%
# train_epochs = 100
train_epochs = 30
# batch_size = 20
# batch_size = 256
batch_size = 128
# batch_size = 64
# batch_size = 16
# batch_size = 32
autoencoder_train = autoencoder.fit(x1_training_input_batch, 
                                    x1_training_input_batch,
                                    batch_size=batch_size, epochs=train_epochs,shuffle=True,
                                    validation_data=(x1_validation_input_batch, x1_validation_input_batch), # data for validation
                                    # steps_per_epoch=len(x1_training_input_batch) // (batch_size*2)
                                    )
#%%
number_of_predicted = 6
# start_number = 1242
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_512.pkl")

start_number = 500
cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))

# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_bio_tok_scaled_32_128.pkl")
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_cfe_cold_scaled_32_128.pkl")
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_cfe_tok_scaled_32_128.pkl")
cwtms_cfe_test_validation_input = cwtms_cfe_test_validation.reshape(cwtms_cfe_test_validation.shape + (1,))    

predicted_x2_filtered_input = cwtms_cfe_test_validation[start_number:start_number+number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs=952381
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    # freq = np.linspace(1, fs/5, BatchesSliceHeight)
    # BatchesSliceHeight = 32
    # freq = np.linspace(1, fs/2, BatchesSliceHeight) #ch2
    freq = np.linspace(1, fs, BatchesSliceHeight) 
    # freq = np.linspace(1,203000, num_coef) 
    plt.pcolormesh(previous_time_data, freq, cwtms_cfe_test_validation[start_number+img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Input'+str(img))
    plt.show()
    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Output'+str(img))
    plt.show()
    
x_pure_pred_enc = encoder.predict(predicted_x2_filtered_input)
#%% save weights
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
save_dir = save_dir_root + '\isndcm_seperate_decoder_lambda_32x512_{filename}_v2'.format(filename=filename) +  ver_str + '.h5'
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

#%% load weights
autoencoder, encoder, decoder = autoencoderConv2D_1()
autoencoder.summary()

# save_dir = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 10
batch_size = 20
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
save_dir = save_dir_root + '\\isndcm_seperate_decoder_lambda_32x512_45-02s-13_v1__16_32_14_256_6_50_64' + '.h5'
autoencoder.load_weights(save_dir)
#%% load data predict
cwtms_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
# cwtms_bio_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_512.pkl")
# cwtms_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_512.pkl")
# cwtms_cfe_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_512.pkl")
# cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_cfe-100_80_288.pkl")
# cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_gfe-20_80_288.pkl")

BatchesSliceHeight = 32
BatchesLength = 512

cwtms_bio_cold_input = cwtms_bio_cold.reshape(cwtms_bio_cold.shape + (1,))
# cwtms_bio_tok_input = cwtms_bio_tok.reshape(cwtms_bio_tok.shape + (1,))
# cwtms_cfe_cold_input = cwtms_cfe_cold.reshape(cwtms_cfe_cold.shape + (1,))
# cwtms_cfe_tok_input = cwtms_cfe_tok.reshape(cwtms_cfe_tok.shape + (1,))

#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x_pure_pred_enc = encoder.predict(cwtms_bio_cold)

classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename))
#%
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
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm'
save_dir = save_dir_root + '\merged_features_lambda_seperate_dec_emb_cla_w7_{filename}_32x512'.format(filename=filename) +  ver_str + '.pkl'
pd.to_pickle(merged_features_embedded_classical, save_dir)
# save_dir = save_dir_root + '\merged_features_embedded_classical_rho20_lambda_cfe_cold' +  ver_str + '.xlsx'
# merged_features_embedded_classical.to_excel(save_dir, index = False) 

# merged_features_embedded_classical = pd.read_excel(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.xlsx")
# save_dir = save_dir_root + '\merged_features_embedded_classical_ch2' +  ver_str + '.pkl'
# pd.to_pickle(merged_features_embedded_classical, save_dir)
# cfe_cold


#%% CLustering Layer Initialization

# autoencoder.load_weights(save_dir+'\cae_v2_weights_40_10_34_434.h5')
# autoencoder, encoder = autoencoderConv2D_1()
# autoencoder.summary()

# train_epochs = 15
# batch_size = 512

# save_dir = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.load_weights(save_dir+'\Clustering_CAE_v4_vandetta_3_6_12_30_3_128.h5')

n_clusters = 3
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
for layer in model.layers[:]:
        layer.trainable = True
        
# optimizer = optimizers.Adam(learning_rate=0.000001,beta_1=0.9,beta_2=0.999,epsilon=1e-07, amsgrad=False)

model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

# x_pure_input_scaled_pca = features_ch2_pca
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
x_pure_pred_enc = encoder.predict(x3_pure_input)
y_pred = kmeans.fit_predict(x_pure_pred_enc)
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
model.summary()
model_weights = model.get_weights()

# x_pure_input
# x2a
# x_pure_pred_enc
# output = x_pure_pred_enc

loss = 0
index = 0
maxiter = 5000
update_interval = 140
batch_size = 1024
index_array = np.arange(x3_pure_input.shape[0])
tol = 0.001 # tolerance threshold to stop training
labels_true_3_ctw_w12 = labels_true_3_ctw_w12[0:len(x3_pure_input)]

x_input = x3_pure_input # ali x3a --> vsi
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=22).output)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        output = model.predict(x3_pure_input, verbose=0)
        output_distribution = target_distribution(output)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        # output from intermediate layer
        intermediate_output = intermediate_layer_model.predict(x3_pure_input)  
        y_pred = output.argmax(1)
        y_pred_auxilary = output_distribution.argmax(1)
        cluster_nums = [3]

        for cluster_num in cluster_nums:
            # clustering_layer_kmeans_model = KMeans(n_clusters=cluster_num, random_state=1).fit(intermediate_output)
            # labels_kmeans = clustering_layer_kmeans_model.labels_
            if cluster_num == 3:
                fig = plt.figure(1, figsize=(7, 7))
                plt.clf()
                ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
                fig.add_axes(ax)
                plt.cla()
                for name, label in [('Category1', 0), ('Category2', 1), ('Category3', 2)]:
                    a = 0
                # labels_clusters3 = np.choose(labels_kmeans, [0, 2, 1]).astype(float)
                ax.scatter(intermediate_output[:, 0], intermediate_output[:, 1], intermediate_output[:, 2], c=y_pred)
                ax.w_xaxis.set_ticklabels([])
                ax.w_yaxis.set_ticklabels([])
                ax.w_zaxis.set_ticklabels([])
                plt.title('3 clusters K-means')
                plt.show() 
                
        for cluster_num in cluster_nums:
            if cluster_num == 3:
                fig = plt.figure(1, figsize=(7, 7))
                plt.clf()
                ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
                fig.add_axes(ax)
                plt.cla()
                for name, label in [('Category1', 0), ('Category2', 1), ('Category3', 2)]:
                    a = 0
                # labels_clusters3 = np.choose(labels_kmeans, [0, 2, 1]).astype(float)
                ax.scatter(intermediate_output[:, 0], intermediate_output[:, 1], intermediate_output[:, 2], c=labels_true_3_ctw_w12)
                ax.w_xaxis.set_ticklabels([])
                ax.w_yaxis.set_ticklabels([])
                ax.w_zaxis.set_ticklabels([])
                plt.title('3 clusters True')
                plt.show() 
                
        if labels_true_3_ctw_w12 is not None:
            acc = np.round(accuracy_score(labels_true_3_ctw_w12, y_pred), 5)
            nmi = np.round(normalized_mutual_info_score(labels_true_3_ctw_w12, y_pred), 5)
            ari = np.round(adjusted_rand_score(labels_true_3_ctw_w12, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    # if index == 0:
    #     np.random.shuffle(index_array)
    idx = index_array[index * batch_size: min((index+1) * batch_size, x_input.shape[0])]
    loss = model.train_on_batch(x=x_input[idx], y=output_distribution[idx])
    index = index + 1 if (index + 1) * batch_size <= x_input.shape[0] else 0

# model.save_weights(save_dir + '\clustering_v2_weights_30_3_34_434.h5')
# model.load_weights(save_dir + '\clustering_v2_weights_30_3_34_434.h5')

#%% Eval.

output = model.predict(x2a, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
y = y[0:len(x2a)]
if y is not None:
    acc = np.round(accuracy_score(y, y_pred), 5)
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

sns.set(font_scale=3)
confusion_matrix = confusion_matrix(y, y_pred)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
#%% predicting average cwtm of cluster classes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

filename = filename.replace('-','_')
# mean_of_deep_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\mean_features_df_{filename}_16_32_14_256_6_100_256.pkl".format(filename=filename))
mean_of_deep_features = pd.read_excel(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\mean_features_df_lambda_seperate_dec_emb_cla_{filename}_16_32_14_256_6_100_256.xlsx".format(filename=filename))
#%
deep_columns = [x for x in mean_of_deep_features.columns if 'Emb' in x]
mean_of_deep_features = mean_of_deep_features[deep_columns]
x_pure_pred_dec = decoder.predict(np.array(mean_of_deep_features))

number_of_clusters = 4
start_number = 0


for img in range(number_of_clusters):
    image_decropped = x_pure_pred_dec.reshape((number_of_clusters,BatchesSliceHeight,BatchesLength))
    fs=952381
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    # freq = np.linspace(1, fs/5, BatchesSliceHeight)
    # BatchesSliceHeight = 32
    # freq = np.linspace(1, fs/2, BatchesSliceHeight) #ch2
    freq = np.linspace(1, fs, BatchesSliceHeight) 

    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Output'+str(img))
    plt.show()