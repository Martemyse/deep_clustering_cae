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

layers_depths=[16, 32, 256, 6] #nujno bottleneck = stevilo clustrov
layer_depth_latent_cnn = 14
train_epochs = 100
batch_size = 256

# layers_depths=[30, 60, 512, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12
# train_epochs = 30
# batch_size = 256
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + \
    str(train_epochs) + "_" + str(batch_size)


Output = r"C:\Users\Martin-PC\Magistrska Matlab\Silhuette_clustering_analysis_re"
Output_filename = Output + ver_str + ".csv"

def autoencoderConv2D_1(input_shape= (48,432)):
    conv2Dfilters = [(3,5), (3,5)]
    BatchesSliceHeight = 48
    BatchesLength = 432
    
    inputs_1 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    slice_height = 12
    
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

   
autoencoder, encoder = autoencoderConv2D_1()
autoencoder.summary()

save_dir = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'

autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
# # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#%%

cfe_hits_raw = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_ch1_w11_raw_fl16_std_48_432.pkl")
# cfe_hits_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_ch2_w11_pure_fl16_std_48_432.pkl")
gfe_hits_raw = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_gfe_hits_ch1_w11_raw_fl16_std_48_432.pkl")
# gfe_hits_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_gfe_hits_ch2_w11_pure_fl16_std_48_432.pkl")
cfe_hits_razbr_raw = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_filtered_hits_razbr_ch1_w11_raw_fl16_std_48_432.pkl")

# cfe_filtered_raw = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_loc_fil_ch2_w11_raw_fl16_std_48_432.pkl")
# cfe_filtered_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_loc_fil_ch2_w11_pure_fl16_std_48_432.pkl")
# gfe_filtered_raw = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_gfe_loc_fil_ch2_w11_raw_fl16_std_48_432.pkl")
# gfe_filtered_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_gfe_loc_fil_ch2_w11_pure_fl16_std_48_432.pkl")

BatchesSliceHeight = 48
BatchesLength = 432
# num_val_data = 100

cfe_hits_raw_input = cfe_hits_raw.reshape(cfe_hits_raw.shape + (1,))
# cfe_hits_pure_input = cfe_hits_pure.reshape(cfe_hits_pure.shape + (1,))
gfe_hits_raw_input = gfe_hits_raw.reshape(gfe_hits_raw.shape + (1,))
# gfe_hits_pure_input = gfe_hits_pure.reshape(gfe_hits_pure.shape + (1,))
cfe_hits_razbr_raw_input = cfe_hits_razbr_raw.reshape(cfe_hits_razbr_raw.shape + (1,))

# cfe_filtered_raw_input = cfe_filtered_raw.reshape(cfe_filtered_raw.shape + (1,))
# cfe_filtered_pure_input = cfe_filtered_pure.reshape(cfe_filtered_pure.shape + (1,))
# gfe_filtered_raw_input = gfe_filtered_raw.reshape(gfe_filtered_raw.shape + (1,))
# gfe_filtered_pure_input = gfe_filtered_pure.reshape(gfe_filtered_pure.shape + (1,))

# x1_training_input_mixed = np.concatenate((cfe_hits_pure_input,gfe_hits_pure_input,cfe_hits_raw_input,gfe_hits_raw_input),axis=0)
# x2_training_input_pure = np.concatenate((cfe_hits_pure_input,gfe_hits_pure_input),axis=0)
cfe_gfe_input_raw = np.concatenate((cfe_hits_raw_input,gfe_hits_raw_input),axis=0)

batch_size_samples = 20000
index = 0
index_array = np.arange(cfe_gfe_input_raw.shape[0])

np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cfe_gfe_input_raw.shape[0])]
x1_training_input_pure_batch = cfe_gfe_input_raw[idx]
#%%

#% CAE
   
autoencoder, encoder = autoencoderConv2D_1()
autoencoder.summary()

save_dir = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'

autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
# # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#%


# autoencoder_train = autoencoder.fit(cfe_hits_pure_input, cfe_hits_pure_input, batch_size=batch_size, epochs=train_epochs,shuffle=True)
# autoencoder_train = autoencoder.fit(cfe_hits_raw_input, cfe_hits_raw_input, batch_size=batch_size, epochs=train_epochs,shuffle=True)
# autoencoder_train = autoencoder.fit(cfe_hits_raw_input, cfe_hits_raw_input, batch_size=batch_size, epochs=train_epochs,shuffle=True)
autoencoder_train = autoencoder.fit(x1_training_input_pure_batch, x1_training_input_pure_batch, batch_size=batch_size, epochs=train_epochs,shuffle=True, validation_data=(cfe_hits_razbr_raw_input,cfe_hits_razbr_raw_input))
save_dir_root = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'
save_dir = save_dir_root + '\yUpsilon_CAE_GFE_CFE_ch1' +  ver_str + '.h5'
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

#%% predict output test

number_of_predicted = 4
start_number = 1000

# predicted_x2_filtered_input = cfe_hits_pure_input[:number_of_predicted,:,:]
# for img in range(number_of_predicted):
#     predicted = autoencoder.predict([predicted_x2_filtered_input])
#     image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
#     fs = 5000000
#     dt = 1/fs
#     previous_time_data = np.arange(0,BatchesLength)*dt
#     freq = np.linspace(1, fs/5, BatchesSliceHeight)
#     plt.pcolormesh(previous_time_data, freq, cfe_hits_pure[img,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Validation Data, Input'+str(img))
#     plt.show()
#     plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Validation Data, Output'+str(img))
#     plt.show()
    
predicted_x2_filtered_input = cfe_hits_raw_input[start_number:start_number+number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs = 5000000
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    # freq = np.linspace(1, fs/5, BatchesSliceHeight)
    num_coef = 48
    freq = np.linspace(100000,450000, num_coef) 
    plt.pcolormesh(previous_time_data, freq, cfe_hits_raw[start_number+img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Input'+str(img))
    plt.show()
    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Training Data, Output'+str(img))
    plt.show()

#%% CAE load weights
# autoencoder, encoder = autoencoderConv2D_1()
# autoencoder.summary()

# train_epochs = 15
# batch_size = 512

# save_dir = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.load_weights(save_dir+'\Sigma_CAE_CFE_v5_200_epochs_1536_batch_8_latent.h5')

#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
x_pure_pred_enc_raw = encoder.predict(gfe_hits_raw_input)
# x_pure_pred_enc_raw = encoder.predict(cfe_filtered_raw_input)
x_pure_pred_enc = x_pure_pred_enc_raw
#%%
# er
classical_features_filtered_hits_ch1_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\gfe_hits_ch1_classical_features_fft_hanning_V1_sigma.pkl")
# classical_features_filtered_hits_ch2_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\gfe_hits_ch2_classical_features_fft_hanning_V1_sigma.pkl")
# classical_features_filtered_hits_ch1_cfe_gfe = pd.concat([classical_features_filtered_hits_ch1_cfe, classical_features_filtered_hits_ch2_gfe], axis=0)
# df_features_trai = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\filtered_hits_ch2_cfe.pkl")
# df_features_trai = df_features_trai.reset_index()
# classical_features_filtered_hits_ch1_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cfe_hits_ch2_classical_features.pkl")
# classical_features_filtered_hits_ch1_cfe_selected = classical_features_filtered_hits_ch1_cfe.iloc[:, [0,7]] # hits
# classical_features_filtered_hits_ch1_cfe_selected = classical_features_filtered_hits_ch1_cfe.iloc[:, [0,5]] # filtered
# classical_features_filtered_hits_ch1_cfe_selected = classical_features_filtered_hits_ch1_cfe.iloc[:, [0,1,2,3,4,5,6,7,8]]
# x_pure_pred_enc_df = pd.DataFrame(x_pure_pred_enc,columns = ['Emb_feature1','Emb_feature2','Emb_feature3','Emb_feature4','Emb_feature5'])
columns_vec = []
for i in range(1,layers_depths[3]+1):
    columns_vec.append('Emb'+str(i))
x_pure_pred_enc_df = pd.DataFrame(x_pure_pred_enc,columns = columns_vec)
x_pure_pred_enc_df.reset_index(drop=True)
classical_features_filtered_hits_ch1_cfe.reset_index(inplace=True, drop=True)
merged_features_embedded_classical = pd.concat([classical_features_filtered_hits_ch1_cfe, x_pure_pred_enc_df], axis=1)
merged_features_embedded_classical.astype('float32').dtypes

merged_features_embedded_classical = merged_features_embedded_classical.drop(merged_features_embedded_classical[merged_features_embedded_classical.FMXA_ch1 < 0.1].index)
save_dir_root = r'C:\Users\Martin-PC\Magistrska Matlab'
save_dir = save_dir_root + '\yUpsilon_GFE_merged_features_embedded_classical_ch1' +  ver_str + '.pkl'
pd.to_pickle(merged_features_embedded_classical, save_dir)
#%% excel       
# true_single_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_cwt_energy_relative.pkl")
true_double_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_cwt_energy_relative.pkl")
# true_single_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_fft_energy_relative.pkl")
# true_double_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_fft_energy_relative.pkl")
# true_single_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_single_cfe_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")
# true_double_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_double_cfe_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")

true_labels_cfe_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_cfe_hits_x_true_double_cwt_relative.pkl"))
# true_labels_gfe_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_gfe_hits_x_true_double_cwt_relative.pkl"))
labels_true_3_cfe_hits_cwt_pure = [int(str(x)[0]) for x in true_labels_cfe_filtered_hits_ch2_amp05_200_400_2500_w11_pure]
# labels_true_3_gfe_hits_cwt_pure = [int(str(x)[0]) for x in true_labels_gfe_hits_ch2_amp05_200_400_2500_w11_pure]
# labels_true_cfe_gfe = np.concatenate((labels_true_3_cfe_hits_cwt_pure, labels_true_3_gfe_hits_cwt_pure), axis = 0)

features_merged_labels = merged_features_embedded_classical.copy()
# features_merged_labels['true_single_cwt_rel'] = true_single_cwt
features_merged_labels['true_double_cwt_rel'] = labels_true_3_cfe_hits_cwt_pure
# features_merged_labels['true_single_fft_rel'] = true_single_fft
# features_merged_labels['true_double_fft_rel'] = true_double_fft
# features_merged_labels['true_single_cwt_abs'] = true_single_cwt_abs
# features_merged_labels['true_double_cwt_abs'] = true_double_cwt_abs
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
save_dir_root = r'C:\Users\Martin-PC\Magistrska Matlab'
save_dir = save_dir_root + '\Sigma_Xi_CFE_GFE_merged_features_embedded_classical_ch1' +  ver_str + '.xlsx'
features_merged_labels.to_excel(save_dir, index = False)  
