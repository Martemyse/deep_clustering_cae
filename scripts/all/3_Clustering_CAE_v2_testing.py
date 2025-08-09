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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
#%%

def autoencoderConv2D_1(input_shape= (40,432)):
        
    layers_depths=[3, 6, 30, 3] #nujno bottleneck = stevilo clustrov
    conv2Dfilters = [(3,5), (3,5)]
    BatchesSliceHeight = 32
    BatchesLength = 512
    
    inputs_1 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    
    # split_1 = Lambda(lambda x : x[:,0:8,:,:])(inputs_1)
    # split_2  = Lambda(lambda x : x[:,8:16,:,:])(inputs_1)
    # split_3  = Lambda(lambda x : x[:,16:,:,:])(inputs_1)
    
    conv_1_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_1)
    pool_1_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_1_1)
    conv_1_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_1_1)
    pool_1_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_1_2)
    
    # inputs_2 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    # conv_2_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_2)
    # pool_2_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_2_1)
    # conv_2_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_2_1)
    # pool_2_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_2_2)
    
    # inputs_3 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
    # conv_3_1 = Conv2D(layers_depths[0], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(split_3)
    # pool_3_1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_3_1)
    # conv_3_2 = Conv2D(layers_depths[1], (3,5), activation = 'relu', padding = "SAME", strides=(1, 3))(pool_3_1)
    # pool_3_2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv_3_2)
    
    
    # concatenate both feature layers and define output layer after some dense layers
    # concat = Concatenate(axis=1)([pool_1_2, pool_2_2, pool_3_2])
    conv_latent_1 = Conv2D(12, (3,3), activation = 'relu', padding = "SAME", strides=(1, 1))(pool_1_2)
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
    conv_latent_2 = Conv2DTranspose(12, (3,3), activation = 'relu', padding = "SAME", strides=(1, 1))(upsampling_latent_1)
    remerge_size = (pool_2_2.get_shape().as_list())[1]
    split_1 = Lambda(lambda x : x[:,0:remerge_size,:,:])(conv_latent_2)
    split_2  = Lambda(lambda x : x[:,remerge_size:remerge_size*2,:,:])(conv_latent_2)
    split_3  = Lambda(lambda x : x[:,remerge_size*2:,:,:])(conv_latent_2)
    
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
    
    output_1 = Concatenate(axis=1)([dec_tconv_1_2, dec_tconv_2_2, dec_tconv_3_2])

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
        input_dim = 3
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
x1_hits = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_ch2_w11_raw_fl16_std_40_432.pkl")
# y = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_true_3.pkl")
x2_filtered = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_loc_filtered_ch2_w11_raw_fl16_std_40_432.pkl")
x3_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_loc_filtered_ch2_w11_pure_fl16_std_40_432.pkl")
# x_validation = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_razbr_val_loc_filtered_ch2_w12_pure_f16_36_434.pkl")
x_validation = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_razbr_ch2_w11_pure_fl16_std_40_432.pkl")
x5_pure_hits = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_ch2_w11_pure_fl16_std_40_432.pkl")

BatchesSliceHeight = 40
BatchesLength = 432
# num_val_data = 100

x1_hits = x1_hits[:,:,:]
x1_hits_input = x1_hits.reshape(x1_hits.shape + (1,))

x2_filtered = x2_filtered[:,:,:]
x2_filtered_input = x2_filtered.reshape(x2_filtered.shape + (1,))

x3_pure = x3_pure[:,:,:]
x3_pure_input = x3_pure.reshape(x3_pure.shape + (1,))

x_validation = x_validation[:,:,:]
x_validation_input = x_validation.reshape(x_validation.shape + (1,))

x5_pure_hits = x5_pure_hits[:,:,:]
x5_pure_hits_input = x5_pure_hits.reshape(x5_pure_hits.shape + (1,))

x4_merged_pure_hits = np.concatenate((x1_hits_input,x5_pure_hits_input),axis=0)
#%% CAE
   
autoencoder, encoder = autoencoderConv2D_1()
autoencoder.summary()

save_dir = r'C:\Users\Martin-PC\Magistrska Matlab\Clustering_CAE'

autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
# # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#%%
train_epochs = 15
batch_size = 256
autoencoder_train = autoencoder.fit(x4_merged_pure_hits, x4_merged_pure_hits, batch_size=batch_size, epochs=train_epochs,shuffle=True, validation_data=(x_validation_input,x_validation_input))

autoencoder.save_weights(save_dir+'\Clustering_CAE_v4_vandetta_3_6_12_30_3_128.h5')
#%% Training CAE
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
start_number = 686

predicted_x2_filtered_input = x_validation_input[:number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs = 5000000
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    freq = np.linspace(1, fs/5, BatchesSliceHeight)
    plt.pcolormesh(previous_time_data, freq, x_validation[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Validation Data, Input'+str(img))
    plt.show()
    plt.pcolormesh(previous_time_data, freq, image_decropped[img,:,:], cmap='viridis', shading='gouraud')
    plt.title('Validation Data, Output'+str(img))
    plt.show()
    
predicted_x2_filtered_input = x2_filtered[start_number:start_number+number_of_predicted,:,:]
for img in range(number_of_predicted):
    predicted = autoencoder.predict([predicted_x2_filtered_input])
    image_decropped = predicted.reshape((number_of_predicted,BatchesSliceHeight,BatchesLength))
    fs = 5000000
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    freq = np.linspace(1, fs/5, BatchesSliceHeight)
    plt.pcolormesh(previous_time_data, freq, x2_filtered[start_number+img,:,:], cmap='viridis', shading='gouraud')
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
# autoencoder.load_weights(save_dir+'\cae_v2_weights_30_3_34_434.h5')


#%%
# x_pure = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_channel2_df_pure_36_544.pkl")
# x_pure = x_pure[:,2:,:434]
# x_pure_input = x_pure.reshape(x_pure.shape + (1,))
# x_pure_pred_enc = encoder.predict(x_pure_input)
# scaler = StandardScaler()
# scaler.fit(x_pure_pred_enc)
# features_ch2_scaled = scaler.transform(x_pure_pred_enc)
y1 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_cfe_cwt_loc_filtered_ch2_amp02_200_400_2500_w30_pure.pkl"))
y2 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_cfe_slabsktkrbalanced_cwt_loc_filtered_ch2_amp02_200_400_2500_bins_raw.pkl"))
y3 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_cfe_cwt_loc_filtered_ch2_amp05_200_400_2500_w30_pure.pkl"))
y4 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_cfe_cwt_loc_filtered_ch2_amp02_200_400_2500_w12_pure.pkl"))
y5 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_cfe_cwt_loc_filtered_ch2_amp05_200_400_2500_w12_pure.pkl"))
y6 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_fft_loc_filtered_ch2_h02_200_400_2500_bins.pkl"))
y7 = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_fft_loc_filtered_ch2_h05_200_400_2500_bins.pkl"))
all_labels = np.array([y1,y2,y3,y4,y5,y6,y7]).T
df_labels = pd.DataFrame(all_labels, columns=['ctw_w30_prag_0.2_pure', 'ctw_w30_prag_0.2_raw','ctw_w30_prag_0.5_pure','ctw_w12_prag_0.2_pure','ctw_w12_prag_0.5_pure','fft_prag_0.2_raw','fft_prag_0.5_raw'])
#%%
x_pure_true_labels_ctw_w12 = pd.to_numeric(df_labels['ctw_w12_prag_0.2_pure'])
x_pure_true_labels_ctw_w12 = x_pure_true_labels_ctw_w12.astype(str)
labels_unique_keys_x_pure_true_labels_ctw_w12 = list(Counter(x_pure_true_labels_ctw_w12).keys())

replace_dict1_matlab = {'0': 0, '21': 2, '2': 2, '1': 1, \
                        '12': 1, '10': 1, '120': 1, '102': 1, \
                        '210': 2, '201': 2, '20': 2}

labels_true_3_ctw_w12 = np.array([float(y.replace(y,str(replace_dict1_matlab[y]))) for y in x_pure_true_labels_ctw_w12])

x_pure_true_labels_ctw_w30 = pd.to_numeric(df_labels['ctw_w30_prag_0.2_pure'])
x_pure_true_labels_ctw_w30 = x_pure_true_labels_ctw_w30.astype(str)
labels_unique_keys_x_pure_true_labels_ctw_w30 = list(Counter(x_pure_true_labels_ctw_w30).keys())

replace_dict1_matlab = {'0': 0, '21': 2, '2': 2, '12': 1, \
                        '1': 1, '210': 2, '10': 1, '201': 2, \
                        '102': 1, '20': 2, '120': 1}
        
labels_true_3_ctw_w30 = np.array([float(y.replace(y,str(replace_dict1_matlab[y]))) for y in x_pure_true_labels_ctw_w30])

x_pure_true_labels_ctw_fft05 = pd.to_numeric(df_labels['fft_prag_0.5_raw'])
x_pure_true_labels_ctw_fft05 = x_pure_true_labels_ctw_fft05.astype(str)
labels_unique_keys_x_pure_true_labels_ctw_fft05 = list(Counter(x_pure_true_labels_ctw_fft05).keys())

replace_dict1_matlab = {'21': 2, '120': 1, '12': 1, '102': 1, \
                        '210': 2, '201': 2, '2': 2, '0': 0, \
                        '1': 1, '10': 1, '20' : 2}
        
labels_true_3_ctw_fft05 = np.array([float(y.replace(y,str(replace_dict1_matlab[y]))) for y in x_pure_true_labels_ctw_fft05])
#%%
x_pure_pred_enc = encoder.predict(x3_pure_input)
# x_pure_pred_enc = encoder.predict(x2_filtered_input)

filename = r"C:\Users\Martin-PC\Magistrska Matlab\cae_v4_vandetta_3_6_12_30_3_128_x3_pure_encoded_output.pkl"
pd.to_pickle(x_pure_pred_enc, filename)
#%% PCA
# scaler = StandardScaler()
# scaler.fit(x_pure_pred_enc)
# features_ch2_scaled = scaler.transform(x_pure_pred_enc)
# pca = PCA(n_components=3)
# # pca.explained_variance_ratio_
# # decomposition = pca.fit(features_ch2)
# features_ch2_pca = pca.fit_transform(features_ch2_scaled)
# pca.explained_variance_ratio_
# sum(pca.explained_variance_ratio_)

# x_pure_pred_enc = features_ch2_pca #PCA VISUALIZATION @@@@@@@@@@@@@@@@@@@@@@@@@
#%%

cluster_nums = [8]

for cluster_num in cluster_nums:
    kmeans_model8 = KMeans(n_clusters=cluster_num, random_state=1).fit(x_pure_pred_enc)
    labels_kmeans8 = kmeans_model8.labels_
    if cluster_num == 8:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3), ('Category4', 4), ('Category5', 5),('Category6', 6), ('Category7', 7)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans8 == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans8 == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans8 == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        labels_kmeans_8 = np.choose(labels_kmeans8, [0, 2, 1,3,4,5,6,7]).astype(float)
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_kmeans_8)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        ax.legend()
        plt.title('8 clusters K-means')
        plt.show() 
        # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))

cluster_nums = [5]

for cluster_num in cluster_nums:
    kmeans_model = KMeans(n_clusters=cluster_num, random_state=1).fit(x_pure_pred_enc)
    labels_kmeans = kmeans_model.labels_
    if cluster_num == 5:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3), ('Category4', 4)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        labels_kmeans_3 = np.choose(labels_kmeans, [0, 2, 1,3,4]).astype(float)
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_kmeans_3)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        ax.legend()
        plt.title('8 clusters K-means')
        plt.show() 
        # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))

cluster_nums = [3]

for cluster_num in cluster_nums:
    kmeans_model = KMeans(n_clusters=cluster_num, random_state=1).fit(x_pure_pred_enc)
    labels_kmeans = kmeans_model.labels_
    if cluster_num == 3:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        labels_kmeans_3 = np.choose(labels_kmeans, [1, 2, 0]).astype(float)
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_kmeans_3)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        ax.legend()
        plt.title('K-means clustering')
        plt.show() 
        # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))    
        
# for cluster_num in cluster_nums:
#     if cluster_num == 3:
#         fig = plt.figure(1, figsize=(8, 8))
#         ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
#         fig.add_axes(ax)
#         for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
#             ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
#                       x_pure_pred_enc[labels_kmeans == label, 1].mean(),
#                       x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
#                       name,
#                       bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
#                       )
#         labels_kmeans_3 = np.choose(labels_kmeans, [3, 4, 5]).astype(float)
#         ax.scatter(x_pure_pred_enc[:, 3], x_pure_pred_enc[:, 4], x_pure_pred_enc[:, 5], c=labels_kmeans_3)
#         # ax.w_xaxis.set_ticklabels([])
#         # ax.w_yaxis.set_ticklabels([])
#         # ax.w_zaxis.set_ticklabels([])
#         ax.legend()
#         plt.title('3 clusters K-means PCA 3-4-5')
#         plt.show() 
#         # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))       
    
cluster_nums = [3]
for cluster_num in cluster_nums:
    spectral_clustering_model = SpectralClustering(n_clusters=cluster_num, assign_labels='discretize', random_state=0).fit(x_pure_pred_enc)
    labels_spectral = spectral_clustering_model.labels_
    if cluster_num == 3:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        labels_spectral_3 = np.choose(labels_spectral, [0, 1, 2]).astype(float)
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_spectral_3)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        plt.title('Spectral Clustering')
        plt.show()
        # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_spectral_3), 3))
        
cluster_nums = [3]
for cluster_num in cluster_nums:
    if cluster_num == 3:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_true_3_ctw_w12)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        plt.title('Labels True 3 CWT w12')
        plt.show()

# cluster_nums = [3]
# for cluster_num in cluster_nums:
#     if cluster_num == 3:
#         fig = plt.figure(1, figsize=(8, 8))
#         ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
#         fig.add_axes(ax)
#         for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
#             ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
#                       x_pure_pred_enc[labels_kmeans == label, 1].mean(),
#                       x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
#                       name,
#                       bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
#                       )
#         ax.scatter(x_pure_pred_enc[:, 3], x_pure_pred_enc[:, 4], x_pure_pred_enc[:, 5], c=labels_true_3_ctw_w12)
#         # ax.w_xaxis.set_ticklabels([])
#         # ax.w_yaxis.set_ticklabels([])
#         # ax.w_zaxis.set_ticklabels([])
#         plt.title('3 clusters True PCA 3-4-5')
#         plt.show()

for cluster_num in cluster_nums:
    if cluster_num == 3:
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
        fig.add_axes(ax)
        for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2)]:
            ax.text3D(x_pure_pred_enc[labels_kmeans == label, 0].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 1].mean(),
                      x_pure_pred_enc[labels_kmeans == label, 2].mean(), 
                      name,
                      bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
                      )
        ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 2], c=labels_true_3_ctw_fft05)
        # ax.w_xaxis.set_ticklabels([])
        # ax.w_yaxis.set_ticklabels([])
        # ax.w_zaxis.set_ticklabels([])
        plt.title('Labels True 3 FFT0.5')
        plt.show()

# filename = r"C:\Users\Martin-PC\Magistrska Matlab\cae_v4_vandetta_3_6_12_30_3_128_labels_spectral_3_.pkl"
# pd.to_pickle(labels_spectral_3, filename)

# filename = r"C:\Users\Martin-PC\Magistrska Matlab\cae_v4_vandetta_3_6_12_30_3_128_labels_kmeans_3_.pkl"
# pd.to_pickle(labels_kmeans_3, filename)

# filename = r"C:\Users\Martin-PC\Magistrska Matlab\cae_v4_vandetta_3_6_12_30_3_128_labels_kmeans_8_.pkl"
# pd.to_pickle(labels_kmeans_8, filename)

# x2 = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_loc_filtered_ch2_w12_pure_f16_34_434.pkl")
# pure_samples_2_predict = [0,1,2,3,4,5,42,43,44,45,120,121,122,123,124]        
# x_pure_pred_cae = x_pure[pure_samples_2_predict,:,:]
# k = 0
# for sample in (pure_samples_2_predict):
#     predicted = autoencoder.predict([x_pure_pred_cae])
#     image_decropped = predicted.reshape((len(pure_samples_2_predict),BatchesSliceHeight,BatchesLength))
#     fs = 5000000
#     dt = 1/fs
#     previous_time_data = np.arange(0,BatchesLength)*dt
#     freq = np.linspace(1, fs/5, BatchesSliceHeight)
#     plt.pcolormesh(previous_time_data, freq, x_pure_pred_cae[k,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Input'+str(sample))
#     plt.show()
#     plt.pcolormesh(previous_time_data, freq, image_decropped[k,:,:], cmap='viridis', shading='gouraud')
#     plt.title('Output'+str(sample))
#     plt.show()
#     k = k+1

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