# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:15:13 2023

@author: Martin-PC
"""

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
from tensorflow import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose, Dropout
from keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from keras import optimizers
import numpy as np
import pandas as pd
from time import time
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
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
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam

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
# lambda_cwtms_bio_cfe_gfe_zeta_lower_rev_w4_32x32_v1_16_32_14_256_6_100_64
# layers_depths=[80, 160, 56, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[40, 80, 64, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[4, 8, 56, 4] #nujno bottleneck = stevilo clustrov
layers_depths=[36, 36, 512, 6, 256, 128]
layer_depth_latent_cnn=16
dropout_ratio = 0.01
identifier = 'chatGPT_32x64_july'
# layers_depths=[30, 60, 64, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12


# Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Silhuette_clustering_analysis_re"
# Output_filename = Output + ver_str + ".csv"

def autoencoderConv2D_1(input_shape= (32,64),
                        layers_depths=[36, 36, 512, 6, 256, 128],
                        layer_depth_latent_cnn=16
                        ):
    # layers_depths=[36, 18, 256, 6]
    # layer_depth_latent_cnn = 10
    conv2Dfilters = [(3,3), (3,3)]
    BatchesSliceHeight = 32
    BatchesLength = 64
    stride_pool_h = 2
    stride_pool_w = 2
    max_pool_filter_h = 2
    max_pool_filter_w = 2

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
    conv_latent_1 = Conv2D(layer_depth_latent_cnn, ((pool_1_2.get_shape().as_list())[1:][0],(pool_1_2.get_shape().as_list())[1:][1]), activation = 'relu', padding = "SAME", strides=(1, stride_latent_2dconv))(concat)
    pool_latent_1 = MaxPooling2D(pool_size = (2,2), strides = stride_latent_max_pooling)(conv_latent_1)
    flat_latent_1  = Flatten()(pool_latent_1)
    dense_num = reduce((lambda x, y: x * y), (conv_latent_1.get_shape().as_list())[1:])    
    dense_latent_1 = Dense(dense_num, activation = 'relu')(flat_latent_1)
    dropout_latent_1 = Dropout(dropout_ratio)(dense_latent_1)
    dense_latent_2 = Dense(layers_depths[2], activation = 'relu')(dropout_latent_1)
    dropout_latent_2 = Dropout(dropout_ratio)(dense_latent_2)
    dense_latent_3 = Dense(layers_depths[4], activation = 'relu')(dropout_latent_2)
    dropout_latent_3 = Dropout(dropout_ratio)(dense_latent_3)
    dense_latent_4 = Dense(layers_depths[5], activation = 'relu')(dropout_latent_3)
    dropout_latent_4 = Dropout(dropout_ratio)(dense_latent_4)
    dense_latent_bottleneck = Dense(layers_depths[3], activation = 'relu')(dropout_latent_4)
    # encoder_output_linear_regression = Dense(1, activation='linear')(dense_latent_bottleneck)
    clustering_layer = ClusteringLayer(n_clusters=4)(dense_latent_bottleneck)
    dense_latent_a = Dense(layers_depths[4], activation = 'relu')(clustering_layer)
    dropout_latent_a = Dropout(dropout_ratio)(dense_latent_a)
    dense_latent_b = Dense(layers_depths[5], activation = 'relu')(dropout_latent_a)
    dropout_latent_b = Dropout(dropout_ratio)(dense_latent_b)
    dense_latent_c = Dense(layers_depths[2], activation = 'relu')(dropout_latent_b)
    dropout_latent_c = Dropout(dropout_ratio)(dense_latent_c)
    dense_latent_d = Dense(dense_num, activation = 'relu')(dropout_latent_c)
    dropout_latent_d = Dropout(dropout_ratio)(dense_latent_d)
    reshape_latent_1 = Reshape(tuple(conv_latent_1.get_shape().as_list())[1:])(dropout_latent_d) # v isto kot pool_latent_1
    # upsampling_latent_1= UpSampling2D((2, 2))(reshape_latent_1)
    
    reshape_latent_1 = Reshape(tuple(conv_latent_1.get_shape().as_list())[1:])(dropout_latent_d) # v isto kot pool_latent_1
    # upsampling_latent_1= UpSampling2D((2, 2))(reshape_latent_1)
    conv_latent_2 = Conv2DTranspose(layer_depth_latent_cnn, ((pool_1_2.get_shape().as_list())[1:][0],(pool_1_2.get_shape().as_list())[1:][1]), activation = 'relu', padding = "SAME", strides=(1, stride_latent_2dconv))(reshape_latent_1)
    remerge_size = (pool_1_2.get_shape().as_list())[1]
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
    # encoder_linear_regression = Model(inputs=inputs_1, outputs=encoder_output_linear_regression, name='encoder_linear_regression')
    # encoder.summary()
    # decoder = Model(inputs=dense_latent_bottleneck_input, outputs=output_1, name='decoder')
    # decoder.summary()
    AE =  Model(inputs=inputs_1, outputs=output_1, name='AE')
    # AE = Model(inputs=inputs_1, outputs=decoder(encoder(inputs_1)), name='autoencoder')
    
    return AE, encoder, None


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
        input_dim = layers_depths[3]
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
   
autoencoder, encoder, encoder_linear_regression = autoencoderConv2D_1()
autoencoder.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
save_dir = r'C:\Users\Martin-PC\\Biocomposite_analysis_isndcm\Clustering_CAE'
autoencoder.compile(optimizer=opt, loss='mse') #mse različn način kako se penalizira razlikovanje vhodnega in izhodnega spektograma

def balance_dataset_clustering(cluster_cols, n_clusters, max_rows_per_cluster, df):
    df = df.drop(df[df['FVTA'] == 0].index)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df[cluster_cols])
    df['cluster'] = kmeans.predict(df[cluster_cols])
    desired_samples = min(max_rows_per_cluster, len(df) // kmeans.n_clusters)
    sampled_data = pd.concat([df[df['cluster'] == i].sample(desired_samples, replace=True) for i in range(kmeans.n_clusters)])
    print(sampled_data['cluster'].value_counts())
    return sampled_data.index
#%%
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# # save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
# save_dir = save_dir_root + '\\Clustering_CAE_isndcm_seperate_decoder_chatGPT_32x64_45-02s-11_v4_36_36_16_512_6_10_16' + '.h5'
# autoencoder.load_weights(save_dir)
#%%
filename = '45-02s-13'
cwtms_bio_cold1 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_bio_cold1_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
cwtms_bio_cold1_features['Relative time [%]'] = (cwtms_bio_cold1_features['time']/max(cwtms_bio_cold1_features['time']))*100
# counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()
# indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=20) & (cwtms_bio_cold1_features['Relative time [%]']<= 80))[0])
#
# indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=50))[0])
# cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[indices]
# cwtms_bio_cold1 = cwtms_bio_cold1[indices,:,:]
#%%
filename = '45-02s-11'
cwtms_bio_cold2 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_bio_cold2_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
cwtms_bio_cold2_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold2_features)
cwtms_bio_cold2 = cwtms_bio_cold2[cwtms_bio_cold2_features_indices,:,:]
cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[cwtms_bio_cold2_features_indices,:]
cwtms_bio_cold2_features['Relative time [%]'] = (cwtms_bio_cold2_features['time']/max(cwtms_bio_cold2_features['time']))*100
#
# indices=list(np.where((cwtms_bio_cold2_features['Relative time [%]']>=50))[0])
# cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[indices]
# cwtms_bio_cold2 = cwtms_bio_cold2[indices,:,:]
#%%
# cwtms_bio_cold = np.concatenate((cwtms_bio_cold1,cwtms_bio_cold2),axis=0)
# cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0)
cwtms_bio_cold = cwtms_bio_cold2
cwtms_bio_cold_features = cwtms_bio_cold2_features
#%%
# files_extra = [None]*3
# files_extra_features = [None]*3
# for ind,(file,features) in enumerate(zip(
#         ['cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_512.pkl',
#           'cwtms_zeta_lower_bio_cold_squared_scaled_w4_32_512.pkl',
#         'cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_512.pkl'],
        
#         ['merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_tok_32_64_32_512_16_10_20.pkl',
#           'merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_16_256_6_75_256.pkl',
#           'merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_16_256_6_75_256.pkl'])):
    
#     files_extra[ind] = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis"+'\\'+file)
#     files_extra_features[ind] = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis"+'\\'+features)
#     files_extra_features_indices = balance_dataset_clustering(['FVTA'],4,6000,files_extra_features[ind])
#     files_extra[ind] = files_extra[ind][files_extra_features_indices,:,:]
    
# files_extra = np.concatenate(files_extra,axis=0)
# cwtms_bio_cold = np.concatenate((cwtms_bio_cold,files_extra),axis=0)

# x1_validation = files_extra.reshape(files_extra.shape + (1,))
# batch_size_samples = 1000
# index = 0
# index_array = np.arange(x1_validation.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_validation.shape[0])]
# x1_validation_input_batch = x1_validation[idx]
# x1_validation_input_batch = x1_validation_input_batch.astype(np.float64)
# y_dim_height = 32
# x_dim_length = 64
# zoom_factors = (1,y_dim_height / x1_validation_input_batch.shape[1], x_dim_length / x1_validation_input_batch.shape[2],1)

# # Perform bilinear interpolation on all images in the array
# from scipy.ndimage import zoom
# x1_validation_input_batch = zoom(x1_validation_input_batch, zoom_factors, order=1)
# x1_validation_input_batch = x1_validation_input_batch
#%%



# cwtms_bio_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_cfe-100_80_288.pkl")
# cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_gfe-20_80_288.pkl")

from scipy.ndimage import zoom
cwtms_bio_cold = cwtms_bio_cold.astype(np.float64)
y_dim_height = 32
x_dim_length = 64
zoom_factors = (1,y_dim_height / cwtms_bio_cold.shape[1], x_dim_length / cwtms_bio_cold.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_bio_cold_resized = zoom(cwtms_bio_cold, zoom_factors, order=1)
cwtms_bio_cold = cwtms_bio_cold_resized
#%

BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

cwtms_bio_cold_input = cwtms_bio_cold.reshape(cwtms_bio_cold.shape + (1,))

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
# x1_training_input_batch = np.concatenate((cwtms_bio_cold_input,cwtms_bio_tok_input),axis=0)
x1_training_input_batch = cwtms_bio_cold_input
#%%
batch_size_samples = 1000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_validation_input_batch = x1_training_input_batch[idx]

# batch_size_samples = 30000
# index = 0
# index_array = np.arange(x1_training_input_batch.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
# x1_training_input_batch = x1_training_input_batch[idx]

#bio cold training
# batch_size_samples = 32000
# index = 0
# index_array = np.arange(cwtms_bio_cold_input.shape[0])
# np.random.shuffle(index_array)
# idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, cwtms_bio_cold_input.shape[0])]
# x1_training_input_batch = cwtms_bio_cold_input[idx]
#%%
# define the function to create the model
# def create_model(learning_rate=0.001, dropout_ratio=0.2, layers_depths=[36, 18, 256, 6], layer_depth_latent_cnn=10):
#     model, _ = autoencoderConv2D_1(layers_depths=layers_depths, layer_depth_latent_cnn=layer_depth_latent_cnn)
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='mse')
#     return model

# # create KerasRegressor object
# model = KerasRegressor(build_fn=create_model, epochs=5, batch_size=64)

# # define the hyperparameter grid
# param_grid = {
#     'learning_rate': [0.005, 0.0001, 0.00001],
#     'dropout_ratio': [0.01, 0.1,0.2],
#     'layers_depths': [[36, 18, 512, 6]],
#     'layer_depth_latent_cnn': [10, 12, 14]
# }

# # create GridSearchCV object
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=1)

# # fit the GridSearchCV object to the data
# grid_result = grid.fit(x1_training_input_batch, x1_training_input_batch)

# # print the best parameters and best score
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#%% data augmentation
# datagen = ImageDataGenerator(
#     # featurewise_center=True,
#     # featurewise_std_normalization=True,
#     rotation_range=30,
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     horizontal_flip=True,
#     validation_split=0.3)
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x1_training_input_batch)
#%%
# train_epochs = 3
# # train_epochs = 50
# # batch_size = 20
# # batch_size = 256
# # batch_size = 128
# # batch_size = 64
# # batch_size = 16
# batch_size = 32
# # # fits the model on batches with real-time data augmentation:
# # autoencoder_train = autoencoder.fit(datagen.flow(x1_training_input_batch, x1_training_input_batch, batch_size=batch_size,
# #                                                  subset='training'),
# #          validation_data=(x1_validation_input_batch, x1_validation_input_batch),
# #          # steps_per_epoch=len(x1_training_input_batch) / 32, epochs=train_epochs,
# #          shuffle=True,)

# autoencoder_train = autoencoder.fit(datagen.flow(x1_training_input_batch, x1_training_input_batch, batch_size=batch_size), 
#                     epochs=train_epochs, # one forward/backward pass of training data
#                     steps_per_epoch=x1_training_input_batch.shape[0]//batch_size, # number of images comprising of one epoch
#                     validation_data=(x1_validation_input_batch, x1_validation_input_batch), # data for validation
#                     validation_steps=x1_validation_input_batch.shape[0]//batch_size)
#%%
train_epochs = 100
# train_epochs = 5
# batch_size = 20
batch_size = 128
# batch_size = 64
# batch_size = 64
# batch_size = 16
# batch_size = 32
autoencoder_train = autoencoder.fit(x1_training_input_batch, 
                                    x1_training_input_batch,
                                    batch_size=batch_size, epochs=train_epochs,shuffle=True,
                                    validation_data=(x1_validation_input_batch, x1_validation_input_batch), # data for validation
                                    # steps_per_epoch=len(x1_training_input_batch) // (batch_size*2)
                                    )
# breakpoint()
#%%
number_of_predicted = 6
# start_number = 1242
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_64.pkl")

start_number = 500
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_64.pkl".format(filename=filename))
cwtms_cfe_test_validation = cwtms_bio_cold
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_bio_tok_scaled_32_64.pkl")
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_cfe_cold_scaled_32_64.pkl")
# cwtms_cfe_test_validation = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_rho_cfe_tok_scaled_32_64.pkl")
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
save_dir = save_dir_root + '\Clustering_CAE_{identifier}_{filename}_v1'.format(filename=filename,identifier=identifier) +  ver_str + '.h5'
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
autoencoder, encoder, encoder_linear_regression = autoencoderConv2D_1()
autoencoder.summary()

# save_dir = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 10
batch_size = 32
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
# save_dir = save_dir_root + f'\\Clustering_CAE__{identifier}_45-02s-11_v3_36_36_16_512_6_100_32' + '.h5'
save_dir = save_dir_root + '\\Clustering_CAE_chatGPT_32x64_july_45-02s-11_v1_36_36_14_512_3_100_64' + '.h5'
autoencoder.load_weights(save_dir)
#%% load data predict
# cwtms_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_64.pkl".format(filename=filename))
# # cwtms_bio_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_64.pkl")
# # cwtms_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_64.pkl")
# # cwtms_cfe_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_64.pkl")
# # cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_cfe-100_80_288.pkl")
# # cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_gfe-20_80_288.pkl")

# BatchesSliceHeight = 32
# BatchesLength = 32

# cwtms_bio_cold_input = cwtms_bio_cold.reshape(cwtms_bio_cold.shape + (1,))
# # cwtms_bio_tok_input = cwtms_bio_tok.reshape(cwtms_bio_tok.shape + (1,))
# # cwtms_cfe_cold_input = cwtms_cfe_cold.reshape(cwtms_cfe_cold.shape + (1,))
# # cwtms_cfe_tok_input = cwtms_cfe_tok.reshape(cwtms_cfe_tok.shape + (1,))

# #%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# x_pure_pred_enc = encoder.predict(cwtms_bio_cold)

# classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename))
# #%
# columns_vec = []
# for i in range(1,layers_depths[3]+1):
#     columns_vec.append('Emb'+str(i))
# x_pure_pred_enc_df = pd.DataFrame(x_pure_pred_enc,columns = columns_vec)
# x_pure_pred_enc_df.reset_index(drop=True)
# x_pure_pred_enc_df.astype('float32').dtypes
# classical_features_filtered_hits_tok.reset_index(inplace=True, drop=True)
# merged_features_embedded_classical = pd.concat([classical_features_filtered_hits_tok, x_pure_pred_enc_df], axis=1)
# # merged_features_embedded_classical.astype('float32').dtypes

# # merged_features_embedded_classical = merged_features_embedded_classical.drop(merged_features_embedded_classical[merged_features_embedded_classical.FMXA_ch1 < 0.1].index)
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm'
# save_dir = save_dir_root + '\merged_features_lambda_seperate_dec_emb_cla_w7_{filename}_32x64'.format(filename=filename) +  ver_str + '.pkl'
# pd.to_pickle(merged_features_embedded_classical, save_dir)
# # save_dir = save_dir_root + '\merged_features_embedded_classical_rho20_lambda_cfe_cold' +  ver_str + '.xlsx'
# # merged_features_embedded_classical.to_excel(save_dir, index = False) 

# # merged_features_embedded_classical = pd.read_excel(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.xlsx")
# # save_dir = save_dir_root + '\merged_features_embedded_classical_ch2' +  ver_str + '.pkl'
# # pd.to_pickle(merged_features_embedded_classical, save_dir)
# # cfe_cold

#%% CLustering Layer Initialization
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# filename = '45-02s-13'
# cwtms_bio_cold1 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
# cwtms_bio_cold1_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# # cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
# # cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
# # cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
# cwtms_bio_cold1_features['Relative time [%]'] = (cwtms_bio_cold1_features['time']/max(cwtms_bio_cold1_features['time']))*100
# # counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()
# # indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=20) & (cwtms_bio_cold1_features['Relative time [%]']<= 80))[0])
# #
# # indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=50))[0])
# # cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[indices]
# # cwtms_bio_cold1 = cwtms_bio_cold1[indices,:,:]
#%
filename = '45-02s-11'
cwtms_bio_cold2 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_bio_cold2_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# cwtms_bio_cold2_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold2_features)
# cwtms_bio_cold2 = cwtms_bio_cold2[cwtms_bio_cold2_features_indices,:,:]
# cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[cwtms_bio_cold2_features_indices,:]
cwtms_bio_cold2_features['Relative time [%]'] = (cwtms_bio_cold2_features['time']/max(cwtms_bio_cold2_features['time']))*100

# indices=list(np.where((cwtms_bio_cold2_features['Relative time [%]']>=50))[0])
# cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[indices]
# cwtms_bio_cold2 = cwtms_bio_cold2[indices,:,:]
# %
# cwtms_bio_cold = np.concatenate((cwtms_bio_cold1,cwtms_bio_cold2),axis=0)
# cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0)
cwtms_bio_cold = cwtms_bio_cold2
cwtms_bio_cold_features = cwtms_bio_cold2_features

from scipy.ndimage import zoom
cwtms_bio_cold = cwtms_bio_cold.astype(np.float64)
y_dim_height = 32
x_dim_length = 64
zoom_factors = (1,y_dim_height / cwtms_bio_cold.shape[1], x_dim_length / cwtms_bio_cold.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_bio_cold_resized = zoom(cwtms_bio_cold, zoom_factors, order=1)
cwtms_bio_cold = cwtms_bio_cold_resized
#%

# BatchesSliceHeight = y_dim_height
# BatchesLength = x_dim_length

cwtms_bio_cold_input = cwtms_bio_cold.reshape(cwtms_bio_cold.shape + (1,))

x1_training_input_batch = cwtms_bio_cold_input
#%
# del cwtms_bio_cold1_features
del cwtms_bio_cold2_features
del cwtms_bio_cold_input
# del cwtms_bio_cold_resized
# x1_training_input_batch = cwtms_bio_cold_input
del cwtms_bio_cold
del cwtms_bio_cold_features
# del cwtms_bio_cold1
del cwtms_bio_cold2
#%%
# autoencoder, encoder, encoder_linear_regression = autoencoderConv2D_1()
# autoencoder.summary()

# # save_dir = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
# train_epochs = 10
# batch_size = 32
# ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# # save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
# save_dir = save_dir_root + '\\Clustering_CAE_isndcm_seperate_decoder_chatGPT_32x64_45-02s-11_v4_36_36_16_512_6_10_16' + '.h5'
# autoencoder.load_weights(save_dir)
# #%%
# save_dir = save_dir_root + '\\Clustering_CAE_encoder_clustering_chatGPT_32x64_6_clusters_all_files_v3_36_36_16_512_6_10_32' + '.h5'
# encoder.load_weights(save_dir)

#%%  retraining encoder
# n_clusters = 5
# clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
# model = Model(inputs=encoder.input, outputs=clustering_layer)
# for layer in model.layers[:]:
#     print(layer.name )
#     layer.trainable = True
# # for layer in model.layers[:]:
# #     print(layer.name )
# #     layer.trainable = False
        
# # opt_clustering = optimizers.SGD(learning_rate=0.000001)
# opt_clustering = keras.optimizers.Adam(learning_rate=0.00005) #šel vseh 4000 iteracij
# # opt_clustering = keras.optimizers.Adam(learning_rate=0.002)  # 32_64_chatGPT

# model.compile(optimizer=opt_clustering, loss='mse')

# # x_pure_input_scaled_pca = features_ch2_pca
# clustering_alg = KMeans(n_clusters=n_clusters)
# # from sklearn.cluster import SpectralClustering
# # clustering_alg = SpectralClustering(n_clusters=n_clusters)

# x_pure_pred_enc = model.predict(x1_training_input_batch)

# X = x_pure_pred_enc

# # # db = DBSCAN(eps=6.9, min_samples=100).fit(X) # Robust Scaler, klasične značilke
# # # db = DBSCAN(eps=1.9, min_samples=100).fit(X) # Robust Scaler, standardScaler, PCA
# # db = DBSCAN(eps=0.03, min_samples=200).fit(X) # Robust Scaler, standardScaler, PCA
# # # db = DBSCAN(eps=0.018, min_samples=200).fit(X) # Robust Scaler, standardScaler, PCA

# # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# # core_samples_mask[db.core_sample_indices_] = True
# # labels_dbscan = db.labels_

# # dim_x = 0 #energija CFE
# # dim_y = 1# globoka 1 CFE
# # dim_z = 2 # FCOG CFE

# # plt.style.use('dark_background')
# # fig = plt.figure(1, figsize=(8, 8))
# # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=24, auto_add_to_figure=False)
# # fig.add_axes(ax)

# # scatter = ax.scatter(X[:, dim_x], X[:, dim_y], X[:, dim_z], c=labels_dbscan, cmap='viridis', label = labels_dbscan, edgecolors='none') #GFE
# # ax.tick_params(axis = 'both', labelsize=16)
# # plt.rcParams['legend.title_fontsize'] = 22
# # plt.title('DBScan filtering')
# # plt.show() 
# # x_pure_pred_enc_df = pd.DataFrame(X)
# # x_pure_pred_enc_df['labels_DBscan'] =  labels_dbscan
# # x_pure_pred_enc_df_dbscan_filtered = x_pure_pred_enc_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index)

# # # hits_channel2_df.reset_index(inplace=True, drop=True)
# # # X = features_merged_labels_pure_selected_deep_FVTA_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
# # # features_merged_dropped_dbscan = features_merged.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
# # x_pure_pred_enc_unscaled_features = np.array(X)
# # # pd.to_pickle(hits_channel2_df_dbscan, r"C:\Users\Martin-PC\Magistrska Matlab\CFE_hits_channel2_df_labels_ch1_dbscan.pkl")       

# # # labels_true_input_cwt05_pure_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_True_ctw'])
# # labels_only_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_DBscan'])
# # x_pure_pred_enc_df_dbscan_filtered.drop(x_pure_pred_enc_df_dbscan_filtered.columns[np.arange(x_pure_pred_enc_df_dbscan_filtered.shape[-1]-1,x_pure_pred_enc_df_dbscan_filtered.shape[-1])], axis = 1, inplace = True)
# # x_pure_pred_enc_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered)     
# # X=x_pure_pred_enc_dbscan
# # print(len(x_pure_pred_enc_df)-len(x_pure_pred_enc_dbscan))
# # x_pure_pred_enc = X
# # X = x1_training_input_batch

# y_pred = clustering_alg.fit_predict(x_pure_pred_enc)
# y_pred_last = np.copy(y_pred)

# model.get_layer(name='clustering').set_weights([clustering_alg.cluster_centers_])
# model.summary()
# model_weights = model.get_weights()

# del x_pure_pred_enc
# del X
# # del x_pure_pred_enc_dbscan
# # del x_pure_pred_enc_df_dbscan_filtered
# # del x_pure_pred_enc_df
# # del x_pure_pred_enc_unscaled_features

# # x_pure_input
# # x2a
# # x_pure_pred_enc
# # output = x_pure_pred_enc

# #% retraining encoder
# train_epochs = 10
# batch_size = 32
# ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# # save_dir = save_dir_root + f'\\Clustering_CAE_encoder_clustering_{identifier}_6_clusters_all_files_v1_36_36_16_512_6_10_32' + '.h5'
# save_dir = save_dir_root + '\\Clustering_CAE_encoder_clustering_chatGPT_32x64_6_clusters_all_files_v3_36_36_16_512_6_10_32' + '.h5'
# # model.load_weights(save_dir)
# # model.summary()

# #%
# # opt_clustering = optimizers.SGD(learning_rate=0.00008)
# # opt_clustering = keras.optimizers.Adam(learning_rate=0.0001) #šel vseh 4000 iteracij
# opt_clustering = keras.optimizers.Adam(learning_rate=0.00005)
# # opt_clustering = keras.optimizers.Adam(learning_rate=0.002)

# model.compile(optimizer=opt_clustering, loss='mse')
# # x1_training_input_batch = x1_training_input_batch

#%% initialize clustering layer
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

x3_pure_input = x1_training_input_batch
autoencoder, encoder, encoder_linear_regression = autoencoderConv2D_1()
encoder.summary()

# save_dir = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
autoencoder.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 10
batch_size = 20
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
# save_dir = save_dir_root + '\yUpsilon_cwtms_bio_cfe_gfe_v4_retrained' +  ver_str + '.h5'
save_dir = save_dir_root + '\\Clustering_CAE_isndcm_seperate_decoder_chatGPT_32x64_45-02s-11_v4_36_36_16_512_6_10_16' + '.h5'
autoencoder.load_weights(save_dir)

n_clusters = 6
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
for layer in model.layers[-4:]:
        layer.trainable = True
for layer in model.layers[:-4]:
        layer.trainable = False
# optimizer = optimizers.Adam(learning_rate=0.000001,beta_1=0.9,beta_2=0.999,epsilon=1e-07, amsgrad=False)
# 
model.compile(optimizer=SGD(learning_rate=0.001), loss='kld')
# opt_clustering = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=opt_clustering, loss='mse')

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
#%%
import matplotlib.colors as mcolors
Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_analysis"
Output_filename = Output + ver_str + ".csv"

filename = 'all_files'
# filename = '45-02s-13'
# identifier = '16_8_16_512_3_30_256'
identifier = 'chatGPT_32x64_july_last_6_layers_train'
loss = 0
index = 0
maxiter = 4000
update_interval = 1000
batch_size = 32
index_array = np.arange(x3_pure_input.shape[0])
tol = 0.00001 # tolerance threshold to stop training
# labels_true_3_ctw_w12 = labels_true_3_ctw_w12[0:len(x3_pure_input)]

layer_names = [layer.name for layer in model.layers]
x_input = x3_pure_input # ali x3a --> vsi
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[-2]).output)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        output = model.predict(x3_pure_input, verbose=0)
        output_distribution = target_distribution(output)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        # output from intermediate layer
        intermediate_output = intermediate_layer_model.predict(x3_pure_input)  
        y_pred = output.argmax(1)
        y_pred_auxilary = output_distribution.argmax(1)
        cluster_nums = [n_clusters]
        
        pca = PCA(n_components = 3)
        intermediate_output = pca.fit_transform(intermediate_output)
        
        # from sklearn.preprocessing import StandardScaler
        # from sklearn.cluster import DBSCAN


        # # db = DBSCAN(eps=6.9, min_samples=105).fit(X) # brez PCA

        # # db = DBSCAN(eps=0.9, min_samples=100).fit(X) # PCA
        

        # # db = DBSCAN(eps=6.9, min_samples=100).fit(intermediate_output) # Robust Scaler, klasične značilke
        
        # # db = DBSCAN(eps=1.9, min_samples=100).fit(intermediate_output) # Robust Scaler, standardScaler, PCA
        # # db = DBSCAN(eps=0.8, min_samples=100).fit(intermediate_output) # Robust Scaler, standardScaler, PCA
        # db = DBSCAN(eps=0.3, min_samples=100).fit(intermediate_output) # Robust Scaler, standardScaler, PCA
        # # db = DBSCAN(eps=150, min_samples=200).fit(intermediate_output)

        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels_dbscan = db.labels_

        # dim_x = 0 #energija CFE
        # dim_y = 1# globoka 1 CFE
        # dim_z = 2 # FCOG CFE

        # plt.style.use('dark_background')
        # fig = plt.figure(1, figsize=(8, 8))
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=24, auto_add_to_figure=False)
        # fig.add_axes(ax)
        # # for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3),('Category4', 4),('Category5', 5),('Category6', 6),('Category7', 7),('Category8', 8),('Category9', 9),('Category10', 10),('Category11', 11),('Category12', 12),('Category13', 13)]:
        # #     ax.text3D(X[labels_dbscan == label, 0].mean(),
        # #               X[labels_dbscan == label, 1].mean(),
        # #               X[labels_dbscan == label, 2].mean(), 
        # #               name,
        # #               bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
        # #               )
        # # # unsupervised_labels = np.choose(labels_dbscan, [1, 2, 0,3,0,0,0,0,0,0,0]).astype(float)
        # # ax.scatter(x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 5], x_pure_pred_enc[:, 6], c=labels_kmeans_3)
        # scatter = ax.scatter(intermediate_output[:, dim_x], intermediate_output[:, dim_y], intermediate_output[:, dim_z], c=labels_dbscan, cmap='viridis', label = labels_dbscan, edgecolors='none') #GFE
        # # plt.xlim(-2,4)  
        # # plt.ylim(-4,3) 
        # # ax.set_zlim(-2,4) 
        # # ax.scatter(X[:, 0], X[:, 1], X[:,4], c=labels_dbscan) #CFE
        # # plt.xlim(-2,4)  
        # # plt.ylim(-4,4) 
        # # ax.set_zlim(-2,3) 
        # ax.tick_params(axis = 'both', labelsize=16)
        # # plt.xlim(0,8)  # cluster belongings, K-means point of view 0-1-2
        # # plt.ylim(-6,2) 
        # # ax.set_zlim(-3,0)

        # # plt.legend(handles=scatter.legend_elements()[0], 
        # #         labels=['0-200 kHz','200-400 kHz','400-1000 kHz'], loc='best', bbox_to_anchor=(0.0, 0., 0.3, 0.1),fontsize = 16,
        # #         title="Legenda")

        # plt.rcParams['legend.title_fontsize'] = 22
        # # ax.set_xlabel('Globoka značilka 2', fontsize=20)
        # # ax.set_ylabel('Globoka značilka 3')
        # # ax.set_zlabel('Globoka značilka 4', fontsize=30, rotation = 0)
        # # ax.xaxis.set_label_coords(1.55, -0.55)
        # plt.title('DBScan filtering')
        # plt.show() 
        #         # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))
        # # print(sum(labels_dbscan))
        # #%
        # x_pure_pred_enc_df = pd.DataFrame(intermediate_output)
        # x_pure_pred_enc_df['labels_DBscan'] =  labels_dbscan
        # # x_pure_pred_enc_df['labels_True_ctw'] =  labels_true


        # x_pure_pred_enc_df_dbscan_filtered = x_pure_pred_enc_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index)

        # # hits_channel2_df.reset_index(inplace=True, drop=True)
        # # X = features_merged_labels_pure_selected_deep_FVTA_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
        # # features_merged_dropped_dbscan = features_merged.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
        # x_pure_pred_enc_unscaled_features = np.array(intermediate_output)
        # # pd.to_pickle(hits_channel2_df_dbscan, r"C:\Users\Martin-PC\Magistrska Matlab\CFE_hits_channel2_df_labels_ch1_dbscan.pkl")       


        # # labels_true_input_cwt05_pure_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_True_ctw'])
        # labels_only_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_DBscan'])
        # x_pure_pred_enc_df_dbscan_filtered.drop(x_pure_pred_enc_df_dbscan_filtered.columns[np.arange(x_pure_pred_enc_df_dbscan_filtered.shape[-1]-1,x_pure_pred_enc_df_dbscan_filtered.shape[-1])], axis = 1, inplace = True)
        # x_pure_pred_enc_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered)     
        # X=x_pure_pred_enc_dbscan
        # print(len(x_pure_pred_enc_df)-len(x_pure_pred_enc_dbscan))
        
        # intermediate_output = X

        range_n_clusters = [n_clusters]
        silhouette_avg_list_kmeans = []
        clusterer = KMeans(n_clusters=n_clusters, random_state=2)
        cluster_labels = clusterer.fit_predict(intermediate_output)
        labels_true = cluster_labels

        plt.style.use('dark_background')

        #classical features
        dim_x = 0 #energija CFE
        dim_y = 1# globoka 1 CFE
        dim_z = 2 # FCOG CFE
        
        fig1 = plt.figure(2, figsize=(11.5, 11.5))
        ax = Axes3D(fig1, rect=[0, 0, .95, 1], elev=10, azim=26, auto_add_to_figure=False)
        fig1.add_axes(ax)
        unsupervised_labels = np.choose(labels_true, [0, 1, 2,3,4,5,6,7]).astype(float) #pac channel 1

        mapping = {# old: new
          0: 19,
          1: 19,
          2: 19,
          3: 19,
          4: 19,
          5: 19}
        
        # viridis_cols = ['#440154','#365c8d','#1fa187','#dde318','tomato','magenta']
        viridis_cols = ['#440154','#365c8d','#1fa187','#dde318','tomato','magenta']
        # viridis_cols = ['tomato','#365c8d','#1fa187']
        
        mapping_colors = {# old: new
          0: viridis_cols[0],
          1: viridis_cols[1],
          2: viridis_cols[2],
            3: viridis_cols[3],
            4: viridis_cols[4],
            5: viridis_cols[5]
          }
        

        class_marker_sizes = [mapping.get(number, number) for number in unsupervised_labels]
        scatter =  ax.scatter(intermediate_output[:, dim_x], intermediate_output[:, dim_y], intermediate_output[:,dim_z], c=unsupervised_labels,cmap = mcolors.ListedColormap(viridis_cols), label = unsupervised_labels, edgecolors='none', s=class_marker_sizes)
        ax.tick_params(axis = 'both', labelsize=20)
        plt.legend(handles=scatter.legend_elements()[0], 
                # labels=['Matrix cracking','Fiber breakage','Fiber-matrix debonding','Fiber pullout'], loc='right',
                labels=['Cluster '+str(x) for x in range(1,n_clusters+1)], loc='right',
                bbox_to_anchor=(0.37, 0.36, 0.54, 0.5),
                fontsize = 18,
                title="Legend")
        
        
        plt.rcParams['legend.title_fontsize'] = 21
        
        plt.style.use('default')

        silhouette_avg = silhouette_score(intermediate_output, cluster_labels)
        print("Kmeans","For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        silhouette_avg_list_kmeans.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(intermediate_output, cluster_labels)
        append_list_as_row(Output_filename,  ["Kmeans","For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg])
        y_lower = 10

        # print(performance_metrics)
        fig, (ax1) = plt.subplots()
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(intermediate_output) + (n_clusters + 1) * 10])
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=viridis_cols[i], edgecolor=viridis_cols[i], alpha=0.7)
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                      fontsize=14, fontweight='bold')
        
        plt.show()
        plt.close() 
            
        loss = np.round(loss, 5)

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
#%% cwtms cluster avgs
from scipy.ndimage import zoom
filename = '45-02s-11'
cwtms = pd.read_pickle(r"{filename}_32_512.pkl".format(filename=filename))
# cwtms_bio_cold1_features = pd.read_pickle(r"classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
classical_features = pd.read_pickle(r"classical_features_w7_spectrum_energy_all_80_{filename}_80_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
# cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
# cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
classical_features['Relative time [%]'] = (classical_features['time']/max(classical_features['time']))*100
# counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()
# indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=20) & (cwtms_bio_cold1_features['Relative time [%]']<= 80))[0])
#
indices1=list(np.where((classical_features['Relative time [%]']>=50))[0])
classical_features = classical_features.iloc[indices1]
cwtms = cwtms[indices1,:,:]

cwtms = cwtms.astype(np.float64)

y_dim_height = 32
x_dim_length = 64
zoom_factors = (1,y_dim_height / cwtms.shape[1], x_dim_length / cwtms.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms = zoom(cwtms, zoom_factors, order=1)
BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

cwtms = cwtms.reshape(cwtms.shape + (1,))
x_pure_pred_enc = model.predict(cwtms)

n_clusters = 3
clusterer = KMeans(n_clusters=n_clusters, random_state=2)
# clusterer = SpectralClustering(n_clusters=n_clusters,assign_labels='discretize',random_state=0).fit(intermediate_output)
unsupervised_labels = clusterer.fit_predict(x_pure_pred_enc)

# filename = '45-02s-13'
cwtms = pd.read_pickle("{filename}_32_512.pkl".format(filename=filename))
cwtms_clusters = [None]*n_clusters
cwtms_clusters_means = [None]*n_clusters
classical_features['label_unsupervised'] = unsupervised_labels
classical_features['label_unsupervised'] = ["Cluster "+str(x+1) for x in classical_features['label_unsupervised'] ]
classical_features = classical_features.reset_index(drop=True)
for indc,cluster_id in enumerate(['Cluster {}'.format(i+1) for i in range(n_clusters)]):
    classical_features_cluster=classical_features[classical_features['label_unsupervised'].isin([cluster_id])]
    cwtms_clusters[indc] = cwtms[classical_features_cluster.index,:,:]
    cwtms_clusters_means[indc] = np.mean(cwtms_clusters[indc], axis=0)
    BatchesLength = 512
    BatchesSliceHeight = 32
    fs=952381
    dt = 1/fs
    previous_time_data = np.arange(0,BatchesLength)*dt
    freq = np.linspace(1, fs, BatchesSliceHeight) 
    plt.pcolormesh(previous_time_data, freq, cwtms_clusters_means[indc], cmap='viridis', shading='gouraud')
    plt.title(str(cluster_id))
    plt.show()
#%%
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
save_dir = save_dir_root + '\Clustering_CAE_encoder_clustering_{identifier}_3_clusters_{filename}_v1'.format(filename=filename,identifier=identifier) +  ver_str + '.h5'
model.save_weights(save_dir)

#% Training CAE

# loss = encoder.history['loss']
# val_loss = encoder.history['val_loss']
# epochs_range = range(train_epochs)
# plt.figure()
# plt.plot(epochs_range, loss, 'bo', label='Training loss')
# plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
#%% adding classical features CFE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
from scipy.ndimage import zoom
filename = '45-02s-13'
cwtms_bio_cold1 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_bio_cold1_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
# cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
# cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
cwtms_bio_cold1_features['Relative time [%]'] = (cwtms_bio_cold1_features['time']/max(cwtms_bio_cold1_features['time']))*100
# counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()
# indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=20) & (cwtms_bio_cold1_features['Relative time [%]']<= 80))[0])
#
indices=list(np.where((cwtms_bio_cold1_features['Relative time [%]']>=50))[0])
cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[indices]
cwtms_bio_cold1 = cwtms_bio_cold1[indices,:,:]
#%%
filename = '45-02s-11'
cwtms_bio_cold2 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_bio_cold2_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# cwtms_bio_cold2_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold2_features)
# cwtms_bio_cold2 = cwtms_bio_cold2[cwtms_bio_cold2_features_indices,:,:]
# cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[cwtms_bio_cold2_features_indices,:]
cwtms_bio_cold2_features['Relative time [%]'] = (cwtms_bio_cold2_features['time']/max(cwtms_bio_cold2_features['time']))*100
#
indices=list(np.where((cwtms_bio_cold2_features['Relative time [%]']>=50))[0])
cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[indices]
cwtms_bio_cold2 = cwtms_bio_cold2[indices,:,:]
#%%
indices_crop = indices
# filename = '45-02s-13'
cwtms_export_merged_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_32_512.pkl".format(filename=filename))
cwtms_export_merged_features = cwtms_export_merged_features.astype(np.float64)
cwtms_export_merged_features = cwtms_export_merged_features[indices_crop,:,:]

y_dim_height = 32
x_dim_length = 64
zoom_factors = (1,y_dim_height / cwtms_export_merged_features.shape[1], x_dim_length / cwtms_export_merged_features.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_export_merged_features = zoom(cwtms_export_merged_features, zoom_factors, order=1)
BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

cwtms_export_merged_features = cwtms_export_merged_features.reshape(cwtms_export_merged_features.shape + (1,))
x_pure_pred_enc = encoder.predict(cwtms_export_merged_features)
# classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename))
classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_w7_spectrum_energy_all_80_{filename}_80_Infinite_BatchLength.pkl".format(filename=filename))
classical_features_filtered_hits_tok = classical_features_filtered_hits_tok.iloc[indices_crop]
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
save_dir = save_dir_root + '\merged_features_encoder_clustering_{identifier}_3_clusters_v1_{filename}_32x64_energy_all'.format(filename=filename,identifier=identifier) +  ver_str + '.pkl'
pd.to_pickle(merged_features_embedded_classical, save_dir)
# save_dir = save_dir_root + '\merged_features_embedded_classical_rho20_lambda_cfe_cold' +  ver_str + '.xlsx'
# merged_features_embedded_classical.to_excel(save_dir, index = False) 

# merged_features_embedded_classical = pd.read_excel(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.xlsx")
# save_dir = save_dir_root + '\merged_features_embedded_classical_ch2' +  ver_str + '.pkl'
# pd.to_pickle(merged_features_embedded_classical, save_dir)
# cfe_cold
#%%encoder linear regression training
cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0).reset_index(drop=True)
x1_training_input_batch = cwtms_bio_cold_input
batch_size_samples = 1000
index = 0
index_array = np.arange(x1_training_input_batch.shape[0])
# print(len(idx))  # Check the length
# print(len(set(idx)))  # Check the number of unique values
# print(cwtms_bio_cold_features.index.is_unique)

np.random.shuffle(index_array)
idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, x1_training_input_batch.shape[0])]
x1_validation_input_batch = x1_training_input_batch[idx]
x1_validation_input_batch_features = cwtms_bio_cold_features.iloc[idx]

# Assuming x1_training_input_batch and cwtms_bio_cold_features are numpy arrays
x1_training_input_batch = np.delete(x1_training_input_batch, idx, axis=0)

# If x1_training_input_batch and cwtms_bio_cold_features are pandas dataframes
cwtms_bio_cold_features = cwtms_bio_cold_features.drop(cwtms_bio_cold_features.iloc[list(idx)].index)


encoder_linear_regression.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 20
# train_epochs = 5
# batch_size = 20
# batch_size = 128
# batch_size = 64
# batch_size = 64
# batch_size = 16
batch_size = 32
encoder_linear_regression = encoder_linear_regression.fit(x1_training_input_batch, 
                                    cwtms_bio_cold_features['Relative time [%]'],
                                    batch_size=batch_size, epochs=train_epochs,shuffle=True,
                                    validation_data=(x1_validation_input_batch, x1_validation_input_batch_features['Relative time [%]']), # data for validation
                                    # steps_per_epoch=len(x1_training_input_batch) // (batch_size*2)
                                    )
# breakpoint()classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_w7_spectrum_energy_all_80_{filename}_80_Infinite_BatchLength.pkl".format(filename=filename))
#%% save weights
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
save_dir = save_dir_root + '\encoder_linear_regression_32x64_v1' +  ver_str + '.h5'
encoder_linear_regression.save_weights(save_dir)
#%%
loss = encoder_linear_regression.history['loss']
val_loss = encoder_linear_regression.history['val_loss']
epochs_range = range(train_epochs)
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
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
