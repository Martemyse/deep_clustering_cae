# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:55:54 2023

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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose, Dropout, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from keras import optimizers
import numpy as np
import pandas as pd
from time import time
# from keras.engine.topology import Layer, InputSpec
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
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionV3
import tensorflow as tf
# import keras.backend as K
# K.clear_session()
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

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
# layers_depths=[128, 128, 512, 256, 256, 128],
# layer_depth_latent_cnn=20
# dropout_ratio = 0.01

# layers_depths=[30, 60, 64, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12


# Output = r"Silhuette_clustering_analysis_re"
# Output_filename = Output + ver_str + ".csv"

 # Load the pretrained Xception model without the top (classification) layer
# tf.compat.v1.disable_eager_execution()
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the pretrained layers so they're not updated during training
# for layer in base_model.layers:
#     layer.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(1, activation='linear')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# model.summary()
# # Check if the weights have been loaded correctly
# weights_loaded = base_model.get_weights()

# # Check the first weight tensor for example
# print(weights_loaded[0])

# # You can also print the shape of all weight tensors
# for i, weight_tensor in enumerate(weights_loaded):
#     print(f"Weight tensor {i}: {weight_tensor.shape}")


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
filename = '45-02s-13'
cwtms_bio_cold1 = pd.read_pickle("{filename}_128_512.pkl".format(filename=filename))
cwtms_bio_cold1_features = pd.read_pickle("classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
cwtms_bio_cold1_features['Relative time [%]'] = (cwtms_bio_cold1_features['time']/max(cwtms_bio_cold1_features['time']))*100
# counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()


# cwtms_bio_tok = pd.read_pickle("cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_cold = pd.read_pickle("cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_tok = pd.read_pickle("cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe = pd.read_pickle("cwtms_cfe-100_80_288.pkl")
# cwtms_gfe = pd.read_pickle("cwtms_gfe-20_80_288.pkl")

from scipy.ndimage import zoom
cwtms_bio_cold1 = cwtms_bio_cold1.astype(np.float64)
y_dim_height = 128
x_dim_length = 128
zoom_factors = (1,y_dim_height / cwtms_bio_cold1.shape[1], x_dim_length / cwtms_bio_cold1.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_bio_cold_resized = zoom(cwtms_bio_cold1, zoom_factors, order=1)
cwtms_bio_cold1 = cwtms_bio_cold_resized
#%

BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

# cwtms_bio_cold_input1 = cwtms_bio_cold1.reshape(cwtms_bio_cold1.shape + (1,))
cwtms_bio_cold_input1 = cwtms_bio_cold1

#%%
# filename = '45-02s-11'
# cwtms_bio_cold2 = pd.read_pickle("{filename}_128_512.pkl".format(filename=filename))
# cwtms_bio_cold2_features = pd.read_pickle("classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
# cwtms_bio_cold2_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold2_features)
# cwtms_bio_cold2 = cwtms_bio_cold2[cwtms_bio_cold2_features_indices,:,:]
# cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[cwtms_bio_cold2_features_indices,:]
# cwtms_bio_cold2_features['Relative time [%]'] = (cwtms_bio_cold2_features['time']/max(cwtms_bio_cold2_features['time']))*100
# from scipy.ndimage import zoom
# cwtms_bio_cold2 = cwtms_bio_cold2.astype(np.float64)
# y_dim_height = 128
# x_dim_length = 128
# zoom_factors = (1,y_dim_height / cwtms_bio_cold2.shape[1], x_dim_length / cwtms_bio_cold2.shape[2])

# # Perform bilinear interpolation on all images in the array
# cwtms_bio_cold_resized = zoom(cwtms_bio_cold2, zoom_factors, order=1)
# cwtms_bio_cold2 = cwtms_bio_cold_resized
# #%

# BatchesSliceHeight = y_dim_height
# BatchesLength = x_dim_length

# # cwtms_bio_cold_input2 = cwtms_bio_cold2.reshape(cwtms_bio_cold2.shape + (1,))
# cwtms_bio_cold_input2 = cwtms_bio_cold2
#%%
# cwtms_bio_cold = np.concatenate((cwtms_bio_cold1,cwtms_bio_cold2),axis=0)
# cwtms_bio_cold_input = np.concatenate((cwtms_bio_cold_input1,cwtms_bio_cold_input2),axis=0)
# x1_training_input_batch = cwtms_bio_cold_input
# cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0)

cwtms_bio_cold_input = cwtms_bio_cold_input1
x1_training_input_batch = cwtms_bio_cold_input
cwtms_bio_cold_features = cwtms_bio_cold1_features


#%%model linear regression training
# cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0).reset_index(drop=True)
# del cwtms_bio_cold1_features
# del cwtms_bio_cold2_features
# del cwtms_bio_cold_input1
# del cwtms_bio_cold_input2
# del cwtms_bio_cold_resized
# x1_training_input_batch = cwtms_bio_cold_input
# del cwtms_bio_cold_input
# del cwtms_bio_cold1
# del cwtms_bio_cold2
#%%
del cwtms_bio_cold1_features
del cwtms_bio_cold_input1
del cwtms_bio_cold_resized
x1_training_input_batch = cwtms_bio_cold_input
del cwtms_bio_cold_input
del cwtms_bio_cold1
#%%

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

x1_training_input_batch = np.stack((x1_training_input_batch,) * 3, axis=-1)
x1_training_input_batch *= 255
x1_training_input_batch = x1_training_input_batch.astype(np.uint8)

x1_validation_input_batch = np.stack((x1_validation_input_batch,) * 3, axis=-1)
x1_validation_input_batch *= 255
x1_validation_input_batch = x1_validation_input_batch.astype(np.uint8)
# x1_training_input_batch *= 255
# x1_training_input_batch = np.expand_dims(x1_training_input_batch, axis=-1)
# x1_training_input_batch = np.repeat(x1_training_input_batch, 3, axis=-1)
#%% retraining model
# save_dir = save_dir_root + '\\Clustering_CAE_model_clustering_chatGPT_32x64_6_clusters_all_files_v1_36_36_16_512_6_10_32' + '.h5'
# model.load_weights(save_dir)
# #
# n_clusters = 6
# clustering_layer = ClusteringLayer(n_clusters, name='clustering')(model.output)
# model = Model(inputs=model.input, outputs=clustering_layer)
# for layer in model.layers[:]:
#         layer.trainable = True
        
# # opt_clustering = optimizers.SGD(learning_rate=0.00008)
# opt_clustering = keras.optimizers.Adam(learning_rate=0.0001) #šel vseh 4000 iteracij
# # opt_clustering = keras.optimizers.Adam(learning_rate=0.00005)

# model.compile(optimizer=opt_clustering, loss='mse')
#%% CLustering Layer Initialization

n_clusters = 5

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from keras import optimizers
import numpy as np
import pandas as pd
from time import time
from tensorflow.python.keras.layers import Layer, InputSpec
from keras.optimizers import SGD
# automodel.load_weights(save_dir+'\cae_v2_weights_40_10_34_434.h5')
# automodel, model, model_linear_regression = automodelConv2D_1()
# automodel.summary()

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
        input_dim = input_shape[1]
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

# train_epochs = 15
# batch_size = 64
# x1_training_input_batch = x1_training_input_batch
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False

# Reshape the output tensor of the base model
reshaped_output = GlobalAveragePooling2D()(base_model.output)


# # Add additional layers if needed
x = Dense(128, activation='relu')(reshaped_output)
clustering_layer_dense = Dense(1, activation='linear')(x)

# Add your clustering layer
clustering_layer = ClusteringLayer(n_clusters=n_clusters)(clustering_layer_dense)

# Create the model
# model = Model(inputs=base_model.input, outputs=predictions)
model = Model(inputs=base_model.input, outputs=clustering_layer)
model.summary()

#%%
# x1_training_input_batch = x1_training_input_batch.reshape(x1_training_input_batch.shape + (1,))


# opt_clustering = optimizers.SGD(learning_rate=0.000001)
# opt_clustering = keras.optimizers.Adam(learning_rate=0.0001) #šel vseh 4000 iteracij
opt_clustering = keras.optimizers.Adam(learning_rate=0.00008)

model.compile(optimizer=opt_clustering, loss='mse')

# x_pure_input_scaled_pca = features_ch2_pca
clustering_alg = KMeans(n_clusters=n_clusters)
# from sklearn.cluster import SpectralClustering
# clustering_alg = SpectralClustering(n_clusters=n_clusters)

x_pure_pred_enc = model.predict(x1_training_input_batch)

X = x_pure_pred_enc

# db = DBSCAN(eps=6.9, min_samples=100).fit(X) # Robust Scaler, klasične značilke
db = DBSCAN(eps=1.9, min_samples=100).fit(X) # Robust Scaler, standardScaler, PCA

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_dbscan = db.labels_

dim_x = 2 #energija CFE
dim_y = 3# globoka 1 CFE
dim_z = 0 # FCOG CFE

plt.style.use('dark_background')
fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=24, auto_add_to_figure=False)
fig.add_axes(ax)

scatter = ax.scatter(X[:, dim_x], X[:, dim_y], X[:, dim_z], c=labels_dbscan, cmap='viridis', label = labels_dbscan, edgecolors='none') #GFE
ax.tick_params(axis = 'both', labelsize=16)
plt.rcParams['legend.title_fontsize'] = 22
plt.title('DBScan filtering')
plt.show() 
x_pure_pred_enc_df = pd.DataFrame(X)
x_pure_pred_enc_df['labels_DBscan'] =  labels_dbscan
x_pure_pred_enc_df_dbscan_filtered = x_pure_pred_enc_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index)

# hits_channel2_df.reset_index(inplace=True, drop=True)
# X = features_merged_labels_pure_selected_deep_FVTA_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
# features_merged_dropped_dbscan = features_merged.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
x_pure_pred_enc_unscaled_features = np.array(X)
# pd.to_pickle(hits_channel2_df_dbscan, r"C:\Users\Martin-PC\Magistrska Matlab\CFE_hits_channel2_df_labels_ch1_dbscan.pkl")       

# labels_true_input_cwt05_pure_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_True_ctw'])
labels_only_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_DBscan'])
x_pure_pred_enc_df_dbscan_filtered.drop(x_pure_pred_enc_df_dbscan_filtered.columns[np.arange(x_pure_pred_enc_df_dbscan_filtered.shape[-1]-1,x_pure_pred_enc_df_dbscan_filtered.shape[-1])], axis = 1, inplace = True)
x_pure_pred_enc_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered)     
X=x_pure_pred_enc_dbscan
print(len(x_pure_pred_enc_df)-len(x_pure_pred_enc_dbscan))
x_pure_pred_enc = X

y_pred = clustering_alg.fit_predict(x_pure_pred_enc)
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([clustering_alg.cluster_centers_])
model.summary()
model_weights = model.get_weights()
# breakpoint()
#%%
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
ver_str = 'InceptionV3'
save_dir_root = r'Clustering_CAE'
save_dir = save_dir_root + '\model_linear_regression_128x256_v1' +  ver_str + '.h5'
Output = r"Clustering_analysis"
Output_filename = Output + ver_str + ".csv"

x1_training_input_batch = x1_training_input_batch
filename = 'all_files'
# filename = '45-02s-13'
# identifier = '16_8_16_512_3_30_256'
identifier = 'model_clustering_chatGPT'
loss = 0
index = 0
# maxiter = 500
maxiter = 8000
# maxiter = 12000
# maxiter = 1
update_interval = 1000
batch_size = 8
index_array = np.arange(x1_training_input_batch.shape[0])
tol = 0.00001 # tolerance threshold to stop training
# labels_true_3_ctw_w12 = labels_true_3_ctw_w12[0:len(x1_training_input_batch)]
layer_names = [layer.name for layer in model.layers]

# x_input = x1_training_input_batch # ali x3a --> vsi
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[-2]).output)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        output = model.predict(x1_training_input_batch, verbose=0)
        output_distribution = target_distribution(output)  # update the auxiliary target distribution p
        # evaluate the clustering performance
        # output from intermediate layer
        intermediate_output = intermediate_layer_model.predict(x1_training_input_batch)  
        y_pred = output.argmax(1)
        y_pred_auxilary = output_distribution.argmax(1)
        cluster_nums = [n_clusters]
        # breakpoint()

        # import matplotlib.colors as mcolors
        # from sklearn.cluster import SpectralClustering
        # X = intermediate_output
        #% DBSCAN FILTERING
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN


        # db = DBSCAN(eps=6.9, min_samples=105).fit(X) # brez PCA

        # db = DBSCAN(eps=0.9, min_samples=100).fit(X) # PCA
        

        # db = DBSCAN(eps=6.9, min_samples=100).fit(X) # Robust Scaler, klasične značilke
        
        db = DBSCAN(eps=1.9, min_samples=100).fit(X) # Robust Scaler, standardScaler, PCA

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels_dbscan = db.labels_

        dim_x = 2 #energija CFE
        dim_y = 3# globoka 1 CFE
        dim_z = 0 # FCOG CFE

        plt.style.use('dark_background')
        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=24, auto_add_to_figure=False)
        fig.add_axes(ax)
        # for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3),('Category4', 4),('Category5', 5),('Category6', 6),('Category7', 7),('Category8', 8),('Category9', 9),('Category10', 10),('Category11', 11),('Category12', 12),('Category13', 13)]:
        #     ax.text3D(X[labels_dbscan == label, 0].mean(),
        #               X[labels_dbscan == label, 1].mean(),
        #               X[labels_dbscan == label, 2].mean(), 
        #               name,
        #               bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
        #               )
        # # unsupervised_labels = np.choose(labels_dbscan, [1, 2, 0,3,0,0,0,0,0,0,0]).astype(float)
        # ax.scatter(x_pure_pred_enc[:, 1], x_pure_pred_enc[:, 5], x_pure_pred_enc[:, 6], c=labels_kmeans_3)
        scatter = ax.scatter(X[:, dim_x], X[:, dim_y], X[:, dim_z], c=labels_dbscan, cmap='viridis', label = labels_dbscan, edgecolors='none') #GFE
        # plt.xlim(-2,4)  
        # plt.ylim(-4,3) 
        # ax.set_zlim(-2,4) 
        # ax.scatter(X[:, 0], X[:, 1], X[:,4], c=labels_dbscan) #CFE
        # plt.xlim(-2,4)  
        # plt.ylim(-4,4) 
        # ax.set_zlim(-2,3) 
        ax.tick_params(axis = 'both', labelsize=16)
        # plt.xlim(0,8)  # cluster belongings, K-means point of view 0-1-2
        # plt.ylim(-6,2) 
        # ax.set_zlim(-3,0)

        # plt.legend(handles=scatter.legend_elements()[0], 
        #         labels=['0-200 kHz','200-400 kHz','400-1000 kHz'], loc='best', bbox_to_anchor=(0.0, 0., 0.3, 0.1),fontsize = 16,
        #         title="Legenda")

        plt.rcParams['legend.title_fontsize'] = 22
        # ax.set_xlabel('Globoka značilka 2', fontsize=20)
        # ax.set_ylabel('Globoka značilka 3')
        # ax.set_zlabel('Globoka značilka 4', fontsize=30, rotation = 0)
        # ax.xaxis.set_label_coords(1.55, -0.55)
        plt.title('DBScan filtering')
        plt.show() 
                # print(np.round(accuracy_score(labels_true_3_ctw_w12, labels_kmeans_3), 3))
        # print(sum(labels_dbscan))
        #%
        x_pure_pred_enc_df = pd.DataFrame(X)
        x_pure_pred_enc_df['labels_DBscan'] =  labels_dbscan
        # x_pure_pred_enc_df['labels_True_ctw'] =  labels_true


        x_pure_pred_enc_df_dbscan_filtered = x_pure_pred_enc_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index)

        # hits_channel2_df.reset_index(inplace=True, drop=True)
        # X = features_merged_labels_pure_selected_deep_FVTA_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
        # features_merged_dropped_dbscan = features_merged.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
        x_pure_pred_enc_unscaled_features = np.array(X)
        # pd.to_pickle(hits_channel2_df_dbscan, r"C:\Users\Martin-PC\Magistrska Matlab\CFE_hits_channel2_df_labels_ch1_dbscan.pkl")       


        # labels_true_input_cwt05_pure_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_True_ctw'])
        labels_only_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_DBscan'])
        x_pure_pred_enc_df_dbscan_filtered.drop(x_pure_pred_enc_df_dbscan_filtered.columns[np.arange(x_pure_pred_enc_df_dbscan_filtered.shape[-1]-1,x_pure_pred_enc_df_dbscan_filtered.shape[-1])], axis = 1, inplace = True)
        x_pure_pred_enc_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered)     
        X=x_pure_pred_enc_dbscan
        print(len(x_pure_pred_enc_df)-len(x_pure_pred_enc_dbscan))
        
        intermediate_output = X
        # breakpoint()
        

        range_n_clusters = [n_clusters]
        silhouette_avg_list_kmeans = []
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=2)
            cluster_labels = clusterer.fit_predict(intermediate_output)
            labels_true = cluster_labels

            plt.style.use('dark_background')

            #classical features
            dim_x = 0 #energija CFE
            dim_y = 2# globoka 1 CFE
            dim_z = 3 # FCOG CFE
            
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
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    # if index == 0:
    #     np.random.shuffle(index_array)
    idx = index_array[index * batch_size: min((index+1) * batch_size, x1_training_input_batch.shape[0])]
    x_input_batch = x1_training_input_batch[idx]
    x_input_batch_3d = x1_training_input_batch[idx][:,:,:,0]
    y_desired_output_distribution = output_distribution[idx]
    try:
        loss = model.train_on_batch(x=x1_training_input_batch[idx], y=output_distribution[idx])
    except:
        print('Train_on_batch_failed' + str(index))
        breakpoint()
        pass
    index = index + 1 if (index + 1) * batch_size <= x1_training_input_batch.shape[0] else 0

# model.save_weights(save_dir + '\clustering_v2_weights_30_3_34_434.h5')
# model.load_weights(save_dir + '\clustering_v2_weights_30_3_34_434.h5')
breakpoint()
#%%
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'Clustering_CAE'
save_dir = save_dir_root + '\Clustering_CAE_model_clustering_chatGPT_32x64_6_clusters_{filename}_v2'.format(filename=filename) +  ver_str + '.h5'
model.save_weights(save_dir)

#% Training CAE

# loss = model.history['loss']
# val_loss = model.history['val_loss']
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
cwtms_export_merged_features = pd.read_pickle("{filename}_128_512.pkl".format(filename=filename))
cwtms_export_merged_features = cwtms_export_merged_features.astype(np.float64)
y_dim_height = 128
x_dim_length = 64
zoom_factors = (1,y_dim_height / cwtms_export_merged_features.shape[1], x_dim_length / cwtms_export_merged_features.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_export_merged_features = zoom(cwtms_export_merged_features, zoom_factors, order=1)
BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

cwtms_export_merged_features = cwtms_export_merged_features.reshape(cwtms_export_merged_features.shape + (1,))
x_pure_pred_enc = model.predict(cwtms_export_merged_features)
# classical_features_filtered_hits_tok = pd.read_pickle("classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename))
classical_features_filtered_hits_tok = pd.read_pickle("classical_features_w7_spectrum_energy_all_80_{filename}_80_Infinite_BatchLength.pkl".format(filename=filename))
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
save_dir = save_dir_root + '\merged_features_model_clustering_chatGPT_32x64_6_clusters_v1_{filename}_32x64_energy_all'.format(filename=filename) +  ver_str + '.pkl'
pd.to_pickle(merged_features_embedded_classical, save_dir)
# save_dir = save_dir_root + '\merged_features_embedded_classical_rho20_lambda_cfe_cold' +  ver_str + '.xlsx'
# merged_features_embedded_classical.to_excel(save_dir, index = False) 

# merged_features_embedded_classical = pd.read_excel("merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.xlsx")
# save_dir = save_dir_root + '\merged_features_embedded_classical_ch2' +  ver_str + '.pkl'
# pd.to_pickle(merged_features_embedded_classical, save_dir)
# cfe_cold