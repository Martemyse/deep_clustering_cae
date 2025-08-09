# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:43:59 2023

@author: Martin-PC
"""

import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

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


# Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Silhuette_clustering_analysis_re"
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
# breakpoint()
print(model.summary())
