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
layers_depths=[128, 128, 512, 256, 256, 128],
layer_depth_latent_cnn=20
dropout_ratio = 0.01

# layers_depths=[30, 60, 64, 6] #nujno bottleneck = stevilo clustrov
# layer_depth_latent_cnn = 12


# Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Silhuette_clustering_analysis_re"
# Output_filename = Output + ver_str + ".csv"

 # Load the pretrained Xception model without the top (classification) layer
# tf.compat.v1.disable_eager_execution()
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the pretrained layers so they're not updated during training
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
# Check if the weights have been loaded correctly
weights_loaded = base_model.get_weights()

# Check the first weight tensor for example
print(weights_loaded[0])

# You can also print the shape of all weight tensors
for i, weight_tensor in enumerate(weights_loaded):
    print(f"Weight tensor {i}: {weight_tensor.shape}")


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
cwtms_bio_cold1 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_128_512.pkl".format(filename=filename))
cwtms_bio_cold1_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_w7_cb_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
cwtms_bio_cold1_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold1_features)
cwtms_bio_cold1 = cwtms_bio_cold1[cwtms_bio_cold1_features_indices,:,:]
cwtms_bio_cold1_features = cwtms_bio_cold1_features.iloc[cwtms_bio_cold1_features_indices,:]
cwtms_bio_cold1_features['Relative time [%]'] = (cwtms_bio_cold1_features['time']/max(cwtms_bio_cold1_features['time']))*100
# counts = cwtms_bio_cold1_features_indices['cluster'].value_counts()


# cwtms_bio_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_bio_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_cold_squared_scaled_w4_32_64.pkl")
# cwtms_cfe_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_zeta_lower_cfe_tok_squared_scaled_w4_32_64.pkl")
# cwtms_cfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_cfe-100_80_288.pkl")
# cwtms_gfe = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\cwtms_gfe-20_80_288.pkl")

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
filename = '45-02s-11'
cwtms_bio_cold2 = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}_128_512.pkl".format(filename=filename))
cwtms_bio_cold2_features = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_{filename}_60_Infinite_BatchLength.pkl".format(filename=filename)).reset_index(drop=True)
cwtms_bio_cold2_features_indices = balance_dataset_clustering(['FVTA'],4,6000,cwtms_bio_cold2_features)
cwtms_bio_cold2 = cwtms_bio_cold2[cwtms_bio_cold2_features_indices,:,:]
cwtms_bio_cold2_features = cwtms_bio_cold2_features.iloc[cwtms_bio_cold2_features_indices,:]
cwtms_bio_cold2_features['Relative time [%]'] = (cwtms_bio_cold2_features['time']/max(cwtms_bio_cold2_features['time']))*100
from scipy.ndimage import zoom
cwtms_bio_cold2 = cwtms_bio_cold2.astype(np.float64)
y_dim_height = 128
x_dim_length = 128
zoom_factors = (1,y_dim_height / cwtms_bio_cold2.shape[1], x_dim_length / cwtms_bio_cold2.shape[2])

# Perform bilinear interpolation on all images in the array
cwtms_bio_cold_resized = zoom(cwtms_bio_cold2, zoom_factors, order=1)
cwtms_bio_cold2 = cwtms_bio_cold_resized
#%

BatchesSliceHeight = y_dim_height
BatchesLength = x_dim_length

# cwtms_bio_cold_input2 = cwtms_bio_cold2.reshape(cwtms_bio_cold2.shape + (1,))
cwtms_bio_cold_input2 = cwtms_bio_cold2
#%%
# cwtms_bio_cold = np.concatenate((cwtms_bio_cold1,cwtms_bio_cold2),axis=0)
cwtms_bio_cold_input = np.concatenate((cwtms_bio_cold_input1,cwtms_bio_cold_input2),axis=0)
x1_training_input_batch = cwtms_bio_cold_input
cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0)


#%%encoder linear regression training
cwtms_bio_cold_features = pd.concat([cwtms_bio_cold1_features,cwtms_bio_cold2_features],axis=0).reset_index(drop=True)
del cwtms_bio_cold1_features
del cwtms_bio_cold2_features
del cwtms_bio_cold_input1
del cwtms_bio_cold_input2
del cwtms_bio_cold_resized
x1_training_input_batch = cwtms_bio_cold_input
del cwtms_bio_cold_input
del cwtms_bio_cold1
del cwtms_bio_cold2
# del x1_training_input_batch
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
#%%
model.compile(optimizer='adam', loss='mse') # adam nujno, mse je boljš kt binary_crossentropy
train_epochs = 20
# train_epochs = 5
# batch_size = 20
# batch_size = 128
# batch_size = 64
# batch_size = 64
# batch_size = 16
batch_size = 32
encoder_linear_regression = model.fit(x1_training_input_batch, 
                                    cwtms_bio_cold_features['Relative time [%]'],
                                    batch_size=batch_size, epochs=train_epochs,shuffle=True,
                                    validation_data=(x1_validation_input_batch, x1_validation_input_batch_features['Relative time [%]']), # data for validation
                                    # steps_per_epoch=len(x1_training_input_batch) // (batch_size*2)
                                    )
breakpoint()
#%% 100 epoch
# 1309/1309 [==============================] - 103s 79ms/step - loss: 1111.2642 - val_loss: 1103.0613
# breakpoint()classical_features_filtered_hits_tok = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_w7_spectrum_energy_all_80_{filename}_80_Infinite_BatchLength.pkl".format(filename=filename))
#%%
x1_validation_input_batch_predicted = encoder_linear_regression.predict(x1_validation_input_batch)
loss = encoder_linear_regression_train.history['loss']
val_loss = encoder_linear_regression_train.history['val_loss']
epochs_range = range(train_epochs)
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%% save weights
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_CAE'
save_dir = save_dir_root + '\encoder_linear_regression_128x256_v1' +  ver_str + '.h5'
encoder_linear_regression.save_weights(save_dir)
#%%


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
#

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
save_dir = save_dir_root + '\\encoder_linear_regression_128x256_v1_32_32_12_512_6_100_32' + '.h5'
encoder_linear_regression.load_weights(save_dir)

x1_validation_input_batch_predicted = encoder_linear_regression.predict(x1_validation_input_batch)
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
