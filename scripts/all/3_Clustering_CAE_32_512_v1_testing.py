import os
# import keras
# from keras import layers
import numpy as np
# from keras.callbacks import TensorBoard
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import Lambda
# from ExtractData import ExtractData
import pandas as pd
from functools import reduce
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Softmax, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
import tensorflow as tf

# y = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_true_3.pkl")
# x = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_filtered_hits_ch1_w11_raw_fl16_std_48_432.pkl")
#%
# delitelj = 2
# stevilo_slik = int(x.shape[0]/delitelj)
# x1 = x[0:stevilo_slik,2:,:432]
# x1a = x1.reshape(x1.shape + (1,))

#%
# layers_depths=[60, 120, 128, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[40, 80, 512, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[4, 8, 56, 4] #nujno bottleneck = stevilo clustrov
layers_depths=[16, 8, 256, 6]
# layer_depth_latent_cnn = 8
# conv2Dfilters = [(3,3), (3,3)]
BatchesSliceHeight = 32
BatchesLength = 512
stride_pool_h = 2
stride_pool_w = 2
max_pool_filter_h = 2
max_pool_filter_w = 2

stride_norm_2dconv= 1 #vcasih 3
stride_latent_2dconv = 1 #vcasih 1
# stride_latent_max_pooling = 2 #vcasih 2

# def silhouette_loss(y_true, y_pred):
#     # Compute the silhouette score for each sample
#     x_encoded = encoder.predict(x)
#     y_pred = K.argmax(y_pred, axis=-1)
#     silhouette_scores = silhouette_score(x_encoded, y_pred, metric='euclidean')

#     # Return the negative mean silhouette score as the loss
#     return -K.mean(silhouette_scores)

# def silhouette_loss(y_true, y_pred):
#     # Compute the silhouette score for each sample
#     x_encoded = y_pred[0]  # extract the encoded features from the output
#     cluster_probs = y_pred[1]  # extract the cluster probabilities from the output
#     y_pred = tf.argmax(cluster_probs, axis=-1)
#     batch_size = tf.shape(x_encoded)[0]
#     silhouette_scores = tf.TensorArray(tf.float32, size=batch_size)
#     for i in tf.range(batch_size):
#         sample_features = x_encoded[i]
#         sample_cluster = y_pred[i]
#         cluster_mask = tf.equal(y_pred, sample_cluster)
#         sample_score = tf.cond(tf.reduce_any(cluster_mask),
#                                lambda: silhouette_score(sample_features.numpy(), y_pred.numpy(), metric='euclidean'),
#                                lambda: tf.constant(0.0))
#         silhouette_scores = silhouette_scores.write(i, sample_score)
#     silhouette_scores = silhouette_scores.stack()

#     # Return the negative mean silhouette score as the loss
#     return -tf.reduce_mean(silhouette_scores)

# def silhouette_loss(y_true, y_pred):
#     # Compute the silhouette score for each sample
#     x_encoded = ccae.encoder(y_true)
#     y_pred = K.argmax(y_pred, axis=-1)
#     silhouette_scores = silhouette_score(x_encoded, y_pred, metric='euclidean')

#     # Return the negative mean silhouette score as the loss
#     return -K.mean(silhouette_scores)

def silhouette_loss(y_true, y_pred):
    # Get encoded representation of input data
    x_encoded = ccae.encoder.predict(x)

    # Cluster encoded data using K-means
    kmeans = KMeans(n_clusters=4, random_state=0).fit(x_encoded)
    cluster_labels = kmeans.labels_

    # Compute silhouette score using cluster labels
    silhouette_scores = silhouette_score(x_encoded, cluster_labels, metric='euclidean')

    # Return the negative mean silhouette score as the loss
    return -K.mean(silhouette_scores)

inputs_1 = Input(shape = (BatchesSliceHeight,BatchesLength, 1))
slice_height = int(BatchesSliceHeight/4)

input_img = Input(shape=(BatchesSliceHeight, BatchesLength, 1))

# Encoding Layers
x = Conv2D(layers_depths[0], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(input_img)
x = MaxPooling2D((2, 2), padding='same', strides = (stride_pool_h,stride_pool_w))(x)
x = Conv2D(layers_depths[1], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(x)
x = MaxPooling2D((2, 2), padding='same', strides = (stride_pool_h,stride_pool_w))(x)
x = Conv2D(layers_depths[1], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(x)
encoded = MaxPooling2D((max_pool_filter_h,max_pool_filter_w), padding='same')(x)

# Clustering Layer
n_clusters = 4  # number of clusters
clustering_layer = Softmax(axis=1, name='clustering')(Dense(n_clusters, activation='softmax')(Flatten()(encoded)))

# Decoding Layers
x = Conv2D(layers_depths[1], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(encoded)
x = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(x)
x = Conv2D(layers_depths[1], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(x)
x = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(x)
x = Conv2D(layers_depths[0], (3, 3), activation='relu', padding = "SAME", strides=(1, stride_norm_2dconv))(x)
x = UpSampling2D((max_pool_filter_h,max_pool_filter_w))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding = "SAME", strides=(1, stride_norm_2dconv))(x)

# Compile model
encoder = Model(input_img, encoded)
ccae = Model(input_img, [decoded, clustering_layer])
ccae.compile(optimizer='adam', loss=['mse', silhouette_loss])
# ccae.compile(optimizer='adam', loss=silhouette_loss)
print(ccae.summary())

# Load your dataset here
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
# Fit the model to your dataset
#%%
# train_epochs = 100
train_epochs = 15
# batch_size = 20
# batch_size = 256
batch_size = 128
# batch_size = 64
# batch_size = 16
# batch_size = 32

ccae.fit(x1_training_input_batch, [x1_training_input_batch, np.zeros((len(x1_training_input_batch), n_clusters))],
         epochs=train_epochs, batch_size=batch_size,
         shuffle=True,
         validation_data=(x1_validation_input_batch, [x1_validation_input_batch, np.zeros((len(x1_validation_input_batch), n_clusters))]))

# Extract encoding layer output and perform K-means clustering
get_encoded = K.function([ccae.layers[0].input], [ccae.layers[6].output])
x_encoded = get_encoded([x1_training_input_batch])[0].reshape(len(x1_training_input_batch), -1)
kmeans = KMeans(n_clusters=n_clusters)
y_pred = kmeans.fit_predict(x_encoded)

# Compute the silhouette scores for each sample
silhouette_avg = silhouette_score(x_encoded, y_pred)

# Print the mean silhouette score
print("Silhouette score:", silhouette_avg)
#%%
import matplotlib.pyplot as plt
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
    predicted = ccae.predict([predicted_x2_filtered_input])
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
    
x_pure_pred_enc = ccae.predict(predicted_x2_filtered_input)