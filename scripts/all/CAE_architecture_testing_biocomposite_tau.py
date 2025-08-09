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

# y = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_true_3.pkl")
# x = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_filtered_hits_ch1_w11_raw_fl16_std_48_432.pkl")
#%%
# delitelj = 2
# stevilo_slik = int(x.shape[0]/delitelj)
# x1 = x[0:stevilo_slik,2:,:432]
# x1a = x1.reshape(x1.shape + (1,))

#%%
# layers_depths=[60, 120, 128, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[40, 80, 512, 6] #nujno bottleneck = stevilo clustrov
# layers_depths=[4, 8, 56, 4] #nujno bottleneck = stevilo clustrov
layers_depths=[16, 32, 256, 6]
layer_depth_latent_cnn = 14
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
dense_latent_4 = Dense(layers_depths[2], activation = 'relu')(dense_latent_bottleneck)
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

# create model with two inputs
autoencoder = Model([inputs_1], [output_1])
print(autoencoder.summary())