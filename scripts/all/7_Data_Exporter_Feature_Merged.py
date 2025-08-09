# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 23:25:06 2021

@author: Martin-PC
"""

#%%
# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
# import vallenae as vae
import sys, os
# from scipy import fftpack
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
plt.style.use('ggplot')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
# def Normalize_Data_Signal(signal):
#     signal = np.array(signal)
#     return (signal-signal.mean())/signal.std()

# from pyts.image import GramianAngularField
# from mpl_toolkits.axes_grid1 import ImageGrid
# from sklearn import preprocessing
# from mpl_toolkits.mplot3d import Axes3D
# import plotly.express as px
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# from sklearn.preprocessing import LabelEncoder
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils

# for modeling
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MinMaxScaler
#%%
num_folds = 3
val_acc_list = [None]*num_folds
train_acc_list = [None]*num_folds
# data_cfe = [r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_38_76_34_564_16_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_74_30_432_14_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_34_68_34_648_16_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_30_60_30_512_12_15_32.pkl",
            
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_26_512_10_10_64.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_16_32_14_256_6_100_128.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_20_512_8_10_20.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_12_24_10_256_6_100_256.pkl",
#             ]

# data_bio = [r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_38_76_34_564_16_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_74_30_432_14_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_34_68_34_648_16_15_32.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_30_60_30_512_12_15_32.pkl",
            
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_26_512_10_10_64.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_16_32_14_256_6_100_128.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_20_512_8_10_20.pkl",
#             r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_12_24_10_256_6_100_256.pkl",
#             ]

data_cfe = [r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_16_32_14_256_6_100_256.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_6_12_12_128_4_75_356.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_20_512_8_100_256.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_16_256_6_75_256.pkl",
            ]

data_bio = [r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_16_32_14_256_6_100_256.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_6_12_12_128_4_75_356.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_20_512_8_100_256.pkl",
            r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_vani_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_16_256_6_75_256.pkl",
            ]
i = 0
for cfe,bio in zip(data_cfe,data_bio):
    features_merged_bio_cold = pd.read_pickle(bio)
    features_merged_cfe_cold = pd.read_pickle(cfe)
    emb_columns = [i for i in features_merged_bio_cold if 'Emb' in i]
    emb_columns_range = list(range(23,len(emb_columns)+23))
    # all_features_to_keep = [0,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,-1]
    all_features_to_keep = [0,8,9,10,11,12,13,16,17,18,19,20,21,22] + emb_columns_range + [-1]
    # all_features_to_keep = [0,8,9,10,11,12,13,16,17,18,19,20,21,22,-1] # brez globokih
    # emb_columns_range = list(range(-len(emb_columns)-2,0))
    # globoke_features_to_keep = list(range(-len(emb_columns)-1,0)) # samo globoke
    # globoke_features_to_keep = [0,-7,-6,-5,-4,-3,-2,-1] # samo globoke, amp
    # globoke_features_to_keep = [0,7,-7,-6,-5,-4,-3,-2,-1] # samo globoke, amp pa FMXA
    
    # features_merged_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_bio_tok_40_80_16_512_6_25_64.pkl")
    # features_merged_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_gfe_tok_40_80_16_512_6_25_64.pkl")
    # # all_features_to_keep = [8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,-1]
    # all_features_to_keep = [8,9,10,11,12,13,16,17,18,19,20,21,22,-1] # brez globokih
    # # globoke_features_to_keep = [-7,-6,-5,-4,-3,-2,-1] # samo globoke
    
    tok_labels = np.array([0]*len(features_merged_bio_cold))
    t80_labels = np.array([1]*len(features_merged_cfe_cold))
    labels_tok_t80 = np.concatenate((tok_labels, t80_labels),axis = 0)
    
    #%
    Count = 5
    Amplitude_crop_value = 59 #cold cfe bio
    # Amplitude_crop_value = 49.542 #tok gfe bio
    
    # all_features_to_keep = [8,9,10,11,12,13,16,17,18,23,24,25,26,27] # brez parcialnih spektralnih energij
    features_merged = pd.concat([features_merged_bio_cold, features_merged_cfe_cold], axis=0,ignore_index=True)
    features_merged_export = features_merged.copy()
    features_merged_export['labels_bio-0_cfe-1'] = labels_tok_t80
    features_merged_export = features_merged_export.iloc[:, all_features_to_keep]
    features_merged_export = clean_dataset(features_merged_export).reset_index(drop=True)
    features_merged_export = features_merged_export.loc[features_merged_export['Count'] >= Count]
    features_merged_export = features_merged_export.loc[features_merged_export['Peak Amplitude(dB)'] >= Amplitude_crop_value]
    # features_merged_export = features_merged_export.iloc[:, globoke_features_to_keep]
    features_merged_export=features_merged_export.loc[:, (features_merged_export != 0).any(axis=0)]
    path_export = cfe.replace('pkl','xlsx')
    path_export = path_export.replace('cfe_cold_','')
    # breakpoint()
    features_merged_export.to_excel(path_export, index = False) 
