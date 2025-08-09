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
from scipy import signal
import vallenae as vae
import sys, os
from scipy import fftpack
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
    
def Normalize_Data_Signal(signal):
    signal = np.array(signal)
    return (signal-signal.mean())/signal.std()

from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn import preprocessing
#%%
# filename = '45-02s-11'
# filename = '45-02s-12'
filename = '45-02s-13'
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio20_4sen.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio20_4sen_2.pridb'
path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.pridb'.format(filename=filename)
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio-50-2sen2.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio-50-2sen3.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\gfe20_4sen.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\GFEaliCFE-50-2sen.pridb'
pridb = vae.io.PriDatabase(path_pri)

# print("Tables in database: ", pridb.tables())
# print("Number of rows in data table (ae_data): ", pridb.rows())
# print("Set of all channels: ", pridb.channel())

hits = pridb.read_hits()
# trais_channel2 = hits_channel2_df['trai']

blockPrint()
filtered_hits = pd.DataFrame(columns=hits.columns)
previous_channel = 0
previous_time = 0
previous_index = 0
previous_row = []
# dt_cfe = 9.7e-6
# # dt_gfe = 18.0e-6
# for index, row in hits.iterrows():
#     current_channel = row['channel']
#     current_time = row['time']
#     print(f'Index: {index}, row: {row.values}')
#     if current_time - previous_time < dt_cfe:
#         filtered_hits = filtered_hits.append(previous_row, ignore_index = False)
#         filtered_hits = filtered_hits.append(row, ignore_index = False)
#     previous_channel = current_channel
#     previous_time = current_time
#     previous_index = index
#     previous_row = row
#     # if len(filtered_hits) >= 300:
#     #     break

enablePrint()
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio20_4sen.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio20_4sen_2.tradb'
path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.tradb'.format(filename=filename)
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio-50-2sen2.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\bio-50-2sen3.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\gfe20_4sen.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite Analysis\VallenDB\GFEaliCFE-50-2sen.tradb'

tradb = vae.io.TraDatabase(path_tra)
fs = 5*10e6
dt = 1/fs
j = 0
#%% balanced_dataset

# hits_channel2_df = filtered_hits.loc[hits['channel'] == 2] # filter out hits, select all hits from channel 2
# hits_channel2_df = hits.loc[hits['channel'] == 2] # filter out hits, select all hits from channel 2
# hits_channel2_df = hits.loc[hits['channel'] == 1] # filter out hits, select all hits from channel 2

# y = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_true_3.pkl")
# labels15 = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_comulative_CFLF_1.pkl")
# replace_dict1 = {'0': 0, '120': 1, '021': 0, '1': 1, \
#                         '01': 0, '102': 1, '012': 0, '10': 1, \
#                         '02': 0, '201': 2, '210': 2, '20': 2, \
#                         '2': 2, '21': 2, '12': 1}
    
# labels_balanced = np.array([float(y.replace(y,str(replace_dict1[y]))) for y in labels15])


crop_ind = 0

hits_channel2_df = hits.copy()
# hits_channel2_df_balanced['label'] = labels15
# hits_channel2_df_pure0 = (hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == '0']).iloc[:42]
# hits_channel2_df_pure1 = (hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == '1']).iloc[:42]
# hits_channel2_df_pure2 = (hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == '2']).iloc[:42]
# hits_channel2_df_pure = pd.concat([hits_channel2_df_pure0,hits_channel2_df_pure1,hits_channel2_df_pure2])

# labels_string_list = ['0','120', '021', '1', '01', '102', '012', '10', '02', '201', '210', '20','2', '21', '12']
# max_samples_per_label = 350
# j = 0
# for label_str in labels_string_list:        
#     df_temp_len = len(hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == label_str])
#     if df_temp_len >= max_samples_per_label:    
#         df_temp = (hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == label_str]).iloc[:max_samples_per_label]
#     else:
#         df_temp = (hits_channel2_df_balanced.loc[hits_channel2_df_balanced['label'] == label_str]).iloc[:df_temp_len]
#     if j == 0:
#         balanced_df = df_temp
#     else:
#         balanced_df = balanced_df.append(df_temp)
#     j = j + 1

# filename = r"C:\Users\Martin-PC\Magistrska Matlab\hits_ch2_balanced_filtered.pkl"
# pd.to_pickle(balanced_df, filename)
#%%
# x2 = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_channel2_loc_filtered_36_544.pkl")
# x2slice = x2[0,:,:434]
# plt.pcolormesh(time_data, freq, x2slice, cmap='viridis', shading='gouraud')
# plt.title('raw, ' +"w_width = " + str(w))
# plt.show()
#%%
# suppress_plot = True
# i = 0
# st_slik = 0
# num_coef = 128
# BatchLength = 512
# cwtms = np.empty((len(hits_channel2_df), num_coef-crop_ind, BatchLength), dtype="float16")
# cwtms_pure = np.empty((len(hits_channel2_df), num_coef-crop_ind, BatchLength), dtype="float16")
# for index, row in hits_channel2_df.iterrows():
#     hit = row['trai']
#     channel = row['channel']
#     try:
#         [signal_data, time_data] = tradb.read_wave(hit)
#     except:
#         print('TRAI does not exists')
#         print(str(hit))
#         print(i)
#         print(index)
#         i += 1
#         continue

#     w = 7.0
#     freq = np.linspace(1, fs/5, num_coef) 
#     widths = w*fs / (2*freq*np.pi)
#     cwtm = signal.cwt(signal_data, signal.morlet2, widths,w=w)
#     cwtm_raw = np.abs(cwtm)

#     max_coefs = np.apply_along_axis(max, axis=0, arr=cwtm_raw)
#     ind_max_coef = list(max_coefs).index(max(max_coefs))
#     if ind_max_coef <= BatchLength/2:
#         start_crop = 0
#         end_crop = start_crop + BatchLength
#         time_data = time_data[start_crop:end_crop] # us
#         signal_data = signal_data[start_crop:end_crop]
#         Samples = len(signal_data)
#     else:
#         start_crop = int(ind_max_coef-BatchLength/2)
#         end_crop = int(start_crop+BatchLength)
#         if start_crop+BatchLength/2 < len(signal_data):
#             time_data = time_data[start_crop:end_crop] # us
#             signal_data = signal_data[start_crop:end_crop]
#             Samples = len(signal_data)
#         else:
#             time_data = time_data[start_crop:] # us
#             signal_data = signal_data[start_crop:]
#             Samples = len(signal_data)
    
#     Samples = len(signal_data)
#     w = 7.0
#     freq = np.linspace(1, fs/5, num_coef) #ch2
#     widths = w*fs / (2*freq*np.pi)
#     cwtm = signal.cwt(signal_data, signal.morlet2, widths,w=w)
#     cwtm_raw = np.abs(cwtm)
#     cwtm_raw = (cwtm_raw - np.mean(cwtm_raw)) / np.std(cwtm_raw)
#     cwtm_raw = cwtm_raw/cwtm_raw.max()
    
#     if not suppress_plot:
#         # if (j/len(dataAE['Time of Event'])*100) > 96:
#         if st_slik <= 100:
#             # plt.pcolormesh(np.ones(BatchesLength), freq, asd, cmap='viridis', shading='gouraud')
#             plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
#             # plt.pcolormesh(time_data, freq, abs_cwtm, cmap='viridis', shading='gouraud')
#             # plt.title(str(hit)+str('; w=')+str(w))
#             plt.show()
#             st_slik = st_slik + 1
#             print(cwtm_raw.shape)
                
#     cwtms[i][:num_coef,:Samples] = cwtm_raw
#     # breakpoint()
#     i = i +1
#%% multithreading
from multiprocessing import Pool

suppress_plot = True
i = 0
st_slik = 0
num_coef = 128
BatchLength = 512
num_rows = len(hits_channel2_df)

cwtms = np.empty((num_rows, num_coef-crop_ind, BatchLength), dtype="float16")
cwtms_pure = np.empty((num_rows, num_coef-crop_ind, BatchLength), dtype="float16")

def process_row(index_row):
    global i, st_slik
    index, row = index_row
    hit = row['trai']
    channel = row['channel']
    try:
        [signal_data, time_data] = tradb.read_wave(hit)
    except:
        print('TRAI does not exists')
        print(str(hit))
        print(i)
        print(index)
        i += 1
        return

    w = 7.0
    freq = np.linspace(1, fs/5, num_coef) 
    widths = w*fs / (2*freq*np.pi)
    cwtm = signal.cwt(signal_data, signal.morlet2, widths,w=w)
    cwtm_raw = np.abs(cwtm)

    max_coefs = np.apply_along_axis(max, axis=0, arr=cwtm_raw)
    ind_max_coef = list(max_coefs).index(max(max_coefs))
    if ind_max_coef <= BatchLength/2:
        start_crop = 0
    else:
        start_crop = int(ind_max_coef-BatchLength/2)
    
    end_crop = start_crop + BatchLength if start_crop + BatchLength/2 < len(signal_data) else len(signal_data)
    time_data = time_data[start_crop:end_crop] # us
    signal_data = signal_data[start_crop:end_crop]
    Samples = len(signal_data)
    
    w = 7.0
    freq = np.linspace(1, fs/5, num_coef) #ch2
    widths = w*fs / (2*freq*np.pi)
    cwtm = signal.cwt(signal_data, signal.morlet2, widths,w=w)
    cwtm_raw = np.abs(cwtm)
    cwtm_raw = (cwtm_raw - np.mean(cwtm_raw)) / np.std(cwtm_raw)
    cwtm_raw = cwtm_raw/cwtm_raw.max()
    
    rgb_images = np.empty((img_height, img_length, 3), dtype=np.float32)

    
    if not suppress_plot:
        if st_slik <= 100:
            plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
            plt.show()
            st_slik += 1
            print(cwtm_raw.shape)
            
    cwtms[i][:num_coef,:Samples] = cwtm_raw
    i += 1

if __name__ == '__main__':
    pool = Pool(processes=2)  # Adjust according to your CPU cores
    pool.map(process_row, hits_channel2_df.iterrows())


# https://towardsdatascience.com/how-to-encode-time-series-into-images-for-financial-forecasting-using-convolutional-neural-networks-5683eb5c53d9    
# print(sys.getsizeof(cwtms)*10e-9)  
#%%
name = '_' + str(num_coef) + '_' + str(BatchLength)
filename = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\{filename}".format(filename=filename) + name +'.pkl'
pd.to_pickle(cwtms, filename)

# filename = r"C:\Users\Martin-PC\Magistrska Matlab\cwtms_cfe_hits_ch2_w11_pure_fl16_std" + name +'.pkl'
# pd.to_pickle(cwtms_pure, filename)
# labels = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_comulative_CFLF_1.pkl")

# za učenje avtoenkoderja: lohk bi vse hits channel 2  classes pa po dvoje spektrov raw in pure, sam že filtered jih je dovolj
# za inicializacijo kmeansa: balanced (glede na spektre -> popravt labels) pure location filtered hits channel 2
# za učenje clustering encoderja: augmented (zrcaljeni pure spektri), ki morjo bit v istih clustrih, pol pa še vsi raw

#%%
# features_merged = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\Sigma_Xi_CFE_GFE_merged_features_embedded_classical_16_32_14_256_6_150_256.pkl")
# # true_single_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_cwt_energy_relative.pkl")
# true_double_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_cwt_energy_relative.pkl")
# # true_single_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_fft_energy_relative.pkl")
# # true_double_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_fft_energy_relative.pkl")
# # true_single_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_single_cfe_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")
# # true_double_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_double_cfe_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")
# true_labels_cfe_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_cfe_hits_x_true_double_cwt_relative.pkl"))
# true_labels_gfe_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_gfe_hits_x_true_double_cwt_relative.pkl"))
# cfe_labels = np.array([0]*len(true_labels_cfe_hits_ch2_amp05_200_400_2500_w11_pure))
# gfe_labels = np.array([1]*len(true_labels_gfe_hits_ch2_amp05_200_400_2500_w11_pure))
# cfe_gfe_labels = np.concatenate((cfe_labels, gfe_labels),axis = 0)

# features_merged_labels = features_merged.copy()
# # features_merged_labels['true_single_cwt_rel'] = true_single_cwt
# features_merged_labels['cfe=0_gfe=1'] = cfe_gfe_labels
# # features_merged_labels['true_single_fft_rel'] = true_single_fft
# # features_merged_labels['true_double_fft_rel'] = true_double_fft
# # features_merged_labels['true_single_cwt_abs'] = true_single_cwt_abs
# # features_merged_labels['true_double_cwt_abs'] = true_double_cwt_abs
# # df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# features_merged_labels.to_excel(r"C:\Users\Martin-PC\Magistrska Matlab\Sigma_Xi_CFE_GFE_merged_features_embedded_classical_16_32_14_256_6_150_256.xlsx", index = False)  
# #%% Preparing balanced dataset
# features_merged_labels_pure0 = features_merged_labels.loc[features_merged_labels['true_double_cwt_rel'] == '0']
# features_merged_labels_pure1 = features_merged_labels.loc[features_merged_labels['true_double_cwt_rel'] == '1']
# features_merged_labels_pure2 = features_merged_labels.loc[features_merged_labels['true_double_cwt_rel'] == '2']
# features_merged_labels_pure = pd.concat([features_merged_labels_pure0, features_merged_labels_pure1, features_merged_labels_pure2], axis=0)


