
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:26:07 2021

@author: Martin
"""


# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import vallenae as vae
from itertools import groupby
import collections
import sys, os
import pickle
from collections import Counter
plt.style.use('ggplot')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def percentile_along_axis(array, treshold):
    percentile = np.percentile(array, treshold,interpolation='nearest')
    return percentile

def percent_along_axis(array, treshold_percent):
    percent = max(array)*treshold_percent/100 #z globalnim tresholdom ne mors razlocit med odboji razlicnih magnitud
    return percent

def Normalize_Data_Signal(signal):
    signal = np.array(signal)
    return (signal-signal.mean())/signal.std()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#%%
hits_channel1_df = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.pkl")
labels_unsupervised = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\labels_kmeans_4_deep_FVTA_merged_features_embedded_classical_ch2_16_32_14_256_6_100_256.pkl")
labels = labels_unsupervised
feature = 'Energy in Signal(au)'
#%%
hits_channel1_df_labels = hits_channel1_df.copy()
hits_channel1_df_labels['label_unsupervised'] = labels
hits_channel1_df_labels_0 = hits_channel1_df_labels.copy()
hits_channel1_df_labels_0.loc[hits_channel1_df_labels.label_unsupervised != 0, feature] = 0
hits_channel1_df_labels_1 = hits_channel1_df_labels.copy()
hits_channel1_df_labels_1.loc[hits_channel1_df_labels.label_unsupervised != 1, feature] = 0
hits_channel1_df_labels_2 = hits_channel1_df_labels.copy()
hits_channel1_df_labels_2.loc[hits_channel1_df_labels.label_unsupervised != 2, feature] = 0
hits_channel1_df_labels_3 = hits_channel1_df_labels.copy()
hits_channel1_df_labels_3.loc[hits_channel1_df_labels.label_unsupervised != 3, feature] = 0

Energy0 = np.array(hits_channel1_df_labels_0[feature])
Energy1 = np.array(hits_channel1_df_labels_1[feature])
Energy2 = np.array(hits_channel1_df_labels_2[feature])
Energy3 = np.array(hits_channel1_df_labels_3[feature])

marker_size = 30
class_marker_sizes_amp0 = [marker_size if i !=0 else 0 for i in Energy0]
class_marker_sizes_amp1 = [marker_size if i !=0 else 0 for i in Energy1]
class_marker_sizes_amp2 = [marker_size if i !=0 else 0 for i in Energy2]
class_marker_sizes_amp3 = [marker_size if i !=0 else 0 for i in Energy3]

fig,ax = plt.subplots()
fig.set_size_inches(8, 5)
ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy0), color="purple", label = "Cluster 1", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp0)
ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy1), color="blue", label = "Cluster 2", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp1)
ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy2), color="cyan", label = "Cluster 3", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp2)
ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy3), color="yellow", label = "Cluster 4", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp3)
ax.set_xlabel("Time of Event [s]",fontsize=14)
ax.set_ylabel("Log Energy in Signal [au]",color="black",fontsize=14)

# ax.set_title('Supervised, 3 clusters, location filtered, GFE')

#%%
import collections
cumsum_labels_dict_true_3 = collections.defaultdict(list)
cumsum_labels_dict_true_3[0] = [0]
cumsum_labels_dict_true_3[1] = [0]
cumsum_labels_dict_true_3[2] = [0]
cumsum_labels_dict_true_3[3] = [0]
for label in labels:
    label = int(label)
    cumsum_labels_dict_true_3[label].extend([cumsum_labels_dict_true_3[label][-1]+1])
    if label != 0:
        cumsum_labels_dict_true_3[0].extend([cumsum_labels_dict_true_3[0][-1]])
    if label != 1:
        cumsum_labels_dict_true_3[1].extend([cumsum_labels_dict_true_3[1][-1]])
    if label != 2:
        cumsum_labels_dict_true_3[2].extend([cumsum_labels_dict_true_3[2][-1]])
    if label != 3:
        cumsum_labels_dict_true_3[3].extend([cumsum_labels_dict_true_3[3][-1]])

cumsum_labels_dict_true_3_label_0 = cumsum_labels_dict_true_3[0][1:]
cumsum_labels_dict_true_3_label_1 = cumsum_labels_dict_true_3[1][1:]
cumsum_labels_dict_true_3_label_2 = cumsum_labels_dict_true_3[2][1:]
cumsum_labels_dict_true_3_label_3 = cumsum_labels_dict_true_3[3][1:]
#%%

fig,ax = plt.subplots()
fig.set_size_inches(8, 5)
ax.plot(hits_channel1_df['deflection'], cumsum_labels_dict_true_3_label_0, color="red", label = "Cluster 1")
ax.plot(hits_channel1_df['deflection'], cumsum_labels_dict_true_3_label_1, color="green", label = "Cluster 2")
ax.plot(hits_channel1_df['deflection'], cumsum_labels_dict_true_3_label_2, color="blue", label = "Cluster 3")
ax.plot(hits_channel1_df['deflection'], cumsum_labels_dict_true_3_label_3, color="magenta", label = "Cluster 4")
ax.set_xlabel("Deflection [mm]",fontsize=14)
ax.set_ylabel("Komulativna vsota dogodkov",color="black",fontsize=14)
ax.legend()
ax2=ax.twinx()
ax2.plot(hits_channel1_df['deflection'], hits_channel1_df['Force']/1000, color="orange", marker = '.', linestyle = 'None', markersize = 2.7)
ax2.set_ylabel("Force [kN]",color="orange",fontsize=14)
# ax.set_title('Supervised, 3 clusters')
#%%
labels_unsupervised = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_CFE_kmeans_4_deep_FVTA_ch1_16_32_14_256_6_100_256.pkl")
replacements = {4:3}
replacer = replacements.get
labels_unsupervised = np.array(list(labels_unsupervised) + [0, 1, 2, 3]* 5 + [0],dtype=int)
labels_unsupervised = np.array([replacer(n, n) for n in labels_unsupervised])
labels = labels_unsupervised

#%% CFE
# def percentile50_along_axis(array):
#     percentile = np.percentile(array, 50,interpolation='nearest')
#     index = abs(array-percentile).argmin()
#     return [percentile, index]

# path_pri = r'C:\Users\Martin-PC\Magistrska Matlab\Zadnji dan\normal CFE up3 razbremenitev.pridb'
path_pri = r'C:\Users\Martin-PC\Magistrska Matlab\Zadnji dan\normal CFE up3.pridb'
# path_pri = r'C:\Users\Martin-PC\Magistrska Matlab\Zadnji dan\normal GFE up2.pridb'
pridb = vae.io.PriDatabase(path_pri)

# print("Tables in database: ", pridb.tables())
# print("Number of rows in data table (ae_data): ", pridb.rows())
# print("Set of all channels: ", pridb.channel())

hits = pridb.read_hits()



blockPrint()
filtered_hits = pd.DataFrame(columns=hits.columns)
previous_channel = 0
previous_time = 0
previous_index = 0
previous_row = []
dt_cfe = 9.7e-6
# dt_gfe = 18.0e-6 # ta prav
# dt_gfe = 25.0e-6 # d jih bo mal vec
for index, row in hits.iterrows():
    current_channel = row['channel']
    current_time = row['time']
    print(f'Index: {index}, row: {row.values}')
    if current_time - previous_time < dt_cfe:
        filtered_hits = filtered_hits.append(previous_row, ignore_index = False)
        filtered_hits = filtered_hits.append(row, ignore_index = False)
    previous_channel = current_channel
    previous_time = current_time
    previous_index = index
    previous_row = row
    # if len(filtered_hits) >= 300:
    #     break


hits_channel1_df = filtered_hits.loc[hits['channel'] == 1] # filter out hits, select all hits from channel 2
# hits_channel1_df = hits.loc[hits['channel'] == 2] # filter out hits, select all hits from channel 2

trais_channel2 = hits_channel1_df['trai']
enablePrint()

segmented_signals_dict_data = collections.defaultdict(list)
segmented_signals_dict_data[0] = []
segmented_signals_dict_data[1] = []
segmented_signals_dict_data[2] = []

#
df_markers = pridb.read_markers()
df_parametric = pridb.read_parametric()
filtered_hits_indices = np.array(hits_channel1_df.index)
parametric_indices = np.array(df_parametric.index)
next_closest_indices = [0]*len(hits_channel1_df)
i = 0
for hit in filtered_hits_indices:
    next_closest_indices[i] = find_nearest_val(parametric_indices,hit)
    i = i + 1
    
next_closest_indices_arr = np.array(next_closest_indices)
parametric_df_next_closest = df_parametric.loc[next_closest_indices_arr]
df_parametric_times_indices_filtered = pd.DataFrame(list(parametric_df_next_closest['time']),columns =['time'], index = list(hits_channel1_df.index))
hits_channel1_df['time'] = list(parametric_df_next_closest['time'])
hits_channel1_df['Force'] = list(parametric_df_next_closest['pa0'])
hits_channel1_df['deflection'] = hits_channel1_df['time'] * 0.02  #us * mm/us = mm
hits_channel1_df['reset_index'] = range(0,len(labels))
hits_channel1_df = hits_channel1_df.set_index('reset_index')

#%% Preparing balanced dataset
features_merged = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\Sigma_Xi_CFE_GFE_merged_features_embedded_classical_ch1_16_32_14_256_6_100_256.pkl")
features_merged_copy = features_merged.iloc[:, [4,5,6,7,8,9,10,11,12,13,14,15]]

cfe_gfe_labels = [0]*10194


# x_pure_pred_enc_df = pd.DataFrame(x_pure_pred_enc,columns = columns_vec)
# hits_channel1_df.reset_index(drop=True)
# classical_features_filtered_hits_ch1_cfe.reset_index(inplace=True, drop=True)
# merged_features_embedded_classical = pd.concat([classical_features_filtered_hits_ch1_cfe, x_pure_pred_enc_df], axis=1)
# merged_features_embedded_classical.astype('float32').dtypes


features_trai_merged_hits_cfe_gfe = pd.concat([hits_channel1_df.reset_index(drop=True),features_merged_copy],axis=1)
features_trai_merged_hits_cfe_gfe.astype('float32').dtypes
features_trai_merged_hits_cfe_gfe['lables_true'] = list(labels)
features_trai_merged_hits_cfe_gfe['cfe_gfe'] = cfe_gfe_labels
bac_0123 = [None]*4
ac_0123 = [None]*4
FVTA_mean_CFE = [None]*4
FVTA_mean_GFE = [None]*4

mean_features_CFE = np.empty((4,16), dtype=float, order='C')
mean_features_GFE = np.empty((4,16), dtype=float, order='C')

for n in [0,1,2,3]:
    features_merged_damage_mechanism_each = features_merged[(features_trai_merged_hits_cfe_gfe['lables_true'] == n).values]
    # breakpoint()
    # features_merged_damage_mechanism_each = features_merged_damage_mechanism_each.iloc[:, [0,1,2,3,4,5,6,7,8,9]] #za splošne značilke samo
    # cfe_gfe_labels_damage_mechanism_each = cfe_gfe_labels[(features_trai_merged_hits_cfe_gfe['lables_true'] == n)]
    # features_merged_labels_pure_selected_deep_FVTA_each = features_merged_labels_pure_selected_deep_FVTA[(features_trai_merged_hits_cfe_gfe['lables_true'] == n)]
    features_merged_each = features_merged[(features_trai_merged_hits_cfe_gfe['lables_true'] == n)]
    # features_merged_damage_mechanism_each_CFE = features_merged_labels_pure_selected_deep_FVTA_each[(cfe_gfe_labels_damage_mechanism_each == 0)] #C
    # features_merged_damage_mechanism_each_GFE = features_merged_labels_pure_selected_deep_FVTA_each[(cfe_gfe_labels_damage_mechanism_each == 1)] #C
    #% plot
    # features_merged_each_each_CFE = features_merged_each[(cfe_gfe_labels_damage_mechanism_each == 0)] #C
    # features_merged_each_each_GFE = features_merged_each[(cfe_gfe_labels_damage_mechanism_each == 1)] #C
    # FVTA_mean_CFE[n]=np.mean(features_merged_damage_mechanism_each_CFE[:,0])/1000
    # FVTA_mean_GFE[n]=np.mean(features_merged_damage_mechanism_each_GFE[:,0])/1000
    
    
    for m in range(0,16):
        # breakpoint()
        mean_features_CFE[n,m]=np.mean(features_merged_each[features_merged_each.columns[m]])
        # mean_features_GFE[n,m]=np.mean(features_merged_each_each_GFE[features_merged_each_each_GFE.columns[m]])
        
mean_features_CFE_df = pd.DataFrame(mean_features_CFE,columns = features_merged_each.columns )
# mean_features_GFE_df = pd.DataFrame(mean_features_GFE,columns = features_merged_each_each_GFE.columns )
# mean_features_CFE_df.to_excel(r"C:\Users\Martin-PC\Magistrska Matlab\mean_features_CFE_df_ch1.xlsx", index = False) 
# mean_features_GFE_df.to_excel(r"C:\Users\Martin-PC\Magistrska Matlab\mean_features_GFE_df.xlsx", index = False) 
 
    