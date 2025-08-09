# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:00:08 2023

@author: Martin-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:33:47 2021

@author: Martin-PC
"""

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
# import vallenae as vae
import sys, os
# from scipy import fftpack
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
# plt.style.use('ggplot')

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
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.pyplot as plt
import os, sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
# from mpl_toolkits.mplot3d import Axes3D
# from collections import Counter
# from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import csv
from sklearn.feature_selection import SequentialFeatureSelector
from statsmodels.stats.outliers_influence import variance_inflation_factor

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline = '') as output_file:
        output_writer = csv.writer(output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(list_of_elem)
        
layers_depths=[16, 32, 256, 6] #nujno bottleneck = stevilo clustrov
layer_depth_latent_cnn = 14
train_epochs = 100
batch_size = 256
ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + \
    str(train_epochs) + "_" + str(batch_size)
Output = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\Clustering_analysis_ch1"
Output_filename = Output + ver_str + ".csv"

filename = '45-02s-11'
# filename = '45-02s-13'
# identifier = '16_8_16_512_3_30_256'
identifier = 'encoder_clustering_chatGPT_32x64'
#%
# true_labels_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_hits_x_true_double_cwt_relative.pkl"))
# true_labels_hits_ch2_amp05_200_400_2500_w11_pure = np.array(pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\labels_xi_hits_x_true_double_cwt_relative.pkl"))
# labels_true_3_hits_cwt_pure = [int(str(x)[0]) for x in true_labels_hits_ch2_amp05_200_400_2500_w11_pure]
# labels_true_3_hits_cwt_pure = [int(str(x)[0]) for x in true_labels_hits_ch2_amp05_200_400_2500_w11_pure]
# labels_true = np.concatenate((labels_true_3_hits_cwt_pure, labels_true_3_hits_cwt_pure),axis = 0)
#%
# features_merged = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_lambda_embedded_classical_w7_{filename}_32x512_16_32_14_256_6_10_20.pkl".format(filename=filename))
features_merged = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_lambda_seperate_dec_emb_cla_w7_{filename}_32x512_16_32_14_256_6_25_64.pkl".format(filename=filename))
# features_merged = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_lambda_16_8_16_512_3_30_256_{filename}_32x512_16_8_16_512_3_30_256.pkl".format(filename=filename))
# features_merged = features_merged.iloc[0:len(labels_true_3_hits_cwt_pure)]
# features_merged = features_merged.iloc[len(labels_true_3_hits_cwt_pure):]
#%
features_merged = features_merged.drop(features_merged[features_merged['FVTA'] == 0].index).reset_index(drop=True)
# true_single_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_cwt_energy_relative.pkl")
# true_double_cwt = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_cwt_energy_relative.pkl"
# true_single_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_single_fft_energy_relative.pkl")
# true_double_fft = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\sigma_true_double_fft_energy_relative.pkl")
# true_single_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_single_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")
# true_double_cwt_abs = pd.read_pickle(r"C:\Users\Martin-PC\Magistrska Matlab\true_labels_double_filtered_hits_ch2_amp05_200_400_2500_w11_raw.pkl")

# features_merged_labels = features_merged.copy()
# features_merged_labels['true_single_cwt_rel'] = true_single_cwt
# features_merged_labels['cfe=0=1'] = cfe_labels
# features_merged_labels['true_single_fft_rel'] = true_single_fft
# features_merged_labels['true_double_fft_rel'] = true_double_fft
# features_merged_labels['true_single_cwt_abs'] = true_single_cwt_abs
# features_merged_labels['true_double_cwt_abs'] = true_double_cwt_abs
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# features_merged_labels.to_excel(r"C:\Users\Martin-PC\Magistrska Matlab\Sigma_Nu_merged_features_embedded_classical_3_6_11_96_4_75_256.xlsx", index = False) 
#% Preparing balanced dataset
# all_features_to_keep = [8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28]
# all_features_to_keep = [17] #amp, energy risetime, FMXA
# deep_features_to_keep = [3,5,8,18,26] # FMXA and deep
deep_features_to_keep = [3,5,17,23,24,25] # FMXA and deep
# deep_features_to_keep = [3,5,18] # FMXA and deep
features_to_keep = deep_features_to_keep
# features_to_keep = [3,5,11,13,14,15,16]
# features_to_keep = [3,5,7,11,13,14,15,16]
# features_merged_labels_pure_selected_classical = features_merged_labels.iloc[:, [0,1,2,3,4,5,7,8,9]]
features_merged_deep_FVTA = features_merged.iloc[:, features_to_keep]
# features_merged_deep_FVTA = features_merged.iloc[:, [6,10,11,12,13,14,15]]
# features_merged_labels_pure_selected_classical = features_merged_labels.iloc[:, [0,1,2,3,4,5,7,8,9]]
features_merged_labels_pure_selected_deep_FVTA_df = features_merged.iloc[:, features_to_keep]


# initialize MinMaxScaler
scaler = RobustScaler()

# rescale the DataFrame
# features_merged_labels_pure_selected_deep_FVTA_df = pd.DataFrame(scaler.fit_transform(features_merged_labels_pure_selected_deep_FVTA_df), columns=features_merged_labels_pure_selected_deep_FVTA_df.columns)

# features_merged_labels_pure_selected_classical = np.array(features_merged_labels_pure_selected_classical)
features_merged_labels_pure_selected_deep_FVTA = np.array(pd.DataFrame(scaler.fit_transform(features_merged_labels_pure_selected_deep_FVTA_df), columns=features_merged_labels_pure_selected_deep_FVTA_df.columns))
# features_merged_labels_pure_selected_classical = np.array(features_merged_labels_pure_selected_classical)
features_merged_deep_FVTA = np.array(features_merged_labels_pure_selected_deep_FVTA)
# features_merged_deep_FVTA = features_merged_deep_FVTA[:10173]
X = features_merged_deep_FVTA
x_pure_pred_enc_unscaled_features = X

# X = StandardScaler().fit_transform(X)
# X = MinMaxScaler().fit_transform(X)

# pca = PCA(n_components = 4)
# X = pca.fit_transform(X)
# explained_variance = pca.explained_variance_ratio_
# print(sum(explained_variance))

vif_data = pd.DataFrame()
vif_data["feature"] = features_merged_labels_pure_selected_deep_FVTA_df.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(features_merged_labels_pure_selected_deep_FVTA_df.values, i)
                          for i in range(len(features_merged_labels_pure_selected_deep_FVTA_df.columns))]
print(vif_data)
#%% DBSCAN FILTERING
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# db = DBSCAN(eps=6.9, min_samples=105).fit(X) # brez PCA

# db = DBSCAN(eps=0.9, min_samples=100).fit(X) # PCA

# db = DBSCAN(eps=6.9, min_samples=100).fit(X) # Robust Scaler, klasične značilke
db = DBSCAN(eps=3.9, min_samples=100).fit(X) # Robust Scaler, standardScaler, PCA

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_dbscan = db.labels_
#%
# #%
# dim_x = 1
# dim_y = 0 #petka!
# dim_z = 2

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
#%%
x_pure_pred_enc_df = pd.DataFrame(X)
x_pure_pred_enc_df['labels_DBscan'] =  labels_dbscan
# x_pure_pred_enc_df['labels_True_ctw'] =  labels_true


x_pure_pred_enc_df_dbscan_filtered = x_pure_pred_enc_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index)

# hits_channel2_df.reset_index(inplace=True, drop=True)
X = features_merged_labels_pure_selected_deep_FVTA_df.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
features_merged_dropped_dbscan = features_merged.drop(x_pure_pred_enc_df[~(x_pure_pred_enc_df['labels_DBscan'] == 0)].index).reset_index(drop=True)
x_pure_pred_enc_unscaled_features = np.array(X)
# pd.to_pickle(hits_channel2_df_dbscan, r"C:\Users\Martin-PC\Magistrska Matlab\CFE_hits_channel2_df_labels_ch1_dbscan.pkl")       


# labels_true_input_cwt05_pure_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_True_ctw'])
labels_only_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered['labels_DBscan'])
x_pure_pred_enc_df_dbscan_filtered.drop(x_pure_pred_enc_df_dbscan_filtered.columns[np.arange(x_pure_pred_enc_df_dbscan_filtered.shape[-1]-1,x_pure_pred_enc_df_dbscan_filtered.shape[-1])], axis = 1, inplace = True)
x_pure_pred_enc_dbscan = np.array(x_pure_pred_enc_df_dbscan_filtered)     
X=x_pure_pred_enc_dbscan
print(len(x_pure_pred_enc_df)-len(x_pure_pred_enc_dbscan))
#%% silhuette Kmeans
import matplotlib.colors as mcolors
from sklearn.cluster import SpectralClustering

range_n_clusters = [4]
silhouette_avg_list_kmeans = []
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=2)
    cluster_labels = clusterer.fit_predict(X)
    labels_true = cluster_labels
    
    
    # sc = SpectralClustering(n_clusters=n_clusters).fit(X)
    # print(sc)
     
    # SpectralClustering(affinity='nearest_neighbors', assign_labels='cluster_qr',
    #                    coef0=1, degree=3,
    #                eigen_solver=None, eigen_tol=0.0, gamma=8.5,
    #                kernel_params=None, n_clusters=n_clusters, n_components=None,
    #                n_init=10, n_jobs=None, n_neighbors=20, random_state=None) 
    
    # cluster_labels = sc.labels_
    # labels_true = cluster_labels
    # sfs = SequentialFeatureSelector(clusterer, n_features_to_select=8)
    
    # x_pure_pred_enc = X
    # labels_true = cfe_labels
    # labels_true = labels_true.astype(int)
    # x_pure_pred_enc = np.array(features_merged_labels_pure_selected_deep_FVTA)
    #%
    plt.style.use('dark_background')
    #deep features
    # dim_x = 1 #energija deep
    # dim_y = 3 # globoka 1 deep
    # dim_z = 0 # FCOG deep
    
    #classical features
    dim_x = 0 #energija CFE
    dim_y = 2# globoka 1 CFE
    dim_z = 3 # FCOG CFE
    
    fig1 = plt.figure(2, figsize=(11.5, 11.5))
    ax = Axes3D(fig1, rect=[0, 0, .95, 1], elev=10, azim=26, auto_add_to_figure=False)
    fig1.add_axes(ax)
    # for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3),('Category4', 4),('Category5', 5),('Category6', 6),('Category7', 7),('Category8', 8),('Category9', 9),('Category10', 10),('Category11', 11),('Category12', 12),('Category13', 13)]:
    # for name, label in [('Category0', 0), ('Category1', 1), ('Category2', 2),('Category3', 3),('Category4', 4)]:
    #     ax.text3D(x_pure_pred_enc[labels == label, 0].mean(),
    #               x_pure_pred_enc[labels == label, 1].mean(),
    #               x_pure_pred_enc[labels == label, 2].mean(), 
    #               name,
    #               bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # replacements = {0:1,1:0}
    # replacer = replacements.get
    # unsupervised_labels = np.array([replacer(n, n) for n in labels_true]) 
    
    unsupervised_labels = np.choose(labels_true, [0, 1, 2,3,4,5,6,7]).astype(float) #pac channel 1
    # unsupervised_labels = np.choose(labels_true, [0, 4,1,2]).astype(float) #pac channel 2
    # unsupervised_labels = np.choose(labels, [2, 0, 2,0,3,2,2,0,1,0,0,0,0,0]).astype(float) #sorte+d gfe
    # unsupervised_labels = np.choose(labels, [1, 0, 0,0,2,0,0,0,1,0,0,0,0,0]).astype(float) #sorte+d gfe
    # unsupervised_labels = np.choose(labels, [0, 1, 2,3,4,5,6,7,8,9,10,0,0,0]).astype(float) #sorte+d gfe
    # unsupervised_labels = np.choose(labels, [1, 0, 2,1,0,2,1,2,0,0,4,3,2,1]).astype(float) #sorte+d cfe
    
    mapping = {# old: new
      0: 19,
      1: 19,
      2: 19,
      3: 19,
      4: 19,
      5: 19}
    
    # viridis_cols = ['#440154','#365c8d','#1fa187','#dde318','tomato','magenta']
    viridis_cols = ['#440154','#365c8d','#1fa187','#dde318']
    
    mapping_colors = {# old: new
      0: viridis_cols[0],
      1: viridis_cols[1],
      2: viridis_cols[2],
      3: viridis_cols[3],
      # 4: viridis_cols[4],
      # 5: viridis_cols[5]
      }
    
    # ['purple','blue','green','red']
    
    class_marker_sizes = [mapping.get(number, number) for number in unsupervised_labels]
    # class_marker_colors = [mapping_colors.get(number, number) for number in unsupervised_labels]
    # class_marker_colors = np.asarray(class_marker_colors)
    # scatter = ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 5], x_pure_pred_enc[:, 6], c=unsupervised_labels, cmap='Set1', label = unsupervised_labels, edgecolors='none', s=class_marker_sizes)
    # scatter = ax.scatter(x_pure_pred_enc_dbscan[:, 0], x_pure_pred_enc_dbscan[:, 2], x_pure_pred_enc_dbscan[:, 4], c=unsupervised_labels, cmap='viridis', label = unsupervised_labels, edgecolors='none', s=class_marker_sizes) #GFE
    # plt.xlim(-2,4)  
    # plt.ylim(-4,2) 
    # ax.set_zlim(-2,4) 
    # scatter =  ax.scatter(x_pure_pred_enc[:, 0], x_pure_pred_enc[:, 1], x_pure_pred_enc[:,4], c=unsupervised_labels, cmap='viridis', label = unsupervised_labels, edgecolors='none', s=class_marker_sizes)
    scatter =  ax.scatter(x_pure_pred_enc_unscaled_features[:, dim_x], x_pure_pred_enc_unscaled_features[:, dim_y], x_pure_pred_enc_unscaled_features[:,dim_z], c=unsupervised_labels,cmap = mcolors.ListedColormap(viridis_cols), label = unsupervised_labels, edgecolors='none', s=class_marker_sizes)
    # scatter =  ax.scatter(x_pure_pred_enc[:, 11], x_pure_pred_enc[:, 13], x_pure_pred_enc[:,6], c=unsupervised_labels, cmap='Set1', label = unsupervised_labels, edgecolors='none', s=class_marker_sizes)
    # plt.xticks(np.arange(min(x_pure_pred_enc[:, dim_x]), max(x_pure_pred_enc[:, dim_x])+1, 7.0))
    # plt.xlim(-2,4)  
    # plt.ylim(-4,3) 
    # ax.set_zlim(-2,3) 
    ax.tick_params(axis = 'both', labelsize=20)
    # plt.xlim(0,xlim)  # cluster belongings, K-means point of view 0-1-2
    # plt.ylim(0,34) 
    # ax.set_zlim(100000,zlim)
    
    # plt.legend(handles=scatter.legend_elements()[0], 
            # labels=['CFE','GFE'], loc='best', bbox_to_anchor=(0.0, 0., 0.3, 0.1),fontsize = 16,
            # title="Legenda")
    
    plt.legend(handles=scatter.legend_elements()[0], 
            # labels=['Matrix cracking','Fiber breakage','Fiber-matrix debonding','Fiber pullout'], loc='right',
            labels=['Cluster '+str(x) for x in range(1,n_clusters+1)], loc='right',
            bbox_to_anchor=(0.37, 0.36, 0.54, 0.5),
            fontsize = 18,
            title="Legend")
    
    
    plt.rcParams['legend.title_fontsize'] = 21
    
    ax.set_xlabel(features_merged_labels_pure_selected_deep_FVTA_df.columns[dim_x], fontsize=20)
    ax.set_ylabel(features_merged_labels_pure_selected_deep_FVTA_df.columns[dim_y], fontsize=20)
    # ax.set_xlabel('Deep 1', fontsize=20)
    # ax.set_ylabel('Deep 2', fontsize=20)
    ax.set_zlabel(features_merged_labels_pure_selected_deep_FVTA_df.columns[dim_z], fontsize=20, rotation = 0)
    #%
    plt.style.use('default')
    # fig1.show()
    # fig1.close()
    # ax.xaxis.set_label_coords(1.55, -0.55)
    # plt.title('K means',fontsize=24)
    # fig.suptitle('test title', fontsize=20)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("Kmeans","For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    silhouette_avg_list_kmeans.append(silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    append_list_as_row(Output_filename,  ["Kmeans","For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg])
    y_lower = 10
    
    # silhou_true_cwt = metrics.silhouette_score(x_pure_pred_enc, labels_true, metric='euclidean') # the higher the better
    # calinski_true_cwt = metrics.calinski_harabasz_score(x_pure_pred_enc, labels_true) # the higher the better
    # davies_true_cwt = metrics.davies_bouldin_score(x_pure_pred_enc, labels_true) # the lower the better
    # performance_metrics = [silhou_true_cwt, calinski_true_cwt, davies_true_cwt]
    # print(performance_metrics)
    fig, (ax1) = plt.subplots()
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
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
#%%
hits_channel1_df = features_merged_dropped_dbscan
# hits_channel1_df = features_merged
labels_unsupervised = cluster_labels
labels = labels_unsupervised
feature = 'Energy in Signal(au)'
#%%
hits_channel1_df = hits_channel1_df.rename(columns={'time':'Time of Event','energy':'Energy in Signal(au)'})
hits_channel1_df_labels = hits_channel1_df.copy()
hits_channel1_df_labels['label_unsupervised'] = labels
hits_channel1_df_labels['label_unsupervised'] = hits_channel1_df_labels['label_unsupervised'].apply(lambda x: 'Cluster '+str(x+1))
# hits_channel1_df_labels['label_unsupervised'] = hits_channel1_df_labels['label_unsupervised'].astype(str)


hits_channel1_df_labels_list = [None]*n_clusters
Energy_list = [None]*n_clusters
class_marker_sizes_list = [None]*n_clusters
marker_size = 30

for ind,hits_channel1_df_labels_clus in enumerate(hits_channel1_df_labels_list):
    hits_channel1_df_labels_list[ind] = hits_channel1_df_labels.copy()
    hits_channel1_df_labels_list[ind].loc[hits_channel1_df_labels.label_unsupervised != ind, feature] = 0
    Energy_list[ind] = np.array(hits_channel1_df_labels_list[ind][feature])
    class_marker_sizes_list[ind] = [marker_size if i !=0 else 0 for i in Energy_list[ind]]

# hits_channel1_df_labels_0 = hits_channel1_df_labels.copy()
# hits_channel1_df_labels_0.loc[hits_channel1_df_labels.label_unsupervised != 0, feature] = 0
# hits_channel1_df_labels_1 = hits_channel1_df_labels.copy()
# hits_channel1_df_labels_1.loc[hits_channel1_df_labels.label_unsupervised != 1, feature] = 0
# hits_channel1_df_labels_2 = hits_channel1_df_labels.copy()
# hits_channel1_df_labels_2.loc[hits_channel1_df_labels.label_unsupervised != 2, feature] = 0
# hits_channel1_df_labels_3 = hits_channel1_df_labels.copy()
# hits_channel1_df_labels_3.loc[hits_channel1_df_labels.label_unsupervised != 3, feature] = 0
# hits_channel1_df_labels_4 = hits_channel1_df_labels.copy()
# hits_channel1_df_labels_4.loc[hits_channel1_df_labels.label_unsupervised != 4, feature] = 0

# Energy0 = np.array(hits_channel1_df_labels_0[feature])
# Energy1 = np.array(hits_channel1_df_labels_1[feature])
# Energy2 = np.array(hits_channel1_df_labels_2[feature])
# Energy3 = np.array(hits_channel1_df_labels_3[feature])

# marker_size = 30
# class_marker_sizes_amp0 = [marker_size if i !=0 else 0 for i in Energy0]
# class_marker_sizes_amp1 = [marker_size if i !=0 else 0 for i in Energy1]
# class_marker_sizes_amp2 = [marker_size if i !=0 else 0 for i in Energy2]
# class_marker_sizes_amp3 = [marker_size if i !=0 else 0 for i in Energy3]

fig,ax = plt.subplots()
fig.set_size_inches(8, 5)
for ind,hits_channel1_df_labels_clus in enumerate(hits_channel1_df_labels_list):
    ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy_list[ind]), color=viridis_cols[ind], label = "Cluster "+str(ind+1), marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_list[ind])
# ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy1), color=viridis_cols[1], label = "Cluster 2", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp1)
# ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy2), color=viridis_cols[2], label = "Cluster 3", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp2)
# ax.scatter(hits_channel1_df['Time of Event'], np.log(Energy3), color=viridis_cols[3], label = "Cluster 4", marker = '.', alpha=0.5, linestyle = 'None', s=class_marker_sizes_amp3)
ax.set_xlabel("Time of Event [s]",fontsize=14)
ax.set_ylabel("Log Energy in Signal [au]",color="black",fontsize=14)


hits_channel1_df_labels['Log Energy in Signal'] = np.log(hits_channel1_df_labels['Energy in Signal(au)'])
hits_channel1_df_labels['Log Energy in Signal'] = np.abs(hits_channel1_df_labels['Log Energy in Signal'])

["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
for z_axis in ['Log Energy in Signal','amplitude','rise_time']:
    fig_3d = px.scatter_3d(hits_channel1_df_labels, x='Time of Event', y='label_unsupervised', z=z_axis,
                           template='plotly_dark',
                           size='Log Energy in Signal', size_max=14, opacity=1,color='label_unsupervised',color_discrete_map={'Cluster '+str(num_cl+1):viridis_cols[num_cl] for num_cl in range(0,n_clusters)})
    # fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor='rgba(0,0,0,0)')
    fig_html_3d = fig_3d.to_html()
    with open(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\3dscatter_{identifier}_{filename}_{z_axis}.html".format(filename=filename,z_axis=z_axis,identifier=identifier), "w") as file:
        file.write(fig_html_3d)
#%%
import collections
cumsum_labels_dict_true_3 = collections.defaultdict(list)
# hits_channel1_df_labels_list = [None]*n_clusters
# cumsum_labels_dict_true_3[0] = [0]
# cumsum_labels_dict_true_3[1] = [0]
# cumsum_labels_dict_true_3[2] = [0]
# cumsum_labels_dict_true_3[3] = [0]
# n_clusters = 4
for i in range(0,n_clusters):
    cumsum_labels_dict_true_3[i] = [0]

# breakpoint()
for label in labels:
    label = int(label)
    cumsum_labels_dict_true_3[label].extend([cumsum_labels_dict_true_3[label][-1]+1])
    for i in range(0,n_clusters):
        if label != i:
            cumsum_labels_dict_true_3[i].extend([cumsum_labels_dict_true_3[i][-1]])
        # if label != 1:
        #     cumsum_labels_dict_true_3[1].extend([cumsum_labels_dict_true_3[1][-1]])
        # if label != 2:
        #     cumsum_labels_dict_true_3[2].extend([cumsum_labels_dict_true_3[2][-1]])
        # if label != 3:
        #     cumsum_labels_dict_true_3[3].extend([cumsum_labels_dict_true_3[3][-1]])
cumsum_labels_dict_true_3_label_list = [None]*(n_clusters)

for i in range(0,n_clusters):
    cumsum_labels_dict_true_3_label_list[i] = cumsum_labels_dict_true_3[i][1:]
# cumsum_labels_dict_true_3_label_0 = cumsum_labels_dict_true_3[0][1:]
# cumsum_labels_dict_true_3_label_1 = cumsum_labels_dict_true_3[1][1:]
# cumsum_labels_dict_true_3_label_2 = cumsum_labels_dict_true_3[2][1:]
# cumsum_labels_dict_true_3_label_3 = cumsum_labels_dict_true_3[3][1:]
#%%
# feature_cumsum = 'deflection'
fig,ax = plt.subplots()
fig.set_size_inches(8, 5)
for i in range(0,n_clusters):
    ax.plot(hits_channel1_df['Time of Event'], cumsum_labels_dict_true_3_label_list[i],linewidth=2, color=viridis_cols[i], label = "Cluster "+str(i+1))
# ax.plot(hits_channel1_df['Time of Event'], cumsum_labels_dict_true_3_label_1, color=viridis_cols[1], label = "Cluster 2")
# ax.plot(hits_channel1_df['Time of Event'], cumsum_labels_dict_true_3_label_2, color=viridis_cols[2], label = "Cluster 3")
# ax.plot(hits_channel1_df['Time of Event'], cumsum_labels_dict_true_3_label_3, color=viridis_cols[3], label = "Cluster 4")
ax.set_xlabel(feature,fontsize=14)
ax.set_ylabel("Komulativna vsota dogodkov",color="black",fontsize=14)
ax.legend()
ax2=ax.twinx()
ax2.plot(hits_channel1_df['Time of Event'], hits_channel1_df[feature], color="gray", marker = '.', linestyle = 'None', markersize = 1.7)
ax2.set_ylabel(feature,color="gray",fontsize=14)

#%%
# breakpoint()
# hits_channel1_df = features_merged
# hits_channel1_df = features_merged_dropped_dbscan
# labels_unsupervised = cluster_labels
# labels = labels_unsupervised


# all_features_to_keep = all_features_to_keep + deep_features_to_keep #deep
all_features_to_keep = deep_features_to_keep #deep

features_merged_copy = hits_channel1_df
features_merged_copy['label_unsupervised'] = labels_unsupervised


# FVTA_mean = [None]*4
# FVTA_mean = [None]*4

mean_features = np.empty((n_clusters,features_merged_copy.shape[1]), dtype=float, order='C')

for n in range(n_clusters):
    features_merged_damage_mechanism_each = features_merged_copy[(features_merged_copy['label_unsupervised'] == n)]
    # breakpoint()
    # features_merged_damage_mechanism_each = features_merged_damage_mechanism_each.iloc[:, [0,1,2,3,4,5,6,7,8,9]] #za splošne značilke samo
    # cfe_labels_damage_mechanism_each = cfe_labels[(features_merged_copy['label_unsupervised'] == n)]
    # features_merged_labels_pure_selected_deep_FVTA_each = features_merged_labels_pure_selected_deep_FVTA[(features_merged_copy['label_unsupervised'] == n)]
    features_merged_each = features_merged_copy[(features_merged_copy['label_unsupervised'] == n)]
    # features_merged_damage_mechanism_each = features_merged_labels_pure_selected_deep_FVTA_each[(cfe_labels_damage_mechanism_each == 0)] #C
    # features_merged_damage_mechanism_each = features_merged_labels_pure_selected_deep_FVTA_each[(cfe_labels_damage_mechanism_each == 1)] #C
    #% plot
    # features_merged_each_each = features_merged_each[(cfe_labels_damage_mechanism_each == 0)] #C
    # features_merged_each_each = features_merged_each[(cfe_labels_damage_mechanism_each == 1)] #C
    # FVTA_mean[n]=np.mean(features_merged_damage_mechanism_each[:,0])/1000
    # FVTA_mean[n]=np.mean(features_merged_damage_mechanism_each[:,0])/1000
    
    
    for m in range(0,features_merged_each.shape[1]-1):
        # breakpoint()
        mean_features[n,m]=np.mean(features_merged_each[features_merged_each.columns[m]])
        mean_features[n,m]=np.mean(features_merged_each[features_merged_each.columns[m]])
    
# print('FVTA_mean = '+ str(FVTA_mean))
# print('FVTA_mean = '+ str(FVTA_mean))
features_merged_initial = hits_channel1_df
# breakpoint()
mean_features_df = pd.DataFrame(data=mean_features,columns=features_merged_initial.columns)
mean_features_df['Cluster'] = ['Cluster '+str(x+1) for x in range(0,n_clusters)]
save_dir_root = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm'
save_dir = save_dir_root + '\mean_features_df_{identifier}_{filename}'.format(filename=filename,identifier=identifier) +  ver_str + '.pkl'
pd.to_pickle(mean_features, save_dir)
save_dir = save_dir_root + '\mean_features_df_{identifier}_{filename}'.format(filename=filename,identifier=identifier) +  ver_str + '.xlsx'
mean_features_df.to_excel(save_dir, index = False) 
