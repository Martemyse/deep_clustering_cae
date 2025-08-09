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
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#%%
num_folds = 3
val_acc_list = [None]*num_folds
train_acc_list = [None]*num_folds
i = 0
for i in list(range(0,num_folds)):
    features_merged_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_cold_32_64_32_512_16_10_32.pkl")
    features_merged_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_32_512_16_10_32.pkl")
    # features_merged_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_cfe_tok_32_64_32_512_16_10_20.pkl")
    # features_merged_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_tok_32_64_32_512_16_10_20.pkl")
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
    features_merged_export.to_excel(r"C:\Users\Martin-PC\Biocomposite Analysis\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_32_64_32_512_16_10_32.xlsx", index = False) 
    labels_tok_t80 = np.array(features_merged_export['labels_bio-0_cfe-1'])
    num_labels_bio = len(features_merged_export.loc[features_merged_export['labels_bio-0_cfe-1'] == 0].reset_index(drop=True))
    num_labels_cfe = len(features_merged_export.loc[features_merged_export['labels_bio-0_cfe-1'] == 1].reset_index(drop=True))
    features_merged_export.drop('labels_bio-0_cfe-1', inplace=True, axis=1)
    features_merged = features_merged_export.copy()
    features_merged=features_merged.loc[:, (features_merged != 0).any(axis=0)]
    features_merged.drop('ID', inplace=True, axis=1)
    #%%
    # features_merged=(features_merged-features_merged.mean())/features_merged.std()
    scaler = MinMaxScaler()
    scaler.fit(features_merged)
    features_merged = scaler.transform(features_merged)
    #
    #%
    batch_size_samples = 1000
    index = 0
    index_array = np.arange(features_merged.shape[0])
    np.random.shuffle(index_array)
    idx = index_array[index * batch_size_samples: min((index+1) * batch_size_samples, features_merged.shape[0])]
    #%
    X_val = features_merged[idx,:]
    Y_val = labels_tok_t80[idx]
    
    mask = np.ones(len(features_merged), np.bool)
    mask[idx] = 0
    X_train = features_merged[mask,:]
    
    mask = np.ones(len(labels_tok_t80), np.bool)
    mask[idx] = 0
    Y_train = labels_tok_t80[mask]
    
    #%
    model = Sequential()
    # model.add(Dense(features_merged.shape[1]*2, input_shape=(features_merged.shape[1],), activation='relu')) # Add an input shape! (features,)
    model.add(Dense(50, input_shape=(features_merged.shape[1],), activation='relu')) # Add an input shape! (features,)
    model.add(Dense(25, activation='relu'))
    # model.add(Dense(features_merged.shape[1]*1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary() 
    
    # compile the model
    model.compile(optimizer='Adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # early stopping callback
    # This callback will stop the training when there is no improvement in  
    # the validation loss for 10 consecutive epochs.  
    # es = EarlyStopping(monitor='val_accuracy', 
    #                                    mode='max', # don't minimize the accuracy!
    #                                    patience=10,
    #                                    restore_best_weights=True)
    #%
    # now we just update our model fit call
    history = model.fit(X_train,
                        Y_train,
                        # callbacks=[es],
                        epochs=200, # you can set this to a big number!
                        batch_size=32,
                        # batch_size=64,
                        # validation_split=0.01,
                        validation_data=(X_val,Y_val),
                        shuffle=True,
                        verbose=2)
    
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    # Learning curve(accuracy)
    # let's see the training and validation accuracy by epoch
    
    # accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # range of X (no. of epochs)
    epochs = range(1, len(acc) + 1)
    #%
    # plot
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # orange is for "orange"
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Vse wavelet width rev w4')
    plt.suptitle('Razlocevanje BIO/CFE pri nizkih temp')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # this is the max value - should correspond to
    # the HIGHEST train accuracy
    np.max(val_acc)
    train_acc_list[i] = acc
    val_acc_list[i]=val_acc
    

# ver_str = "_" + str(layers_depths[0]) + "_" + str(layers_depths[1]) + "_" + str(layer_depth_latent_cnn) + "_" + str(layers_depths[2]) + "_" + str(layers_depths[3]) + "_" + str(train_epochs) + "_" + str(batch_size)
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite Analysis\Clustering_CAE'
# save_dir = save_dir_root + '\AE_x2_x1_merged_features_embedded_classical_w4_32x512_zeta_lower_cfe_cold_32_64_32_512_16_15_32'  + '.h5'
# model.save_weights(save_dir)

#%%
# save_dir_root = r'C:\Users\Martin-PC\Biocomposite Analysis\kfold_Validation'
# save_dir = save_dir_root + '\\val_acc_merged_features_only_classical_rev_w4_32x512_&_200_Epochs_200_50_25_x2_x1' + '.pkl'
# df_val_acc = pd.DataFrame(val_acc_list).T
# df_val_acc.columns = ['Fold1', 'Fold2', 'Fold3']
# pd.to_pickle(df_val_acc, save_dir)

# save_dir = save_dir_root + '\\train_acc_merged_features_only_classical_rev_w4_32x512_&_200_Epochs_200_50_25_x2_x1' + '.pkl'
# df_train_acc = pd.DataFrame(train_acc_list).T
# df_train_acc.columns = ['Fold1', 'Fold2', 'Fold3']
# pd.to_pickle(df_train_acc, save_dir)
# # model.load_weights(save_dir)

# for fold in ['Fold1','Fold2','Fold3']:
#     print(np.round(df_val_acc[fold][-10:].mean(),decimals=3))
    
# #%%
# df_val_acc = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite Analysis\kfold_Validation\acc_merged_features_embedded_classical_rev_w4_32x512\val_acc_merged_features_embedded_classical_rev_w4_32x512_zeta_lower_32_64_32_512_16_10_32_&_200_Epochs_200_50_25_x2_x1.pkl")
# for fold in ['Fold1','Fold2','Fold3']:
#     print(np.round(df_val_acc[fold][-10:].mean(),decimals=3))
#%%
breakpoint()
y_DT = np.round(model.predict(features_merged),0)
y_DT=np.reshape(y_DT, (-1))
true_false = labels_tok_t80 == y_DT
features_merged_export['true_false'] = true_false
features_merged_export['labels_bio-0_cfe-1'] = labels_tok_t80
features_merged_export_bio_missclassified = features_merged_export.loc[(features_merged_export['labels_bio-0_cfe-1'] == 0) & (features_merged_export['true_false'] == False)]
features_merged_export_cfe_missclassified = features_merged_export.loc[(features_merged_export['labels_bio-0_cfe-1'] == 1) & (features_merged_export['true_false'] == False)]
# pyplot.show()
#%%
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# def create_larger():
#  	# create model
#  	model = Sequential()
#  	model.add(Dense(features_merged.shape[1]*2, input_dim=features_merged.shape[1], activation='relu'))
#  	model.add(Dense(features_merged.shape[1], activation='relu'))
#  	model.add(Dense(1, activation='sigmoid'))
#  	# Compile model
#  	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#  	return model
# estimators = []
# # estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_larger,
#                                           epochs=250,
#                                           batch_size=64,
#                                           verbose=1,
#                                           validation_data=(X_val,Y_val),
#                                           shuffle=True)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=3, shuffle=True)
# results = cross_val_score(pipeline, X_train,
#                     Y_train, cv=kfold)
# print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#%%
# y_DT = np.round(model.predict(features_merged),0)
# true_false = labels_tok_t80 == y_DT
# features_merged_export['true_false'] = true_false
# features_merged_export['labels_bio-0_cfe-1'] = labels_tok_t80
# features_merged_export_bio_missclassified = features_merged_export.loc[(features_merged_export['labels_bio-0_cfe-1'] == 1) & (features_merged_export['true_false'] == False)]
# pyplot.show()

bac_all = balanced_accuracy_score(labels_tok_t80, y_DT)
ac_all = accuracy_score(labels_tok_t80, y_DT)

#%%

# breakpoint()
# features_merged_export_missclassified = features_merged_export_bio_missclassified
features_merged_export_missclassified = features_merged_export_cfe_missclassified
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
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
from nptdms import TdmsFile
import datetime #for time calculations
#sampling frequency
folder_csv = "\\bio-80\\" # T -80°C
# folder_csv = "\\bio20-4sen2\\" # T 20°C
# folder_csv = "\\CFE-100opt\\" # CFE T -100°C
# folder_csv = "\\GFE20_4sen\\" # CFE T -100°C
fs=952381
#window size for FFT
WS=128
#find a csv file in a directory
# breakpoint()
files = [f for f in os.listdir(os.getcwd()+folder_csv)]
for f in files:
    if f.endswith('.csv'):
        name=f
        
#read csv file
dataAE= pd.read_csv(os.getcwd()+folder_csv+name, encoding='cp1252')
#the string below is to ensure the right format for time
dataAE['Time of Event'] = pd.to_datetime(dataAE['Time of Event'], format="%H:%M:%S.%f")
#for loop to read all tdms file waveforms

suppress_plot = False
crop_ind = 0
i = 0
st_slik = 0
num_coef = 80
BatchLength = 800

# breakpoint()
# dataAE = dataAE.loc[dataAE['Channel Name'] == 'Ch2'].reset_index(drop=True)
cwtms = np.empty((len(dataAE), num_coef-crop_ind, BatchLength), dtype="float16")

# cwtms_pure = np.empty((len(channel1), num_coef-crop_ind, BatchLength), dtype="float16")
# breakpoint()
n = 0
for k in range(len(dataAE['Time of Event'])):
    # breakpoint()
    # print('n:')
    # print(n)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ftdms=os.getcwd()+folder_csv+'\\Threshold_Trigger_Wfm\\'+str(dataAE['TDMS File Name'][n])+".tdms"
    print(ftdms)
    with TdmsFile.open(ftdms) as tdms_file:
        dataAE_tdm = dataAE.loc[dataAE['TDMS File Name'] == str(dataAE['TDMS File Name'][n])]
        j = n
        for m in range(len(dataAE_tdm['Time of Event'])):
            try:
                channel_number=dataAE_tdm['Channel Name'][j]
            except:
                print('Skipped!')
                print(str(j))
                j += 1
                continue
            
            channel = tdms_file['Wfm Data'][str(dataAE_tdm['ID'][j])+" _ "+str(dataAE_tdm['Channel Name'][j])]
            # breakpoint()
            signal_data = channel[:]    
            #you can use data to perform any analysis for instance, FFT
            # if not dataAE_tdm['ID'][j] in features_merged_export_bio_missclassified['ID']:
            #     print(dataAE_tdm['ID'][j])
                
            #     continue
            dicttosave={}
            dicttosave["Amplitude (nm)"]=signal_data
            df=pd.DataFrame.from_dict(dicttosave)
            #to find an index in tdms waveform, which corresponds to time pf event recorded in the CSV file, you can use the code below
            TH = tdms_file['Wfm Data'][str(dataAE_tdm['ID'][j])+" _ "+str(dataAE_tdm['Channel Name'][j])].properties['TH (nm)']
            ind=df[abs(df['Amplitude (nm)'])>TH].head(1).index.values[0]#the index corresponding to threshold crossingc
            time_data=np.linspace(0, (len(signal_data)-1)/fs, len(signal_data))
            TOE=dataAE_tdm['Time of Event'][j]
            ts=[]
            ch=[]
            for i in range(len(signal_data)):
                ch.append(round((time_data[i]-ind/fs)*1000000))
                dif=TOE+datetime.timedelta(microseconds=ch[i])
                ts.append(dif.time())
            dicttosave["Time (s)"]=time_data
            dicttosave["Timestamp"]=ts
            #save again but with timestamps
            df=pd.DataFrame.from_dict(dicttosave)
            # breakpoint()
            start_crop = 0
            # start_crop = int(len(signal_data)/12)
            end_crop = start_crop + BatchLength
            time_data = time_data[start_crop:end_crop] # us
            signal_data = signal_data[start_crop:end_crop]
            Samples = len(signal_data)
            w = 4
            # freq = np.linspace(1, fs/5, num_coef) 
            freq = np.linspace(1, fs/10, num_coef)  #a probam tokle
            widths = w*fs / (2*freq*np.pi)
            # signal_data_norm = Normalize_Data_Signal(signal_data)
            # scaler = MinMaxScaler(feature_range=(0, 1))
            # breakpoint()
            cwtm = signal.cwt(signal_data, signal.morlet2, widths,w=w)
            # cwtm = signal.cwt(signal_data, signal.morlet2, widths)
            # breakpoint()
            cwtm_raw = np.abs(cwtm)
            # breakpoint()
            # max_coefs = np.apply_along_axis(max, axis=0, arr=cwtm_raw)
            # cwtm_raw = np.sqrt(cwtm)
            # cwtm_raw_max = cwtm_raw.max()
            cwtm_raw = (cwtm_raw - np.mean(cwtm_raw)) / np.std(cwtm_raw)
            cwtm_raw = (cwtm_raw**2)
            cwtm_raw = cwtm_raw/cwtm_raw.max() #rescaled to original
            # cwtm_raw = cwtm_raw/cwtm_raw.max() * cwtm_raw_max #rescaled to original
            # cwtm_raw = np.sqrt(cwtm_raw + cwtm_raw.min() + 5)
            
            # cwtm_raw_max = cwtm_raw.max()
            # cwtm_raw_max = cwtm_raw.min()
            # cwtm_raw_scaled = cwtm_raw/cwtm_raw_max
            # breakpoint()
            # cwtm_raw_log = np.log(cwtm_raw)
            # cwtm_raw_scaled = abs_cwtm*10e1
            # abs_cwtm_ravel = np.array(abs_cwtm).ravel()
            # occurances, bins, patches = plt.hist(np.array(abs_cwtm).ravel(), bins=10, density=True);
            # sum_occurances = sum(occurances)
            # bin_percantages = [x / sum_occurances for x in occurances]
            # plt.xlabel("pixel values")
            # plt.ylabel("relative frequency")
            # plt.title("distribution of pixels")
            # plt.show()
            # abs_cwtm = abs_cwtm[crop_ind:,:]
            # asd = cwtm_raw.resize((num_coef,BatchLength))
            # asd = cwtm_raw.transpose().copy()
            
            # asd = cwtm_raw.copy()
            # asd.resize((num_coef,BatchLength), refcheck=False)
            # asd.resize((BatchLength,num_coef), refcheck=False)
            # asd = cwtm_raw.transpose()
            # breakpoint()
            if str(dataAE_tdm['ID'][j]) in features_merged_export_missclassified['ID'].astype(int).astype(str).values:
                # breakpoint()
                # if (j/len(dataAE['Time of Event'])*100) > 96:
                if st_slik <= 1000:
                    # plt.pcolormesh(np.ones(BatchLength), freq, asd, cmap='viridis', shading='gouraud')
                    plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
                    # plt.pcolormesh(time_data, freq, abs_cwtm, cmap='viridis', shading='gouraud')
                    # plt.title(str(hit)+str('; w=')+str(w))
                    plt.show()
                    st_slik = st_slik + 1
                    print(cwtm_raw.shape)
                
                        
            cwtms[j][:num_coef,:Samples] = cwtm_raw
            # print(TOE)
            max_coefs = np.apply_along_axis(max, axis=0, arr=cwtm_raw)
            # breakpoint()
            # print('j:')
            # print(j)
            
            # print('**********************')
            j = j +1
            # breakpoint()
            
        # print(channel)
    j -= 1
    n = j
    n += 1

    print('Completed: ')
    print(str(j/len(dataAE['Time of Event'])*100))
    