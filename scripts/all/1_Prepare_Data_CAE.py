# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:58:34 2022

@author: Martin-PC
"""


#%%
# import os
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
# folder_csv = "\\bio-80\\" # T -80°C
# folder_csv = "\\bio20-4sen2\\" # T 20°C
# folder_csv = "\\CFE-100opt\\" # CFE T -100°C
folder_csv = "\\GFE20_4sen\\" # CFE T -100°C
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

suppress_plot = True
crop_ind = 0
i = 0
st_slik = 0
# num_coef = 64
# BatchLength = 128
num_coef = 16
BatchLength = 512

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
            w = 12.0
            # freq = np.linspace(1, fs/5, num_coef) 
            freq = np.linspace(1, fs/2, num_coef)  #a probam tokle
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
            # cwtm_raw = (cwtm_raw**2)
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
            if not suppress_plot:
                # if (j/len(dataAE['Time of Event'])*100) > 96:
                # if (j >= 23244):
                if st_slik <= 100:
                    # plt.pcolormesh(np.ones(BatchLength), freq, asd, cmap='viridis', shading='gouraud')
                    plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
                    # plt.pcolormesh(time_data, freq, abs_cwtm, cmap='viridis', shading='gouraud')
                    # plt.title(str(hit)+str('; w=')+str(w))
                    plt.show()
                    print(cwtm_raw.shape)
                    st_slik = st_slik + 1
                        
            cwtms[j][:num_coef,:Samples] = cwtm_raw
            # print(TOE)
            # max_coefs = np.apply_along_axis(max, axis=0, arr=cwtm_raw)
            # print(cwtm_raw.shape)
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
    
                
 
#%#%
#%%
# breakpoint()
name = '_' + str(num_coef) + '_' + str(BatchLength)
filename = r"C:\Users\Martin-PC\Biocomposite Analysis\cwtms_cfe_tok_scaled" + name +'.pkl'
pd.to_pickle(cwtms, filename)
