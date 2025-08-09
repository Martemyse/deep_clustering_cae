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

#%%
#sampling frequency
fs=952381
#window size for FFT
WS=128
#find a csv file in a directory
# breakpoint()

def spectral_centroid(x, samplerate=fs):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    idx = find_nearest(freqs,samplerate/2) #max_freq
    return np.sum(magnitudes[:idx]*freqs[:idx]) / np.sum(magnitudes[:idx]) # return weighted mean

folder_csv = "\\bio-80\\" # T -80°C
# folder_csv = "\\bio20-4sen2\\" # T 20°C
# folder_csv = "\\CFE-100opt\\" # CFE T -100°C
# folder_csv = "\\GFE20_4sen\\" # CFE T -100°C

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
# BatchLength = 512

# breakpoint()
# dataAE = dataAE.loc[dataAE['Channel Name'] == 'Ch2'].reset_index(drop=True)

FCOG_list = [0]*(int((len(dataAE))))
# FMXA_list_ch1 = [0]*(int((len(filtered_hits))))
FMXA_list = [0]*(int((len(dataAE))))
FVTA_list = [0]*(int((len(dataAE))))
# FVTA_list_ch1 = [0]*(int((len(filtered_hits))))
# spectrum_energy0_200_ch1 = [0]*(int((len(filtered_hits))))
# spectrum_energy200_400_ch1 = [0]*(int((len(filtered_hits))))
# spectrum_energy400_1000_ch1 = [0]*(int((len(filtered_hits))))
# spectrum_energy0_25 = [0]*(int((len(dataAE))))
# spectrum_energy25_50 = [0]*(int((len(dataAE))))
# spectrum_energy50_100 = [0]*(int((len(dataAE))))
# spectrum_energy100_200 = [0]*(int((len(dataAE))))

spectrum_energy0_75 = [0]*(int((len(dataAE))))
spectrum_energy75_150 = [0]*(int((len(dataAE))))
spectrum_energy150_300 = [0]*(int((len(dataAE))))
spectrum_energy300_475 = [0]*(int((len(dataAE))))

# cwtms = np.empty((len(dataAE), num_coef-crop_ind, BatchLength), dtype="float16")

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
            
            # if channel_number == 'Ch2':
                # continue
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
            # start_crop = 0
            # start_crop = int(len(signal_data)/12)
            # end_crop = start_crop + BatchLength
            # time_data = time_data[start_crop:end_crop] # us
            # signal_data = signal_data[start_crop:end_crop]
            
            # breakpoint()
            Samples = len(signal_data)
            han_window = np.hanning(Samples)
            signal_data_original = signal_data
            signal_data = signal_data * han_window
            
            w = 9.0
            freq = np.linspace(1, fs/2, num_coef) #ch2
            # freq = np.linspace(100000,450000, num_coef) #ch1
            widths = w*fs / (2*freq*np.pi)
            cwtm = signal.cwt(signal_data_original, signal.morlet2, widths, w=w)
            abs_cwtm = np.abs(cwtm)
            max_coefs = np.apply_along_axis(max, axis=1, arr=abs_cwtm)
            index_freq_cuttof = find_nearest(freq,475*1000) # od 200k naprej ignore
            max_coefs[index_freq_cuttof:] = 0
            max_index = list(max_coefs).index(max(list(max_coefs)))
            peak_freq = freq[max_index] #peak_freq je po cwt
            
            spectrum = fftpack.fft(signal_data)
            polovica = int(Samples/2)
            spectrum = np.abs(spectrum[:polovica])
            # spectrum[0:2] = 0
            max_spectrum = np.array(spectrum).max()
                        # pure_spectrum = spectrum**4
            # max_pure_spectrum = np.array(pure_spectrum).max()
            # pure_spectrum_scaled = pure_spectrum/max_pure_spectrum*max_spectrum
            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@ RAW or PURE Spectrum @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # spectrum = pure_spectrum_scaled
            freqs_spectrum = fftpack.fftfreq(len(signal_data)) * fs
            freqs_spectrum = freqs_spectrum[:polovica]
            # f25_idx = find_nearest(freqs_spectrum,25*1000)
            # f50_idx = find_nearest(freqs_spectrum,50*1000)
            # f100_idx = find_nearest(freqs_spectrum,100*1000)
            # f200_idx = find_nearest(freqs_spectrum,200*1000)
            # f100_idx = find_nearest(freqs_spectrum,100*1000)
            # f200_idx = find_nearest(freqs_spectrum,200*1000)
            # f300_idx = find_nearest(freqs_spectrum,300*1000)
            # f450_idx = find_nearest(freqs_spectrum,450*1000)
            f75_idx = find_nearest(freqs_spectrum,75*1000)
            f150_idx = find_nearest(freqs_spectrum,150*1000)
            f300_idx = find_nearest(freqs_spectrum,300*1000)
            f475_idx = find_nearest(freqs_spectrum,475*1000)
            FVTA_list[j] = peak_freq
            FCOG_list[j] = spectral_centroid(signal_data)
            # FVTA_list_ch1[i] = peak_freq
            # FMXA_list_ch1[i] = peak_freq_fft
            # FCOG_list_ch1[i] = spectral_centroid(signal_data)
            
            total_spectrum_energy = sum(np.abs(spectrum))
            # spectrum_energy0_25[j] = sum(np.abs(spectrum[:f25_idx]))/total_spectrum_energy
            # spectrum_energy25_50[j] = sum(np.abs(spectrum[f25_idx:f50_idx]))/total_spectrum_energy
            # spectrum_energy50_100[j] = sum(np.abs(spectrum[f50_idx:f100_idx]))/total_spectrum_energy
            # spectrum_energy100_200[j] = sum(np.abs(spectrum[f100_idx:f200_idx]))/total_spectrum_energy
            
            spectrum_energy0_75[j] = sum(np.abs(spectrum[:f75_idx]))/total_spectrum_energy
            spectrum_energy75_150[j] = sum(np.abs(spectrum[f75_idx:f150_idx]))/total_spectrum_energy
            spectrum_energy150_300[j] = sum(np.abs(spectrum[f150_idx:f300_idx]))/total_spectrum_energy
            spectrum_energy300_475[j] = sum(np.abs(spectrum[f300_idx:f475_idx]))/total_spectrum_energy         
            
            index_freq_cuttof_fft = find_nearest(freqs_spectrum,475*1000) # od 200k naprej ignore
            spectrum[index_freq_cuttof_fft:] = 0
            max_index_fft = np.argmax(spectrum)
            peak_freq_fft = freqs_spectrum[max_index_fft]
            FMXA_list[j] = peak_freq_fft
            # if not suppress_plot:
            #     if st_slik <= 10:
            #         # plt.pcolormesh(np.ones(BatchLength), freq, asd, cmap='viridis', shading='gouraud')
            #         plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
            #         # plt.pcolormesh(time_data, freq, abs_cwtm, cmap='viridis', shading='gouraud')
            #         # plt.title(str(hit)+str('; w=')+str(w))
            #         plt.show()
            #         st_slik = st_slik + 1
                        
            # cwtms[j][:num_coef,:Samples] = cwtm_raw
            # print(TOE)
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
    print(str(n/len(dataAE['Time of Event'])*100))
    
                
 
#%#%
#%%
# breakpoint()
dataAE['FCOG'] = FCOG_list
dataAE['FMXA'] = FMXA_list
dataAE['FVTA'] = FVTA_list

# dataAE['spectrum_energy0_25'] = spectrum_energy0_25
# dataAE['spectrum_energy25_50'] = spectrum_energy25_50
# dataAE['spectrum_energy50_100'] = spectrum_energy50_100
# dataAE['spectrum_energy100_200'] = spectrum_energy100_200

dataAE['spectrum_energy0_75'] = spectrum_energy0_75
dataAE['spectrum_energy75_150'] = spectrum_energy75_150
dataAE['spectrum_energy150_300'] = spectrum_energy150_300
dataAE['spectrum_energy300_475'] = spectrum_energy300_475

# df_merged['spectrum_energy100_200_ch1'] = spectrum_energy100_200_ch1
# df_merged['spectrum_energy200_300_ch1'] = spectrum_energy200_300_ch1
# df_merged['spectrum_energy300_450_ch1'] = spectrum_energy300_450_ch1


# dataAE = df_merged.drop(['counts','trai'], axis=1)
# # features_ch1 = df_merged.drop(['counts','trai'], axis=1)

# dataAE = features.reset_index()

name = '_' + str(num_coef) + '_' + str('Infinite_BatchLength')
#%%
# filename = r"C:\Users\Martin-PC\Biocomposite Analysis\classical_features_ch2" + '.pkl'
filename = r"C:\Users\Martin-PC\Biocomposite Analysis\classical_features_lambda_cfe_tok" + name +'.pkl'
# pd.to_pickle(dataAE, filename)
