# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 12:58:34 2022

@author: Martin-PC
"""


#%%
# import os
import pandas as pd
import vallenae as vae
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
from multiprocessing import Pool

#%%
fs = 5000000
dt = 1/fs
def spectral_centroid(x, samplerate=fs):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    idx = find_nearest(freqs,samplerate/10) #max_freq
    return np.sum(magnitudes[:idx]*freqs[:idx]) / np.sum(magnitudes[:idx]) # return weighted mean

filename = '45-02s-11'
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen_2.pridb'
path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.pridb'.format(filename=filename)
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-112.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-113.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\gfe20_4sen.pridb'
# path_pri = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\GFEaliCFE-50-2sen.pridb'
pridb = vae.io.PriDatabase(path_pri)

hits = pridb.read_hits()

# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen_2.tradb'
path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.tradb'.format(filename=filename)
# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-112.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-113.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\gfe20_4sen.tradb'
# path_tra = r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\GFEaliCFE-50-2sen.tradb'

tradb = vae.io.TraDatabase(path_tra)
crop_ind = 0
hits_channel2_df = hits.copy()
#%%

file_identifier_list = [filename,
                   # 'bio_tok2',
                   # 'bio_cold',
                   # 'bio_cold2',
                   # 'bio_cold3',
                   # 'cfe_tok',
                   # 'cfe_cold'
                   ]

path_pri_files = [r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.pridb'.format(filename=filename),
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen_2.pridb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-11.pridb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-112.pridb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-113.pridb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\gfe20_4sen.pridb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\GFEaliCFE-50-2sen.pridb'
                  ]

path_tra_files = [r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\data2\{filename}.tradb'.format(filename=filename),
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\bio20_4sen_2.tradb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-11.tradb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-112.tradb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\data2\45-02s-113.tradb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\gfe20_4sen.tradb',
                  # r'C:\Users\Martin-PC\Biocomposite_analysis_isndcm\VallenDB\GFEaliCFE-50-2sen.tradb'
                  ]

def find_control_box_center_multiple(stripe, control_box_size_x, control_box_size_y):
    x_max = stripe.shape[1] - control_box_size_x
    y_max = stripe.shape[0] - control_box_size_y

    max_mean = -np.inf
    cb_center = None

    for y in range(y_max + 1):
        for x in range(x_max + 1):
            control_box = stripe[y:y+control_box_size_y, x:x+control_box_size_x]
            mean = control_box.mean(axis=(0, 1))
            if mean > max_mean:
                max_mean = mean
                cb_center = (x + control_box_size_x // 2, y + control_box_size_y // 2)

    return cb_center

def find_control_boxes_multiple(data, control_box_size_x, control_box_size_y, num_processes=4):
    # divide the data into num_processes stripes
    stripe_size = data.shape[0] // num_processes
    stripes = [data[i:i+stripe_size,:] for i in range(0, data.shape[0], stripe_size)]

    # create a pool of processes
    with Pool(num_processes) as pool:
        # apply the function to each stripe in parallel
        results = pool.starmap(find_control_box_center_multiple, [(stripe, control_box_size_x, control_box_size_y) for stripe in stripes])
        
    # extract the cb_center for each stripe
    cb_center_stripes = []
    for stripe_results in results:
        cb_center_stripe = []
        for cb_x, cb_y in stripe_results:
            cb_center_stripe.append((cb_x + control_box_size_x // 2, cb_y + control_box_size_y // 2))
        cb_center_stripes.append(cb_center_stripe)

    return cb_center_stripes


# def find_control_box_center(stripe, control_box_size_x, control_box_size_y):
#     x_max = stripe.shape[1] - control_box_size_x
#     y_max = stripe.shape[0] - control_box_size_y

#     max_mean = -np.inf
#     cb_center = None

#     for y in range(y_max + 1):
#         for x in range(x_max + 1):
#             control_box = stripe[y:y+control_box_size_y, x:x+control_box_size_x]
#             mean = control_box.mean(axis=(0, 1))
#             if mean > max_mean:
#                 max_mean = mean
#                 cb_center = (x + control_box_size_x // 2, y + control_box_size_y // 2)

#     return cb_center[0],cb_center[1]


for path_pri,path_tra,file_identifier in zip(path_pri_files,path_tra_files,file_identifier_list): 
    pridb = vae.io.PriDatabase(path_pri)
    hits = pridb.read_hits()   
    tradb = vae.io.TraDatabase(path_tra)
    crop_ind = 0
    hits_channel2_df = hits.copy()
    # hits_channel2_df = hits_channel2_df.iloc[0:10]
    #%%
    # time_controlbox_x = [0]*(int((len(hits_channel2_df))))
    # freq_controlbox_y = [0]*(int((len(hits_channel2_df))))
    # time_controlbox_x_stripe_1 = [0]*(int((len(hits_channel2_df))))
    # time_controlbox_x_stripe_2 = [0]*(int((len(hits_channel2_df))))
    # time_controlbox_x_stripe_3 = [0]*(int((len(hits_channel2_df))))
    # freq_controlbox_y_stripe_1 = [0]*(int((len(hits_channel2_df))))
    # freq_controlbox_y_stripe_2 = [0]*(int((len(hits_channel2_df))))
    # freq_controlbox_y_stripe_3 = [0]*(int((len(hits_channel2_df))))
    
    FCOG_list = [0]*(int((len(hits_channel2_df))))
    FMXA_list = [0]*(int((len(hits_channel2_df))))
    FVTA_list = [0]*(int((len(hits_channel2_df))))
    
    spectrum_energy0_150 = [0]*(int((len(hits_channel2_df))))
    spectrum_energy150_300 = [0]*(int((len(hits_channel2_df))))
    spectrum_energy300_500 = [0]*(int((len(hits_channel2_df))))
    spectrum_energy500_2000 = [0]*(int((len(hits_channel2_df))))
    #%%
    
    suppress_plot = True
    j = 0
    st_slik = 0
    num_coef = 80
    for index, row in hits_channel2_df.iterrows():
        hit = row['trai']
        channel = row['channel']
        try:
            [signal_data, time_data] = tradb.read_wave(hit)
        except:
            print('TRAI does not exists')
            print(str(hit))
            print(j)
            print(index)
            j += 1
            continue
    
        Samples = len(signal_data)
        # han_window = np.hanning(Samples)
        # signal_data_original = signal_data
        # signal_data = signal_data * han_window
        
        w = 7.0
        freq = np.linspace(1, fs/10, num_coef) #ch2
        # freq = np.linspace(100000,450000, num_coef) #ch1
        widths = w*fs / (2*freq*np.pi)
        cwtm = signal.cwt(signal_data, signal.morlet2, widths, w=w)
        abs_cwtm = np.abs(cwtm)
        max_coefs = np.apply_along_axis(max, axis=1, arr=abs_cwtm)
        # index_freq_cuttof = find_nearest(freq,475*1000) # od 200k naprej ignore
        # max_coefs[index_freq_cuttof:] = 0
        max_index = list(max_coefs).index(max(list(max_coefs)))
        peak_freq = freq[max_index] #peak_freq je po cwt
        x_index = np.where(abs_cwtm[max_index] == max(list(max_coefs)))[0][0]
        # breakpoint()
        # Define the size of the control box
        # control_box_size_x = 6
        # control_box_size_y = 6
        # num_cuts = 3
        # Split the signal array into 4 stripes along the x-axis
        # stripes = np.split(abs_cwtm, num_cuts, axis=0)
        # controlbox_x_123 = [None]*num_cuts
        # controlbox_y_123 = [None]*num_cuts
        
        # y_offsets = list(range(0,abs_cwtm.shape[0],int(abs_cwtm.shape[0]/num_cuts)))
        # for nl,(stripe,y_offset) in enumerate(zip(stripes,y_offsets)):
        #     cb_x, cb_y = find_control_box_center(stripe,control_box_size_x,control_box_size_y)
        #     controlbox_x_123[nl] = cb_x
        #     controlbox_y_123[nl] = cb_y + y_offset
            
        # for nl,stripe in enumerate(stripes):
        #     cb_x, cb_y = find_control_box_center(stripe,control_box_size_x,control_box_size_y)
        #     controlbox_x_123[nl] = cb_x
        #     controlbox_y_123[nl] = cb_y
        # find_control_boxes_multiple
        
        # cb_centers_stripes = find_control_boxes_multiple(abs_cwtm, control_box_size_x, control_box_size_y, num_processes=4)
        
        
        
        # time_controlbox_x_stripe_1[j] = time_data[controlbox_x_123[0]]
        # time_controlbox_x_stripe_2[j] = time_data[controlbox_x_123[1]]
        # time_controlbox_x_stripe_3[j] = time_data[controlbox_x_123[2]]
        
        # freq_controlbox_y_stripe_1[j] = freq[controlbox_y_123[0]]
        # freq_controlbox_y_stripe_2[j] = freq[controlbox_y_123[1]]
        # freq_controlbox_y_stripe_3[j] = freq[controlbox_y_123[2]]
        #global
        # cb_x, cb_y = find_control_box_center(abs_cwtm,control_box_size_x,control_box_size_y)
        
        # time_controlbox_x[j] = time_data[cb_x]
        # freq_controlbox_y[j] = freq[cb_y]
        

        
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
    
        f150_idx = find_nearest(freqs_spectrum,150*1000)
        f300_idx = find_nearest(freqs_spectrum,300*1000)
        f500_idx = find_nearest(freqs_spectrum,500*1000)
        f2000_idx = find_nearest(freqs_spectrum,2000*1000)
        FVTA_list[j] = peak_freq
        FCOG_list[j] = spectral_centroid(signal_data)
    
        total_spectrum_energy = sum(np.abs(spectrum))
        
        spectrum_energy0_150[j] = sum(np.abs(spectrum[:f150_idx]))/total_spectrum_energy
        spectrum_energy150_300[j] = sum(np.abs(spectrum[f150_idx:f300_idx]))/total_spectrum_energy
        spectrum_energy300_500[j] = sum(np.abs(spectrum[f300_idx:f500_idx]))/total_spectrum_energy
        spectrum_energy500_2000[j] = sum(np.abs(spectrum[f500_idx:f2000_idx]))/total_spectrum_energy         
        
        index_freq_cuttof_fft = find_nearest(freqs_spectrum,2000*1000) # od 2000k naprej ignore
        spectrum[index_freq_cuttof_fft:] = 0
        max_index_fft = np.argmax(spectrum)
        peak_freq_fft = freqs_spectrum[max_index_fft]
        FMXA_list[j] = peak_freq_fft
        if not suppress_plot:
            if st_slik <= 10:
                # plt.pcolormesh(np.ones(BatchLength), freq, asd, cmap='viridis', shading='gouraud')
                # plt.pcolormesh(time_data, freq, cwtm_raw, cmap='viridis', shading='gouraud')
                plt.pcolormesh(time_data, freq, abs_cwtm, cmap='viridis', shading='gouraud')
                # plt.title(str(hit)+str('; w=')+str(w))
                plt.show()
                st_slik = st_slik + 1
                    
        if j%50 == 0:
            print(j)
        j = j +1
    
    #%#%
    #%%
    # breakpoint()
    hits_channel2_df['FCOG'] = FCOG_list
    hits_channel2_df['FMXA'] = FMXA_list
    hits_channel2_df['FVTA'] = FVTA_list
    
    # hits_channel2_df['time_controlbox_x'] = time_controlbox_x
    # hits_channel2_df['freq_controlbox_y'] = freq_controlbox_y
    # hits_channel2_df['time_controlbox_x_stripe_1'] = time_controlbox_x_stripe_1
    # hits_channel2_df['time_controlbox_x_stripe_2'] = time_controlbox_x_stripe_2
    # hits_channel2_df['time_controlbox_x_stripe_3'] = time_controlbox_x_stripe_3
    # hits_channel2_df['freq_controlbox_y_stripe_1'] = freq_controlbox_y_stripe_1
    # hits_channel2_df['freq_controlbox_y_stripe_2'] = freq_controlbox_y_stripe_2
    # hits_channel2_df['freq_controlbox_y_stripe_3'] = freq_controlbox_y_stripe_3
    
    # hits_channel2_df['spectrum_energy0_25'] = spectrum_energy0_25
    # hits_channel2_df['spectrum_energy25_50'] = spectrum_energy25_50
    # hits_channel2_df['spectrum_energy50_100'] = spectrum_energy50_100
    # hits_channel2_df['spectrum_energy100_200'] = spectrum_energy100_200
    
    hits_channel2_df['spectrum_energy0_150'] = spectrum_energy0_150
    hits_channel2_df['spectrum_energy150_300'] = spectrum_energy150_300
    hits_channel2_df['spectrum_energy300_500'] = spectrum_energy300_500
    hits_channel2_df['spectrum_energy500_2000'] = spectrum_energy500_2000
    
    # df_merged['spectrum_energy100_200_ch1'] = spectrum_energy100_200_ch1
    # df_merged['spectrum_energy200_300_ch1'] = spectrum_energy200_300_ch1
    # df_merged['spectrum_energy300_450_ch1'] = spectrum_energy300_450_ch1
    
    
    # hits_channel2_df = df_merged.drop(['counts','trai'], axis=1)
    # # features_ch1 = df_merged.drop(['counts','trai'], axis=1)
    
    # hits_channel2_df = features.reset_index()
    
    name = '_' + str(num_coef) + '_' + str('Infinite_BatchLength')
    #%%
    # filename = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_ch2" + '.pkl'
    # filename = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_bio_cold" + name +'.pkl'
    filename = r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_w7_spectrum_energy_all_" + str(num_coef) + '_' + file_identifier + name +'.pkl'
    pd.to_pickle(hits_channel2_df, filename)
    hits_channel2_df.to_excel(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_w7_spectrum_energy_all_" + str(num_coef) + '_' + file_identifier + name +'.xlsx')
    
# features_merged_bio_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\classical_features_VALLEN_lambda_bio_cold3_60_Infinite_BatchLength.pkl")
# features_merged_cfe_cold = pd.read_pickle(r"C:\Users\Martin-PC\Biocomposite_analysis_isndcm\merged_features_embedded_classical_rev_w4_32x512_zeta_lower_bio_cold_32_64_32_512_16_10_32.pkl")

