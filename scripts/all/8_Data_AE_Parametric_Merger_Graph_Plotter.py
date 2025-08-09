# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:58:44 2022

@author: Martin-PC
"""

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
from datetime import timedelta, date
from datetime import datetime
from dateutil.relativedelta import relativedelta

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

#sampling frequency
# folder_csv = "\\bio-80\\" # T -80°C
# folder_csv = "\\bio20-4sen2\\" # T 20°C
# folder_csv = "\\CFE-100opt\\" # CFE T -100°C
folder_csv = "\\GFE20_4sen\\" # 
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
dataAE = pd.read_csv(os.getcwd()+folder_csv+name, encoding='cp1252')
dataAE['Time of Event Datetime'] = pd.to_datetime(dataAE['Time of Event'], format='%H:%M:%S.%f')


#%% bio-80
# dataAE_Index_loma =  5479
# dataAE_ID_loma = 6401
# parametric_index_loma = 1428
# cas_loma = datetime.datetime(1900, 1, 1, 13, 34, 3, 292927)
# parametric = pd.read_excel('podatki_bio_80_10ka.xlsx', index_col=None)
# parametric.dropna(inplace=True,axis=1)  
#%% bio20-4sen2
# dataAE_Index_loma =  754
# dataAE_ID_loma = 1119
# parametric_index_loma = 2496
# cas_loma = datetime.datetime(1900, 1, 1, 12, 24, 20, 990681)
# parametric = pd.read_excel('podatki_bio20_4sen2_9ka.xlsx', index_col=None)
# parametric.dropna(inplace=True,axis=1)
#%% CFE-100opt
# dataAE_Index_loma =  7589
# dataAE_ID_loma = 8801
# parametric_index_loma = 2018
# cas_loma = datetime.datetime(1900, 1, 1, 17, 10, 28, 242355)
# parametric = pd.read_excel('podatki_CFE_100opt_14ka.xlsx', index_col=None)
# parametric.dropna(inplace=True,axis=1)
#%% GFE20_4sen
dataAE_Index_loma =  3144
dataAE_ID_loma = 3917
parametric_index_loma = 2674
cas_loma = datetime.datetime(1900, 1, 1, 14, 3, 25, 546665)
parametric = pd.read_excel('podatki_GFE20_4sen_6ka.xlsx', index_col=None)
parametric.dropna(inplace=True,axis=1)  
#%%
datum_dt = datetime.strptime(dataAE.iloc[dataAE_Index_loma]['Time of Event'], '%H:%M:%S.%f')
parametric['Time of Event'] = datetime(year=1900,month=1,day=1)
parametric['ID'] = 0

parametric.loc[parametric_index_loma,'Time of Event'] = datum_dt
parametric.loc[parametric_index_loma,'ID'] = dataAE_ID_loma
time_ms_prelom = parametric.loc[parametric_index_loma,'time_ms']
#%% above
parametric_above = parametric.copy()
parametric_above.drop(parametric_above.index[parametric_index_loma+1:], inplace=True)
parametric_above = parametric_above.sort_index(ascending=False)

for ind,row in parametric_above.iterrows():
    # breakpoint()
    if ind == parametric_index_loma:
        parametric_above.loc[ind,'dt'] = 0
        continue
    parametric_above.loc[ind,'dt'] = (time_ms_prelom - row['time_ms'])/1000
    parametric_above.loc[ind,'Time of Event'] = datum_dt - relativedelta(seconds=(time_ms_prelom - row['time_ms'])/1000)


# parametric_above.loc[1427]['Time of Event']
#%% above slow
for ind,row in parametric_above.iterrows():
    nearest_dt = nearest(dataAE['Time of Event Datetime'], parametric_above.loc[ind]['Time of Event'])
    force_at_nearest_dt = parametric_above.loc[ind]['force_N']
    dataAE_nearest_dt_index = dataAE.index[dataAE['Time of Event Datetime'] == nearest_dt].tolist()
    dataAE.loc[dataAE_nearest_dt_index,'Force [N]'] = force_at_nearest_dt

# result_index = df['col_to_search'].sub(search_value).abs().idxmin()
#%% below
parametric_below = parametric.copy()
parametric_below.drop(parametric_below.index[:parametric_index_loma], inplace=True)
# parametric_below = parametric_below.sort_index(ascending=False)

for ind,row in parametric_below.iterrows():
    # breakpoint()
    if ind == parametric_index_loma:
        parametric_below.loc[ind,'dt'] = 0
        continue
    parametric_below.loc[ind,'dt'] = (time_ms_prelom - row['time_ms'])/1000
    parametric_below.loc[ind,'Time of Event'] = datum_dt - relativedelta(seconds=(time_ms_prelom - row['time_ms'])/1000)


# parametric_below.loc[1427]['Time of Event']
#%% below slow
for ind,row in parametric_below.iterrows():
    nearest_dt = nearest(dataAE['Time of Event Datetime'], parametric_below.loc[ind]['Time of Event'])
    force_at_nearest_dt = parametric_below.loc[ind]['force_N']
    dataAE_nearest_dt_index = dataAE.index[dataAE['Time of Event Datetime'] == nearest_dt].tolist()
    dataAE.loc[dataAE_nearest_dt_index,'Force [N]'] = force_at_nearest_dt

# result_index = df['col_to_search'].sub(search_value).abs().idxmin()
#%%
# parametric_concatenated = pd.concat([parametric_below,parametric_above],axis=0)
dataAE = dataAE.sort_values(by=['Time of Event Datetime'])
dataAE_interpolated = dataAE.interpolate(method ='linear', limit_direction ='forward')
dataAE_interpolated.to_excel("dataAE_interpolated_GFE20_4sen.xlsx")  

#%%
# dataAE_interpolated = pd.read_excel('dataAE_interpolated_bio-80.xlsx')
# dt_list = [None]*len(dataAE_interpolated)
# for ind,rows in dataAE_interpolated.iterrows():
#     dt_list[ind] = np.abs((dataAE_interpolated.iloc[0]['Time of Event Datetime'] - dataAE_interpolated.iloc[ind]['Time of Event Datetime']).total_seconds())
# dataAE_interpolated['time_elapsed_seconds'] = dt_list
# dataAE_interpolated.to_excel("dataAE_interpolated_bio-80_dt.xlsx") 
