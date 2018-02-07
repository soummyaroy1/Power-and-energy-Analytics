''' Energy_Consumption'''

import pandas as pd
import numpy as np
from scipy.io import loadmat as ld
import matplotlib.pyplot as plt
import tables
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy import signal
import scipy

##Part1

#..........read the .mat data..........#
HVAC        = ld('HVAC.mat')
LIGHTING    = ld('LIGHTING.mat')
MAINS       = ld('MAINS.mat')
OTHER_LOADS = ld('OTHER_LOADS.mat')
APPLIANCES  = ld('APPLIANCES.mat')

#..........reshape the data..........#
MAINSdd         = MAINS['y1'].reshape(262800)
HVACdd          = HVAC['y2'].reshape(262800) 
LIGHTINGdd      = LIGHTING['y3'].reshape(262800)
APPLIANCESdd    = APPLIANCES['y4'].reshape(262800)
OTHER_LOADSdd   = OTHER_LOADS['y5'].reshape(262800)

#..........put the data in a DataFrame..........#
data = {'MAINS':MAINSdd , 'HVAC':HVACdd , 'LIGHTING':LIGHTINGdd , 'APPLIANCE':APPLIANCESdd , 'OTHER_LOAD':OTHER_LOADSdd }
data1 = pd.DataFrame(data=data)



x = [x for x in range (len(data1['APPLIANCE']))]

y = [y for y in range (len(data1['HVAC']))]


z = [z for z in range (len(data1['LIGHTING']))]

w = [w for w in range (len(data1['OTHER_LOAD']))]

plt.figure(1)

plt.subplot(411)
plt.plot(x,data1['APPLIANCE'])

plt.subplot(412)
plt.plot(y,data1['HVAC'])

plt.subplot(413)
plt.plot(z,data1['LIGHTING'])

plt.subplot(414)
plt.plot(w,data1['OTHER_LOAD'])

plt.show()


#..........Problem1     Pre_processing of all the variables by normalization..........#
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data1)
data_n = pd.DataFrame(np_scaled)


data_n.columns = ['Appliance' , 'Hvac' , 'Lighting' , 'Mains' , 'Other_load']



plt.figure(2)

plt.subplot(411)
plt.plot(x,data_n['Appliance'])

plt.subplot(412)
plt.plot(y,data_n['Hvac'])

plt.subplot(413)
plt.plot(z,data_n['Lighting'])

plt.subplot(414)
plt.plot(w,data_n['Other_load'])

plt.show()

##...................Problem2   Euclidean distance............................#

from sklearn.metrics.pairwise import euclidean_distances as ed

A_H=[data1['APPLIANCE'],data1['HVAC']]
A_H_ary = np.array(A_H)
A_H_ed = ed(A_H)

A_L=[data1['APPLIANCE'],data1['LIGHTING']]
A_L_ary = np.array(A_L)
A_L_ed = ed(A_L)

A_M=[data1['APPLIANCE'],data1['MAINS']]
A_M_ary = np.array(A_M)
A_M_ed = ed(A_M)

A_O=[data1['APPLIANCE'],data1['OTHER_LOAD']]
A_O_ary = np.array(A_O)
A_O_ed = ed(A_O)

H_L=[data1['HVAC'],data1['LIGHTING']]
H_L_ary = np.array(H_L)
H_L_ed = ed(H_L)

H_M=[data1['HVAC'],data1['MAINS']]
H_M_ary = np.array(H_M)
H_M_ed = ed(H_M)

H_O=[data1['HVAC'],data1['OTHER_LOAD']]
H_O_ary = np.array(H_O)
H_O_ed = ed(H_O)

L_M=[data1['LIGHTING'],data1['MAINS']]
L_M_ary = np.array(L_M)
L_M_ed = ed(L_M)

L_O=[data1['LIGHTING'],data1['OTHER_LOAD']]
L_O_ary = np.array(L_O)
L_O_ed = ed(L_O)

M_O=[data1['MAINS'],data1['OTHER_LOAD']]
M_O_ary = np.array(M_O)
M_O_ed = ed(M_O)

#....................Problem3........................#

appliance_fft = np.fft.fft(data1['APPLIANCE'])
hvac_fft = np.fft.fft(data1['HVAC'])
lighting_fft = np.fft.fft(data1['LIGHTING'])
main_fft = np.fft.fft(data1['MAINS'])
ohterload_fft = np.fft.fft(data1['OTHER_LOAD'])

A_H_fft=[appliance_fft,hvac_fft]
A_H_ffted = ed(A_H_fft)

A_L_fft=[appliance_fft,lighting_fft]
A_L_ffted = ed(A_L_fft)

A_M_fft=[appliance_fft,main_fft]
A_M_ed = ed(A_M_fft)

A_O_fft=[appliance_fft,ohterload_fft]
A_O_ffted = ed(A_O_fft)

H_L_fft=[hvac_fft,lighting_fft]
H_L_ffted = ed(H_L_fft)

H_M_fft=[hvac_fft,main_fft]
H_M_ffted = ed(H_M_fft)

H_O_fft=[hvac_fft,ohterload_fft]
H_O_ffted = ed(H_O_fft)

L_M_fft=[lighting_fft,main_fft]
L_M_ffted = ed(L_M_fft)

L_O_fft=[lighting_fft,ohterload_fft]
L_O_ffted = ed(L_O_fft)

M_O_fft=[main_fft,ohterload_fft]
M_O_ffted = ed(M_O_fft)


#...................Problam4 KL_Divegence....................#

from sklearn.metrics.cluster import mutual_info_score

appliance_ary = np.array(data1['APPLIANCE'])
hvac_ary = np.array(data1['HVAC'])
lighting_ary = np.array(data1['LIGHTING'])
mains_ary = np.array(data1['MAINS'])
otherload_ary = np.array(data1['OTHER_LOAD'])

a_h = mutual_info_score(appliance_ary,hvac_ary)
a_l = mutual_info_score(appliance_ary,lighting_ary)
a_m = mutual_info_score(appliance_ary,mains_ary)
a_o = mutual_info_score (appliance_ary,otherload_ary)
h_l = mutual_info_score(hvac_ary,lighting_ary)
h_m = mutual_info_score(hvac_ary,mains_ary)
h_o = mutual_info_score(hvac_ary,otherload_ary)
l_m = mutual_info_score(lighting_ary,mains_ary) 
l_o = mutual_info_score(lighting_ary,otherload_ary)
m_o = mutual_info_score(mains_ary,otherload_ary)

#
##......................Problem5 wavelet transform...............#

import pywt

appliance_wt = pywt.dwt(data1['APPLIANCE'],'db1')
hvac_wt      = pywt.dwt(data1['HVAC'],'db1')
lighting_wt  = pywt.dwt(data1['LIGHTING'],'db1')
mains_wt     = pywt.dwt(data1['MAINS'],'db1')
otherload_wt = pywt.dwt(data1['OTHER_LOAD'],'db1')





#######Problem 6 ######


d_main = data1['MAINS']
main_fft = np.fft.fft(d_main)

fs = 1e3
f, t, Sxx = signal.spectrogram(d_main,fs)
plt.figure(6)
plt.pcolormesh(t,f,Sxx)
plt.ylim(0,20)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [Hour]')
plt.colorbar()
plt.show()