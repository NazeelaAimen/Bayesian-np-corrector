#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:48:04 2024

@author: naim769
"""
import bnpc
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
import rpy2
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import pickle
import time
import argparse
import pandas as pd
import os

result=[]
result_r=[]
#R
np_cv_rules = default_converter + numpy2ri.converter
robjects.r['source']('psd_arma.R')
psd_arma = robjects.globalenv['psd_arma']
mcmcr=importr("psplinePsd")

rd=63
num = [128, 256, 512]

#AR(4) coefficients
sig=1
#AR(1) coefficients
a1p=0.1883
sig2p=7.839
arex_r = robjects.FloatVector(np.array([0.9, -0.9, 0.9, -0.9]))
ma_ex = robjects.NA_Integer
#number of data points
n=[128,256,512]
series=[]
freq=[]
truepsd=[]
spar=[]



for i in range(0,len(n)):
    random.seed(rd)
    #time series AR(4)
    series.append({f'{n[i]}': pd.read_csv(f"ts{n[i]}.csv").iloc[i]})
    #frequency[0,pi]
    freq.append({f'{n[i]}': np.pi*( 2 * (np.arange(1, n[i] // 2 + 1) - 1) / n[i])[1:]})
    f_r=robjects.FloatVector(freq[i][f'{n[i]}'])
    #truepsd
    truepsd.append({f'{n[i]}':np.log(psd_arma(f_r,arex_r,ma_ex,sig))})
    spar.append({f'{n[i]}':psd_arma(f_r,a1p,ma_ex,sig2p)})
    file_path =f'sim_res_{n[i]}/{rd}resultpyAMH_AR(1).pkl'
    with open(file_path, 'rb') as file:
        result.append(pickle.load(file))
    file_path=f'sim_res_{n[i]}/{rd}resultr.pkl'
    with open(file_path, 'rb') as file:
        result_r.append(pickle.load(file))
    


# PSD plot for an instance:
#n=128
S=result[0]['psd']+2*np.log(np.std(series[0][f'{n[0]}']))
ci_py = bnpc.compute_ci(S)  

# splines
Sp=result[0]['splines_psd']+2*np.log(np.std(series[0][f'{n[0]}']))
ci_pysp = bnpc.compute_ci(Sp)  

#n=256
S1=result[1]['psd']+2*np.log(np.std(series[1][f'{n[1]}']))
ci_py1 = bnpc.compute_ci(S1)  
#splines
Sp1=result[1]['splines_psd']+2*np.log(np.std(series[1][f'{n[1]}']))
ci_pysp1= bnpc.compute_ci(Sp1)
#n=512
S2=result[2]['psd']+2*np.log(np.std(series[2][f'{n[2]}']))
ci_py2 = bnpc.compute_ci(S2)  
#splines
Sp2=result[2]['splines_psd']+2*np.log(np.std(series[2][f'{n[2]}']))
ci_pysp2 = bnpc.compute_ci(Sp2)  


#pdgrm:
pdgrm=result_r[0]['pdgrm'][1:-1]+2*np.log(np.std(series[0][f'{n[0]}']))
pdgrm1=result_r[1]['pdgrm'][1:-1]+2*np.log(np.std(series[1][f'{n[1]}']))
pdgrm2=result_r[0]['pdgrm'][1:-1]+2*np.log(np.std(series[2][f'{n[2]}']))

#
f=freq[0][f'{n[0]}'])
f1=freq[1][f'{n[1]}'])
f2=freq[2][f'{n[2]}'])
# PSD plot
fig, axs = plt.subplots(3, 1, figsize=(10, 15))


axs[0].plot(f, np.log(pdgrm), linestyle='-', color='black', alpha=0.4,label='Periodogram')
axs[0].plot(f, truepsd[0][f'{n[0]}'], linestyle='--', color='black', label='True')
axs[0].plot(f, ci_pysp.med, linestyle='-', color='purple', label='Splines')
axs[0].fill_between(f, ci_pysp.u05, ci_pysp.u95, color='purple', alpha=0.2, linewidth=0.0)
axs[0].plot(f, np.log(spar[0][f'{n[0]}']), linestyle='-', color='blue', label='AR1')
axs[0].fill_between(f, ci_py.u05, ci_py.u95, color='red', alpha=0.2, linewidth=0.0)
axs[0].plot(f, ci_py.med, linestyle='-', color='red', label='Estimated')


axs[1].plot(f1, np.log(pdgrm1), linestyle='-', color='black', alpha=0.4)
axs[1].plot(f1, truepsd[1][f'{n[1]}'], linestyle='--', color='black')
axs[1].plot(f1, ci_pysp1.med, linestyle='-', color='purple')
axs[1].fill_between(f1, ci_pysp1.u05, ci_pysp1.u95, color='purple', alpha=0.2, linewidth=0.0)
axs[1].plot(f1, np.log(spar[1][f'{n[1]}']), linestyle='-', color='blue')
axs[1].fill_between(f1, ci_py1.u05, ci_py1.u95, color='red', alpha=0.2, linewidth=0.0)
axs[1].plot(f1, ci_py1.med, linestyle='-', color='red')

axs[2].plot(f2, np.log(pdgrm2), linestyle='-', color='black', alpha=0.4)
axs[2].plot(f2, truepsd[2][f'{n[2]}'], linestyle='--', color='black')
axs[2].plot(f2, ci_pysp2.med, linestyle='-', color='purple')
axs[2].fill_between(f2, ci_pysp2.u05, ci_pysp2.u95, color='purple', alpha=0.2, linewidth=0.0)
axs[2].plot(f2, np.log(spar[2][f'{n[2]}']), linestyle='-', color='blue')
axs[2].fill_between(f2, ci_py2.u05, ci_py2.u95, color='red', alpha=0.2, linewidth=0.0)
axs[2].plot(f2, ci_py2.med, linestyle='-', color='red')

for ax in axs:
    axs.set_xlabel('Frequency')
    axs.set_ylabel('log PSD')

axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
plt.tight_layout()
plt.savefig('psd.png', dpi=300, bbox_inches='tight')