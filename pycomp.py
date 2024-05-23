#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:54:43 2024

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

# def uniformmax():
    

class MCMCdata:
    def __init__(self):
        self.fz = None
        self.v = None
        self.dblist = None
        self.psds = None
        self.psd_quantiles = None
        self.lnl = None
        self.samples = None

    @classmethod
    def from_r(cls, mcmc):
        obj = cls()
        obj.fz = None
        obj.v = mcmc["V"]
        obj.dblist = mcmc["db.list"]
        obj.psd = mcmc["fpsd.sample"]
        obj.psd_quantiles = np.array(
            [
                np.array(mcmc["psd.median"]),
                np.array(mcmc["psd.p05"]),
                np.array(mcmc["psd.p95"]),
            ]
        )
        obj.lnl = mcmc["ll.trace"]
        obj.samples = np.array([mcmc["phi"], mcmc["delta"], mcmc["tau"]]).T
        return obj
# Load the R script containing the function
random_seed = 0  
robjects.r['set.seed'](random_seed)

robjects.r['source']('psd_arma.R')

# Access the R function
psd_arma = robjects.globalenv['psd_arma']

n=512
freq =np.pi*( 2 * (np.arange(1, n // 2 + 1) - 1) / n)[1:]
arex = np.array([0.9, -0.9, 0.9, -0.9])

sigma2_ex = 1
f_r = robjects.FloatVector(freq)
arex_r = robjects.FloatVector(arex)
ma_ex = robjects.NA_Integer

truepsd=np.log(psd_arma(f_r,arex_r,ma_ex,sigma2_ex))

y = robjects.r['arima.sim'](robjects.r['list'](ar=arex_r), n=n)
y_c = y - np.mean(y)
scale=np.std(y)
y_c=y_c/scale
#pdgrm = bnpc.makepdgrm_pi(y_c)
#f=pdgrm['f']
#parametric model:
a1p=0.1883
sig2p=7.839
ar1psd=psd_arma(f_r,a1p,ma_ex,sig2p)
mcmcr=importr("psplinePsd")

k=25
degree=3


np_cv_rules = default_converter + numpy2ri.converter
with np_cv_rules.context():
        mcmc_r_res = mcmcr.gibbs_pspline(
            y_c,
            burnin=5000,
            Ntotal=15000,
            degree=3,
            eqSpacedKnots=False,
            k=k,
        )
    
robjects.r['source']('knotloc.R')
knotloc=robjects.globalenv['knotloc']
with np_cv_rules.context():
 knots=knotloc(y_c,k,degree,False)*np.pi

pdgrm=mcmc_r_res['pdgrm'][1:-1]
resultpy=bnpc.mcmc(pdgrm, 15000, k, 5000,freq, spar=ar1psd, modelnum=1)        

S = resultpy['psd']+2*np.log(scale)
psd_med=np.median(S,axis=0)

iae = sum(abs(psd_med - truepsd)) * np.pi / len(psd_med)

S1 = np.log(mcmc_r_res['fpsd.sample']*scale**2)
psd_med1=np.median(S1,axis=1)[1:-1]
iae1 = sum(abs(psd_med1 - truepsd)) * np.pi / len(psd_med)

pdgrm=pdgrm*scale**2


# CI:
# py
psd_help =np.apply_along_axis(bnpc.uniformmax, 0, S)
psd_mad =median_abs_deviation(S, axis=0)
c_value = np.quantile(psd_help, 0.9)
psd_u95 = psd_med + c_value * psd_mad
psd_u05 = psd_med - c_value * psd_mad

# R:

psd_help =np.apply_along_axis(bnpc.uniformmax, 1, S1)
psd_mad =median_abs_deviation(S1, axis=1)[1:-1]
c_value = np.quantile(psd_help, 0.9)
psd1_u95 = psd_med1 + c_value * psd_mad
psd1_u05 = psd_med1 - c_value * psd_mad

plt.plot(freq, np.log(pdgrm), linestyle='-', color='black', alpha=0.5, label='Periodogram')
plt.plot(freq,truepsd,linestyle='--',color='black',label='truepsd')
plt.plot(freq,psd_med,color='red',label='pypsd')
plt.fill_between(freq, psd_u05, psd_u95, color='red', alpha=0.5, linewidth=0.0)

plt.plot(freq,psd_med1,color='purple',label='rpsd')
plt.fill_between(freq, psd1_u05, psd1_u95, color='purple', alpha=0.5, linewidth=0.0)
plt.legend()
plt.savefig('PSD_comp.png', dpi=300, bbox_inches='tight')


plt.plot(freq, np.log(pdgrm), linestyle='-', color='black', alpha=0.5, label='Periodogram')
#plt.plot(freq,np.log(spar),color='blue',label='AR1')
plt.plot(freq,truepsd,linestyle='--',color='black',label='truepsd')
plt.plot(freq,psd_med,color='red',label='pypsd')
plt.fill_between(freq, psd_u05, psd_u95, color='red', alpha=0.5, linewidth=0.0)
plt.legend()
plt.savefig('PSD_py.png', dpi=300, bbox_inches='tight')


plt.plot(freq, np.log(pdgrm), linestyle='-', color='black', alpha=0.5, label='Periodogram')
plt.plot(freq,truepsd,linestyle='--',color='black',label='truepsd')
plt.plot(freq,psd_med1,color='purple',label='rpsd')
plt.fill_between(freq, psd1_u05, psd1_u95, color='purple', alpha=0.5, linewidth=0.0)
plt.legend()
plt.savefig('PSD_comp.png', dpi=300, bbox_inches='tight')



robjects.r['source']('uniform.R')

# Access the R function
unifmax = robjects.globalenv['uniformmax']
apply=robjects.r['apply']
mad=robjects.r['mad']
with np_cv_rules.context():
 psd_help= apply(S1, 1, unifmax)
 psd_mad=apply(S1, 1, mad)[1:-1]
c_value = np.quantile(psd_help, 0.9)
psd1_u95 = psd_med1 + c_value * psd_mad[1:-1]
psd1_u05 = psd_med1 - c_value * psd_mad[1:-1]

#assert
psd_help =np.apply_along_axis(bnpc.uniformmax, 1, S1)
psd_mad =np.apply_along_axis(bnpc.mad, 1, S1)
c_value = np.quantile(psd_help, 0.9)


#plt.fill_between(f, mcmc_r_res['psd_u05']*scale**2,mcmc_r_res['psd_u95']*scale**2, color='purple', alpha=0.2, linewidth=0.0)

# np.savetxt(f"data_{random_seed}.txt",y)
# np.savetxt(f"errors_{random_seed}.txt",[iae,iae1])
# with open('resultpy.pkl', 'wb') as f:
#     pickle.dump(resultpy, f)

# with open('resultR.pkl', 'wb') as f:
#     pickle.dump(mcmc_r_res, f)

# robjects.r['source']('CI.R')

# # Access the R function
# ci = robjects.globalenv['ci']
# with np_cv_rules.context():
#  cis=ci(np.transpose(S))
