#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 04:51:34 2024

@author: naim769
"""

import bnpc
import numpy as np
import random
import matplotlib.pyplot as plt

if (False):
    a1 = 0.9
    a2 = -0.9
    a3 = 0.9
    a4 = -0.9
    sig = 1

    # First set
    n=228
    series = np.zeros((128, 100))
    for i in range(0,100):
            random.seed(i)
            series[:,i] = bnpc.ar4(n, a1, a2, a3, a4, sig)[100:]  # Indexing to get the last 100 values

    # Second set
    n=356
    series1 = np.zeros((256, 100))
    for i in range(0,100):
            random.seed(i+500)
            series1[:,i] = bnpc.ar4(n, a1, a2, a3, a4, sig)[100:]  # Indexing to get the last 100 values


    # Third set:
    n=612
    series2 = np.zeros((512, 100))
    for i in range(0,100):
            random.seed(i+1000)
            series2[:,i] = bnpc.ar4(n, a1, a2, a3, a4, sig)[100:]  # Indexing to get the last 100 values


if (False):
    ## estimating PSD
    a1p=0.1883
    sig2p=7.839
    k=25
    degree=3
    result=[]
    result1=[]
    result2=[]
    for i in range(0,100):
            # n=128
            y_c=series[:,i]-np.mean(series[:,i])
            pdgrm = bnpc.makepdgrm(y_c)
            f = pdgrm['f']

            #parametric model:
            spar = bnpc.s_ar1(a1p, sig2p, f)

            

            result.append(bnpc.mcmc(y_c, 10000, k, 2000, spar, modelnum=1))
            
            # n=256
            y_c=series1[:,i]-np.mean(series1[:,i])
            pdgrm = bnpc.makepdgrm(y_c)
            f = pdgrm['f']

            #parametric model:
            spar = bnpc.s_ar1(a1p, sig2p, f)
            
            result1.append(bnpc.mcmc(y_c, 10000, k, 2000, spar, modelnum=1))
            
            # n=512
            y_c=series2[:,i]-np.mean(series2[:,i])
            pdgrm = bnpc.makepdgrm(y_c)
            f = pdgrm['f']

            #parametric model:
            spar = bnpc.s_ar1(a1p, sig2p, f)

            result2.append(bnpc.mcmc(y_c, 10000, k, 2000, spar, modelnum=1))
            print(i)

if (False):
    #an instance
    i=67
    y_c=series[:,i]-np.mean(series[:,i])
    pdgrm = bnpc.makepdgrm(y_c)
    f = pdgrm['f']
    spar = bnpc.s_ar1(a1p, sig2p, f)
    truepsd=np.log(bnpc.s_ar4e(a1,a2,a3,a4,sig,f))

    y_c=series1[:,i]-np.mean(series1[:,i])
    pdgrm1 = bnpc.makepdgrm(y_c)
    f1 = pdgrm1['f']
    spar1 = bnpc.s_ar1(a1p, sig2p, f1)
    truepsd1=np.log(bnpc.s_ar4e(a1,a2,a3,a4,sig,f1))

    y_c=series2[:,i]-np.mean(series2[:,i])
    pdgrm2 = bnpc.makepdgrm(y_c)
    f2 = pdgrm2['f']
    spar2 = bnpc.s_ar1(a1p, sig2p, f2)
    truepsd2=np.log(bnpc.s_ar4e(a1,a2,a3,a4,sig,f2))

    ## IAE (Patricio's code)
    iae_values = []
    iae1_values = []
    iae2_values = []
    for i in range(len(result)):
            psd_values = np.log(result[i]['psd'])
            psd_med=np.median(psd_values,axis=0)
            
            iae = sum(abs(psd_med - truepsd)) * 0.5 / len(psd_med)
            iae_values.append(iae)
            
            psd_values = np.log(result1[i]['psd'])
            psd_med=np.median(psd_values,axis=0)
            
            iae = sum(abs(psd_med - truepsd1)) * 0.5 / len(psd_med)
            iae1_values.append(iae)
            
            psd_values = np.log(result2[i]['psd'])
            psd_med=np.median(psd_values,axis=0)
            
            iae = sum(abs(psd_med - truepsd2)) * 0.5 / len(psd_med)
            iae2_values.append(iae)
    print(np.median(iae_values))
    print(np.median(iae1_values))
    print(np.median(iae2_values))

    #IAE plot:
    plt.figure(figsize=(15, 5))  

    plt.subplot(1, 3, 1)
    plt.hist(iae_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('n=128', fontsize=16)
    plt.xlabel('IAE Values', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)


    plt.subplot(1, 3, 2)
    plt.hist(iae1_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('n=256', fontsize=16)
    plt.xlabel('IAE Values', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist(iae2_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('n=512', fontsize=16)
    plt.xlabel('IAE Values', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)

    plt.tight_layout()  
    plt.savefig('iae.png', dpi=300, bbox_inches='tight')
    plt.show()




    ## proportion of PSD in uniform CI band (Patricio's code):

    prop=[]
    prop1=[]
    prop2=[]
    for i in range(len(result)):
        S=np.log(result[i]['psd'])
        psd_median = np.median(S, axis=0)
        psd_help =np.max(S, axis=0)
        psd_mad =median_abs_deviation(S, axis=0)

        c_value = np.quantile(psd_help, 0.9)
        
        psd_u95 = psd_median + c_value * psd_mad
        psd_u05 = psd_median - c_value * psd_mad
        
        v=[]
        for x in range(len(psd_median)):
            if (truepsd[x] >= psd_u05[x]) and (truepsd[x] <= psd_u95[x]):
                v.append(1)
            else:
                v.append(0)
        
        prop.append(np.mean(v))
        
        S=np.log(result1[i]['psd'])
        psd_median = np.median(S, axis=0)

        psd_help =np.max(S, axis=0)

        psd_mad =median_abs_deviation(S, axis=0)

        c_value = np.quantile(psd_help, 0.9)

        psd_u95 = psd_median + c_value * psd_mad
        psd_u05 = psd_median - c_value * psd_mad
        v=[]
        for x in range(len(psd_median)):
            if (truepsd1[x] >= psd_u05[x]) and (truepsd1[x] <= psd_u95[x]):
                v.append(1)
            else:
                v.append(0)
        
        prop1.append(np.mean(v))
        
        S=np.log(result2[i]['psd'])
        psd_median = np.median(S, axis=0)

        psd_help =np.max(S, axis=0)

        psd_mad =median_abs_deviation(S, axis=0)

        c_value = np.quantile(psd_help, 0.9)

        psd_u95 = psd_median + c_value * psd_mad
        psd_u05 = psd_median - c_value * psd_mad
        v=[]
        for x in range(len(psd_median)):
            if (truepsd2[x] >= psd_u05[x]) and (truepsd2[x] <= psd_u95[x]):
                v.append(1)
            else:
                v.append(0)
        
        prop2.append(np.mean(v))

    print(np.median(prop))
    print(np.median(prop1))
    print(np.median(prop2))




    # PSD plot for an instance:
    #n=128
    knots=result[i]['knots']
    S=np.log(result[i]['psd'])
    psd_median = np.median(S, axis=0)

    psd_help =np.max(S, axis=0)


    psd_mad =median_abs_deviation(S, axis=0)


    c_value = np.quantile(psd_help, 0.9)


    psd_u95 = psd_median + c_value * psd_mad
    psd_u05 = psd_median - c_value * psd_mad

    # splines
    Sp=np.log(result[i]['splines_psd'])
    spline_psd_med = np.median(Sp, axis=0)

    psd_help =np.max(Sp, axis=0)

    psd_mad =median_abs_deviation(Sp, axis=0)

    c_value = np.quantile(psd_help, 0.9)

    spline_psd_q2 = spline_psd_med + c_value * psd_mad
    spline_psd_q1 = spline_psd_med - c_value * psd_mad

    #n=256
    knots1=result1[i]['knots']
    S1=np.log(result1[i]['psd'])
    psd1_median = np.median(S1, axis=0)

    psd_help =np.max(S1, axis=0)

    psd_mad =median_abs_deviation(S1, axis=0)

    c_value = np.quantile(psd_help, 0.9)

    psd1_u95 = psd1_median + c_value * psd_mad
    psd1_u05 = psd1_median - c_value * psd_mad

    #splines
    Sp1=np.log(result1[i]['splines_psd'])
    spline_psd1_med = np.median(Sp1, axis=0)

    psd_help =np.max(Sp1, axis=0)

    psd_mad =median_abs_deviation(Sp1, axis=0)

    c_value = np.quantile(psd_help, 0.9)

    spline_psd1_q2 = spline_psd1_med + c_value * psd_mad
    spline_psd1_q1 = spline_psd1_med - c_value * psd_mad
     

    #n=512
    knots2=result2[i]['knots']
    S2=np.log(result2[i]['psd'])
    psd2_median = np.median(S2, axis=0)

    psd_help =np.max(S2, axis=0)

    psd_mad =median_abs_deviation(S2, axis=0)

    c_value = np.quantile(psd_help, 0.9)

    psd2_u95 = psd2_median + c_value * psd_mad
    psd2_u05 = psd2_median - c_value * psd_mad

    #splines
    Sp2=np.log(result2[i]['splines_psd'])
    spline_psd2_med = np.median(Sp2, axis=0)

    psd_help =np.max(Sp2, axis=0)

    psd_mad =median_abs_deviation(Sp2, axis=0)

    c_value = np.quantile(psd_help, 0.9)

    spline_psd2_q2 = spline_psd2_med + c_value * psd_mad
    spline_psd2_q1 = spline_psd2_med - c_value * psd_mad

         
    # PSD plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))


    axs[0].plot(f, np.log(pdgrm['pdgrm']), linestyle='-', color='black', alpha=0.4,label='Periodogram')
    axs[0].plot(f, truepsd, linestyle='--', color='black', label='True')
    axs[0].plot(f, spline_psd_med, linestyle='-', color='purple', label='Splines')
    axs[0].fill_between(f, spline_psd_q1, spline_psd_q2, color='purple', alpha=0.2, linewidth=0.0)
    axs[0].plot(f, np.log(spar), linestyle='-', color='blue', label='AR1')
    axs[0].fill_between(f, psd_u05, psd_u95, color='red', alpha=0.5, linewidth=0.0)
    axs[0].plot(f, psd_median, linestyle='-', color='red', label='Estimated')
    axs[0].vlines(knots, np.log(1e-3), np.log(5e-3), color='green', alpha=0.5, label='knots')


    axs[1].plot(f1, np.log(pdgrm1['pdgrm']), linestyle='-', color='black', alpha=0.4)
    axs[1].plot(f1, truepsd1, linestyle='--', color='black')
    axs[1].plot(f1, spline_psd1_med, linestyle='-', color='purple')
    axs[1].fill_between(f1, spline_psd1_q1, spline_psd1_q2, color='purple', alpha=0.2, linewidth=0.0)
    axs[1].plot(f1, np.log(spar1), linestyle='-', color='blue')
    axs[1].fill_between(f1, psd1_u05, psd1_u95, color='red', alpha=0.5, linewidth=0.0)
    axs[1].plot(f1, psd1_median, linestyle='-', color='red')
    axs[1].vlines(knots1, np.log(1e-3), np.log(5e-3), color='green', alpha=0.5, label='knots')

    axs[2].plot(f2, np.log(pdgrm2['pdgrm']), linestyle='-', color='black', alpha=0.4)
    axs[2].plot(f2, truepsd2, linestyle='--', color='black')
    axs[2].plot(f2, spline_psd2_med, linestyle='-', color='purple')
    axs[2].fill_between(f2, spline_psd2_q1, spline_psd2_q2, color='purple', alpha=0.2, linewidth=0.0)
    axs[2].plot(f2, np.log(spar2), linestyle='-', color='blue')
    axs[2].fill_between(f2, psd2_u05, psd2_u95, color='red', alpha=0.5, linewidth=0.0)
    axs[2].plot(f2, psd2_median, linestyle='-', color='red')
    axs[2].vlines(knots2, np.log(1e-3), np.log(5e-3), color='green', alpha=0.5, label='knots')

    for ax in axs:
        ax.set_xlabel('Frequency')
        ax.set_ylabel('log PSD')

    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.savefig('PSD_test.png', dpi=300, bbox_inches='tight')
    plt.show()     

    
