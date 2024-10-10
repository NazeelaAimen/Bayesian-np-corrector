#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nazeela
"""

import numpy as np
from knot_allocation import knot_loc
from utils import get_pdgrm
from core import generate_basis_matrix, panelty_datvar, dens, generate_basis_matrix, lpost, psd, update_phi,update_delta
from compute import llike_psum_s, update_lambda
from my_signal import update_signal_param
def mcmc(T: np.ndarray,n: int,
         k: int, burnin: int,
         A=None, E=None, 
         Spar=1,degree=3,
         modelnum=1,f=None,
         fs=None,blocked=False,
         signal_model=None,
         data_bin_edges=None,
         data_bin_weights=None,
         log_data=False):
    delta = np.zeros(n)
    phi = np.zeros(n)
    llike = np.zeros(n)
    logpost = np.zeros(n)
    b=np.zeros(n)
    g=np.zeros(n)
    psi=np.zeros(n)
    
    #checking for blocked data and whether given one or three channels
    pdgrm=get_pdgrm(blocked, A, T)
    
    #checking if A and signal is provided:
    if A is not None and signal_model is None:
        signal_model=1
    
    if f is None:
        f= np.linspace(0,fs/2, len(pdgrm+1))[1:]
    lam_mat=np.zeros((n,k))
    splines_mat=np.zeros((n,len(pdgrm)))
    psd_mat=np.zeros((n,len(pdgrm)))
    s_s=np.zeros((n,len(pdgrm)))
    s_n=np.zeros((n,len(pdgrm)))
    
    # Knots over Fourier frequencies
    K = k - degree + 1
    #Equidistant
    data=Spar-pdgrm #difference between periodogram and parametric model
    data=data-min(data)+1e-9 #translating so that it is positive and does not loose any variation
    if log_data:
        data=np.log(data)
    knots=knot_loc(data=data,n_knots=K,data_bin_edges=data_bin_edges,f=f,data_bin_weights= data_bin_weights,log_data=log_data)
    m = max(f) - min(f)
    c = min(f)
    knots = m * knots + c #Linear translation from [0,1] to fourier frequencies range

    #making basis matrix:
    gridp=np.linspace(knots[0],knots[-1],len(pdgrm))
    splines=generate_basis_matrix(knots,gridp,3)
   
    #Panelty matrix
    # data variations:
    P=panelty_datvar(d=1,knots=knots,degree=degree)
    
    # Initial values
    lam = pdgrm / np.sum(pdgrm)
    lam = lam[np.round(np.linspace(0, len(lam) - 1, k)).astype(int)]
    lam[lam == 0] = 1e-50
    lam_mat[0,:]=lam

    delta[0] = 1
    phi[0] = 1
    b[0]=1
    g[0]=1        
    a_phi = k / 2 + 1
    a_delta = 1 + 1e-4
    
    sigma= 1
    accept_frac=0.4
     
    #splines:
    splines_mat[0,:]=dens(lam,splines.T)
    s_n[0,:]=psd(splines_mat[0,:], Spar=Spar,  modelnum=modelnum)
    llike[0], prisum, psd_mat[0,:], s_s[0,:]=llike_psum_s(signal_model, lam, phi, delta, b, g, psi, f, P, k, s_n[0,:], A, E, T, 0)
    logpost[0] = lpost(llike[0],prisum)
    for i in range(1,n):
        # updating lambda
        lam, sigma, accept_count=update_lambda(lam, sigma, accept_frac, k, signal_model, phi, delta, b, g, psi, f, P, Spar, modelnum, splines, s_s, A, E, T, i)
        accept_frac = accept_count / k
        splines_mat[i,:]=dens(lam,splines.T)
        lam_mat[i,:]=lam
        s_n[i,:]=psd(splines_mat[i,:], Spar=Spar,  modelnum=modelnum)

        # Sampling phi
        phi[i] = update_phi(lam, P, delta[i-1], a_phi)

        # Sampling delta
        delta[i] = update_delta(phi[i], a_delta)
        
        #MH step for signal parameters
        if signal_model is not None:    
            b,g,psi=update_signal_param(i, b, g, psi, phi, delta, lam, P, k, s_n, A, E, T, f, signal_model) 
        
        llike[i], prisum, psd_mat[i,:], s_s[i,:]=llike_psum_s(signal_model, lam, phi, delta, b, g, psi, f, P, k, s_n[i,:], A, E, T, i)
        logpost[i] = lpost(llike[i], prisum)
        
        
        
    phi = phi[burnin:n]
    delta = delta[burnin:n]
    llike = llike[burnin:n]
    logpost = logpost[burnin:n]
    lam_mat=lam_mat[burnin:n,:]
    psd_mat=psd_mat[burnin:n,:]
    splines_mat=splines_mat[burnin:n,:]
    s_n=s_n[burnin:n,:]
    s_s=s_s[burnin:n,:]
    b=b[burnin:n]
    g=g[burnin:n]
    psi=psi[burnin:n]
    
    result = {'phi': phi,
                  'delta': delta,
                  'llike': llike,
                  'logpost': logpost,
                  'lambda': lam_mat,
                  'knots': knots,
                  'splines_psd':splines_mat,
                  'noise_psd':s_n,
                  'sig_psd':s_s,
                  'tot_psd':psd_mat,
                  'b':b,
                  'g':g,
                  'psi':psi,
                  'A':A,
                  'E':E,
                  'T':T,
                  'f':f}    

    return result
