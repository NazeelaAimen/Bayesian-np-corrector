#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:00:01 2024

@author: naim769
"""

import numpy as np
from utils import get_pdgrm

from knot_allocation import knot_loc
from core import generate_basis_matrix, panelty_datvar, dens, generate_basis_matrix, lpost, psd, update_phi,update_delta
from compute import llike_psum_s

class InitParams:
    def __init__(self, delta, phi, b, g, psi, lam_mat, splines_mat, psd_mat, s_n, s_s, knots, splines, P, llike, logpost, a_phi, a_delta, sigma, accept_frac, f):
        self.delta = delta
        self.phi = phi
        self.b = b
        self.g = g
        self.psi = psi
        self.lam_mat = lam_mat
        self.splines_mat = splines_mat
        self.psd_mat = psd_mat
        self.s_n = s_n
        self.s_s = s_s
        self.knots = knots
        self.splines = splines
        self.P = P
        self.llike = llike
        self.logpost = logpost
        self.a_phi = a_phi
        self.a_delta = a_delta
        self.sigma = sigma
        self.accept_frac = accept_frac
        self.f = f
        
    
def init_param(n, k, Spar, modelnum, blocked, signal_model,  f,  A, E, T, degree, data_bin_edges, data_bin_weights, log_data):
    """
    Initialize parameters.
    """
    #checking for blocked data and whether given one or three channels
    pdgrm=get_pdgrm(blocked, A, T)
    #Frequency:
    if f is None:
        f= np.linspace(0,fs/2, len(pdgrm+1))[1:]
    delta = np.zeros(n)
    phi = np.zeros(n)
    llike = np.zeros(n)
    logpost = np.zeros(n)
    b=np.zeros(n)
    g=np.zeros(n)
    psi=np.zeros(n)
    lam_mat=np.zeros((n,k))
    splines_mat=np.zeros((n,len(pdgrm)))
    psd_mat=np.zeros((n,len(pdgrm)))
    s_s=np.zeros((n,len(pdgrm)))
    s_n=np.zeros((n,len(pdgrm)))
    
    # Knots over Fourier frequencies
    knots=knot_loc(pdgrm=pdgrm, Spar=Spar ,n_knots=k, degree= degree ,data_bin_edges=data_bin_edges,f=f,data_bin_weights= data_bin_weights,log_data=log_data)
    
    #making basis matrix:
    gridp=np.linspace(knots[0],knots[-1],len(pdgrm))
    splines=generate_basis_matrix(knots,gridp,3)
   
    #Panelty matrix
    # data variations:
    P=panelty_datvar(d=1,knots=knots,degree=degree)
    
    # Initial values: Alternative is optimization of initial weights
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
    llike[0], prisum, psd_mat[0,:], s_s[0,:]=llike_psum_s(signal_model, lam, phi[0], delta[0], b[0], g[0], psi[0], f, P, k, s_n[0,:], A, E, T)
    logpost[0] = lpost(llike[0],prisum)

    return InitParams(delta, phi, b, g, psi, lam_mat, splines_mat, psd_mat, s_n, s_s, knots, splines, P, llike, logpost, a_phi, a_delta, sigma, accept_frac, f)
