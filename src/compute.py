#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:29:57 2024

@author: naim769
"""
import numpy as np
from core import dens,lpost,psd, tot_psd, prior_sum,lamb_lprior,phi_lprior,delta_lprior,b_lprior,glprior,psilprior,loglike1,loglike
from my_signal import brokenPower, power_law
from scipy.stats import multivariate_normal

def llike_psum_s(signal_model, lam, phi, delta, b, g, psi, f, P, k, noise, A, E, T):
    """
    This function computes the likelihood, priorsum, and PSD based on the signal model provided and updates the results at a specified index.

    Parameters:
    signal_model: int or None
        Specifies the type of signal model (None, 2, or other).
    lam, phi, delta: Arrays
        lambda vector, phi, delta.
    b, g, psi: Arrays
        SGWB signal parameters
    f: Array
        Frequency values.
    P, k: Panelty matrix and number of knots.
    pdgrm, s_n: Arrays
        periodogram and noise PSD values.
    A, E, T: Arrays
        TDI channels.
    llike: Array
        Likelihood
    ind: int
        Index.
    """
    
    if signal_model is None:
        prisum = prior_sum(lamb_lprior(lam, phi, P, k), 
                            phi_lprior(phi, delta), 
                            delta_lprior(delta))
        S = noise
        
        llike = loglike1(T, S)
        sig=np.zeros(len(noise))
    
    else:
        if signal_model == 2:
            sig = brokenPower(b, g, psi, f)
            prisum = prior_sum(lamb_lprior(lam, phi, P, k), 
                                phi_lprior(phi, delta), 
                                delta_lprior(delta),
                                b_lprior(b), glprior(g), 
                                psilprior(psi))
            
        else:
            sig = power_law(b, g, f)
            prisum = prior_sum(lamb_lprior(lam, phi, P, k), 
                                phi_lprior(phi, delta), 
                                delta_lprior(delta),
                                b_lprior(b), glprior(g))
        
        S = tot_psd(noise, sig)
        
        llike = loglike(A=A, E=E, T=T, S=S, s_n=noise)
    
    return llike, prisum, S, sig

def update_lambda(lam, sigma, accept_frac, k, signal_model, phi, delta, b, g, psi, f, P, Spar, modelnum, splines, s_s, A, E, T):
    """
    Updating lambda vector using MH
    """
    
    
    if accept_frac < 0.30:  
        sigma *= 0.90 
    elif accept_frac > 0.50:  
        sigma *= 1.10  

    accept_count = 0
    aux = np.arange(0, k)
    np.random.shuffle(aux)

    for sth in range(0, len(lam)):
        z = np.random.normal()
        u = np.log(np.random.uniform())
        pos = aux[sth]
        lam_p = lam[pos]
        lam_star = lam_p + sigma * z

       
        noise = psd(dens(lam, splines.T), Spar=Spar, modelnum=modelnum)
        llike, prisum, S, sig = llike_psum_s(signal_model, lam, phi, delta, b, g, psi, f, P, k, noise, A, E, T)
        ftheta = lpost(llike, prisum)

       
        lam[pos] = lam_star
        noise = psd(dens(lam, splines.T), Spar=Spar, modelnum=modelnum)
        llike, prisum, S, sig = llike_psum_s(signal_model, lam, phi, delta, b, g, psi, f, P, k, noise, A, E, T)
        ftheta_star = lpost(llike, prisum)

        
        fac = np.min([0, ftheta_star - ftheta])
        
        
        if u > fac:
            lam[pos] = lam_p
        else:
            accept_count += 1

    return lam, sigma, accept_count
