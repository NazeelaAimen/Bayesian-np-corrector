#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:44:32 2024

@author: naim769
"""
import numpy as np
from core import lpost, tot_psd, prior_sum,lamb_lprior,phi_lprior,delta_lprior,b_lprior,glprior,psilprior,loglike


def power_law(b,g,f):
    #log PSD for power law for SGWB
    y=b+g*np.log(f)
    return y

def brokenPower(amp,gam,psi,frequencies):
    #log PSD for broken power law for SGWB
    f_ref=0.002
    f_br=0.002
    ome_gw = np.piecewise(
    frequencies,
    [frequencies <= f_br, frequencies > f_br],
    [
        lambda f: amp + gam * (np.log(f) - np.log(f_ref)),
        lambda f: amp + gam* (np.log(f_br) -np.log(f_ref)) + psi*(np.log(f) - np.log(f_br)),
    ],
)
    return ome_gw

def S_and_prisum(signal_model, b_val, g_val, psi_val, s_n, f, lam, phival, deltaval, P, k):
    if signal_model == 2:
        S = tot_psd(s_n, brokenPower(b_val, g_val, psi_val, f))
        prisum = prior_sum(
            lamb_lprior(lam, phival, P, k),
            phi_lprior(phival, deltaval),
            delta_lprior(deltaval),
            b_lprior(b_val),
            glprior(g_val),
            psilprior(psi_val)
        )
    else:
        S = tot_psd(s_n, power_law(b_val, g_val, f))
        prisum = prior_sum(
            lamb_lprior(lam, phival, P, k),
            phi_lprior(phival, deltaval),
            delta_lprior(deltaval),
            b_lprior(b_val),
            glprior(g_val)
        )
    return S, prisum
def update_signal_param(ind, b, g, psi, phi, delta, lam, P, k, s_n, A, E, T, f, signal_model):
    """
    Updates signal parameters.

    Parameters:
    ind: int
        Index
    b, g, psi: Arrays
        Values for b, g, and psi.
    phi, delta: Arrays
        Values for phi and delta.
    lam, P, k: Scalars or Arrays
        lambda vector, panelty matrix and number of knots.
    s_n: Array
        Noise PSD values.
    A, E, T: Scalars or Arrays
        A, E, and T channel.
    f: Array
        Frequency values.
    signal_model: int
        Specifies the type of signal model (2 for broken power, others for power law).
    Returns:
    Updated values of b, g, psi.
    """

    # Sample b and g from a reflective normal distribution
    b[ind] = np.random.normal(b[ind - 1], 0.1)
    g[ind] = np.random.normal(g[ind - 1], 0.1)
    
    S, prisum = S_and_prisum(signal_model, b[ind - 1], g[ind - 1], psi[ind - 1], s_n[ind,:], f, lam, phi[ind], delta[ind], P, k)
    ftheta = lpost(loglike(A=A, E=E, T=T, S=S, s_n=s_n[ind, :]), prisum)


    S, prisum = S_and_prisum(signal_model, b[ind], g[ind], psi[ind], s_n[ind,:], f, lam, phi[ind], delta[ind], P, k)
    ftheta_star = lpost(loglike(A=A, E=E, T=T, S=S, s_n=s_n[ind, :]), prisum)

    fac = min(0, ftheta_star - ftheta)
    if np.isnan(fac):
        fac = -100

    # Acceptance or rejection
    if np.log(np.random.rand()) > fac:
        b[ind] = b[ind - 1]
        g[ind] = g[ind - 1]
        if signal_model == 2:
            psi[ind] = psi[ind - 1]

    return b, g, psi


