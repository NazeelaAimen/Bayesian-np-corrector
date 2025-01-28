#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:44:32 2024

@author: naim769
"""
import numpy as np
from core import lpost, tot_psd, prior_sum,lamb_lprior, phi_lprior,delta_lprior,b_lprior,glprior,psilprior,loglike

#LISA response to GW
def response(f):
    f_ref = 299792458  / (2 * np.pi * 2.5e9)
    W_sq = (abs(1-np.exp(-2j*f/f_ref)))**2
    R_A=9*W_sq*(1/(1+f/(4*f_ref/3)))/20
    return R_A

def psd_sgwb(omega,response,f):
    h0=2.7e-18
    fact=np.log((3*(h0**2)*response)/(4*(np.pi**2)*f**3))
    return omega+fact

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


def S_and_prisum(self, b_val, g_val, psi_val, ind):
    res = response(self.f)
    if self.signal_model == 2:
        signal = psd_sgwb(omega=brokenPower(b_val, g_val, psi_val, self.f), response=res, f=self.f)
        S = tot_psd(self.s_nA[ind, :], signal)
        prisum = prior_sum(
            lamb_lprior(self.lam_mat[ind, :], self.phi[ind], self.P, self.k),
            phi_lprior(self.phi[ind], self.delta[ind]),
            delta_lprior(self.delta[ind]),
            lamb_A_lprior(self.lam_mat_A[ind, :], min(np.median(self.lam_mat[:ind+1],axis=0)), max(np.median(self.lam_mat[:ind+1],axis=0))),
            b_lprior(b_val),
            glprior(g_val),
            psilprior(psi_val)
        )
    else:
        signal = psd_sgwb(omega=power_law(b_val, g_val, self.f), response=res, f=self.f)
        S = tot_psd(self.s_nA[ind, :], signal)
        prisum = prior_sum(lamb_lpri = lamb_lprior(self.lam_mat[ind, :], self.phi[ind], self.P, self.k),
                           phi_lpri = phi_lprior(self.phi[ind], self.delta[ind]),
                           delta_lpri = delta_lprior(self.delta[ind]),
                           # lamb_lpri_A = lamb_lprior(self.lam_mat[ind, :], self.phi[ind], self.P, self.k),
                           lamb_lpri_xy = lamb_lprior(self.lam_mat_xy[ind, :], self.phi_xy[ind], self.P_xy, self.k),
                           phi_lpri_xy = phi_lprior(self.phi_xy[ind], self.delta_xy[ind]),
                           delta_lpri_xy = delta_lprior(self.delta_xy[ind]),
                           b_lpri = b_lprior(b_val),
                           g_lpri = glprior(g_val)
        )
    return S, prisum
def update_signal_param(self, ind):
    """
    Updates signal parameters using the current state in self.

    Parameters:
    ind: int
        Index for updating the parameters.

    Returns:
    Updated values of b, g, and psi (if applicable).
    """

    
    if self.signal_model == 2:
        self.psi[ind] = np.random.normal(self.psi[ind - 1], 1)
    else:
        # Sample `b` and `g` from a reflective normal distribution
        self.b[ind] = np.random.normal(self.b[ind - 1], 0.05)
        self.g[ind] = np.random.normal(self.g[ind - 1], 0.05)
        
    S, prisum = S_and_prisum(self, self.b[ind - 1], self.g[ind - 1], self.psi[ind - 1], ind)
    ftheta = lpost(loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=self.s_n[ind, :]), prisum)

    # Calculate `S` and `prisum` for the current proposed values of `b`, `g`, and `psi`
    S, prisum = S_and_prisum(self, self.b[ind], self.g[ind], self.psi[ind], ind)
    ftheta_star = lpost(loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=self.s_n[ind, :]), prisum)

    # Acceptance or rejection
    fac = min(0, ftheta_star - ftheta)
    if np.isnan(fac):
        fac = -100

    if np.log(np.random.rand()) > fac:
        # Reject the proposal; revert `b`, `g`, and possibly `psi`
        self.b[ind] = self.b[ind - 1]
        self.g[ind] = self.g[ind - 1]
        if self.signal_model == 2:
            self.psi[ind] = self.psi[ind - 1]