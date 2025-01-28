#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:29:57 2024

@author: naim769
"""
import numpy as np
from core import noises,dens,lpost,psd, tot_psd, prior_sum,lamb_lprior,phi_lprior,delta_lprior,b_lprior,glprior,psilprior,loglike1,loglike
from my_signal import brokenPower, power_law, response, psd_sgwb, update_signal_param
from scipy.stats import multivariate_normal, gamma

    
def llike_psum_s(self, i, lam_xy_loop=False):
        """
        This function computes the likelihood, priorsum, and PSD and updates the results at a specified index.
        
        i: int
            Index.
        """
        j=i
        if lam_xy_loop:
            j=i+1
        _,_, noise, noise_A= noises(self.lam_mat[j, :], self.lam_mat_xy[i, :], self.splines, self.splines_xy, self.Spar, self.Spar_xy, self.modelnum, self.f)
        
        if self.signal_model == 2: # broken power law
            sig=psd_sgwb(omega=brokenPower(self.b[i], self.g[i], self.psi[i], self.f), response=response(self.f), f=self.f)
            prisum = prior_sum(lamb_lpri = lamb_lprior(self.lam_mat[j, :], self.phi[i], self.P, self.k), 
                                phi_lpri = phi_lprior(self.phi[i], self.delta[i]), 
                                delta_lpri = delta_lprior(self.delta[i]),
                                lamb_lpri_xy = lamb_lprior(self.lam_mat_xy[j, :], self.phi_xy[i], self.P_xy, self.k), 
                                phi_lpri_xy = phi_lprior(self.phi_xy[i], self.delta_xy[i]), 
                                delta_lpri_xy = delta_lprior(self.delta_xy[i]),
                                b_lpri = b_lprior(self.b[i]), 
                                g_lpri = glprior(self.g[i]), 
                                psi_lprior = psilprior(self.psi[i]))
            
        else: # power law
            sig=psd_sgwb(omega=power_law(self.b[i], self.g[i], self.f), response=response(self.f), f=self.f)
            prisum = prior_sum(lamb_lpri = lamb_lprior(self.lam_mat[j, :], self.phi[i], self.P, self.k), 
                                phi_lpri = phi_lprior(self.phi[i], self.delta[i]), 
                                delta_lpri = delta_lprior(self.delta[i]),
                                lamb_lpri_xy = lamb_lprior(self.lam_mat_xy[j, :], self.phi_xy[i], self.P_xy, self.k), 
                                phi_lpri_xy = phi_lprior(self.phi_xy[i], self.delta_xy[i]), 
                                delta_lpri_xy = delta_lprior(self.delta_xy[i]),
                                b_lpri = b_lprior(self.b[i]), 
                                g_lpri = glprior(self.g[i])) 
                               
            
        S = tot_psd(noise_A, sig)
        
        llike = loglike(A=self.A, E=self.E, T=self.T, S=S, s_n=noise)
    
        return llike, prisum, S, sig


def sigmaupdate(accept_frac,sigma):
    if accept_frac < 0.30:  
        sigma *= 0.90 
    elif accept_frac > 0.50:  
        sigma *= 1.10  
    return sigma
        

    
def posterior_cal(self,i,lam_xy_loop):
    llike, prisum, _, _ = llike_psum_s(self, i, lam_xy_loop)
    ftheta = lpost(llike, prisum)
    return ftheta

def lam_loop(self, i, lam_xy_loop=False):
    accept_count = 0
    aux = np.arange(0, self.k)
    np.random.shuffle(aux)
    for sth in range(0, len(self.lam_mat[i, :])):
        z = np.random.normal()
        u = np.log(np.random.uniform())
        pos = aux[sth]
        
        lam_p = self.lam_mat_xy[i, pos] if lam_xy_loop else self.lam_mat[i, pos]
        lam_star = lam_p + self.sigma_xy * z if lam_xy_loop else lam_p + self.sigma * z
        
        ftheta = posterior_cal(self, i, lam_xy_loop)

        if lam_xy_loop:
            self.lam_mat_xy[i, pos] = lam_star
        else:
            self.lam_mat[i, pos] = lam_star

        ftheta_star = posterior_cal(self, i, lam_xy_loop)
        
        fac = np.min([0, ftheta_star - ftheta])
        
        if u > fac:
            # Reject update
            if lam_xy_loop:
                self.lam_mat_xy[i, pos] = lam_p
            else:
                self.lam_mat[i, pos] = lam_p
        else:
            accept_count += 1

    accept_frac = accept_count / self.k
    return self.lam_mat[i,:], self.lam_mat_xy[i,:], accept_frac
 
    
def update_lambda(self, i):
    """
    Updates lambda vector for T and A channel using Metropolis-Hastings (MH) with current parameter values.
    """
    self.sigma = sigmaupdate(self.accept_frac, self.sigma)
    self.lam_mat[i,:],_,self.accept_frac = lam_loop(self,i-1)
    self.sigma_xy = sigmaupdate(self.accept_frac_xy, self.sigma_xy)
    _,self.lam_mat_xy[i,:],self.accept_frac_xy = lam_loop(self, i-1, True)
        
    
def update_phi(lam, P, delta, a_phi):
    """
    conditional posterior distribution of phi.
    
    Parameters:
    lam (array-like): Lambda vector.
    P (array-like): Panelty matrix.
    delta (float): delta.
    a_phi (float): Shape parameter for the gamma distribution.
    
    Returns:
    float: Updated phi value.
    """
    b_phi = 0.5 * np.matmul(np.transpose(lam), np.matmul(P, lam)) + delta
    phi_value = gamma.rvs(a=a_phi, scale=1 / b_phi, size=1)
    return phi_value

def update_delta(phi, a_delta):
    """
    conditional posterior distribution of delta.
    
    Parameters:
    phi (float): phi.
    a_delta (float): Shape parameter for the gamma distribution.
    
    Returns:
    float: Updated delta value.
    """
    b_delta = phi + 1e-4
    delta_value = gamma.rvs(a=a_delta, scale=1 / b_delta, size=1)
    return delta_value

def update_phi_delta(lam, P, delta, a_phi, a_delta):
        # Sampling phi,,,
        phi = update_phi(lam, P, delta, a_phi)

        # Sampling delta
        delta = update_delta(phi, a_delta)
        return phi,delta

def MCMCloop(self):
    for i in range(1, self.n):
        # Updating lambda
        update_lambda(self,i)
        self.Nx_mat[i,:],self.Nxy_mat[i,:], self.s_n[i, :], self.s_nA[i,:]= noises(self.lam_mat[i, :], self.lam_mat_xy[i, :], self.splines, self.splines_xy, self.Spar, self.Spar_xy, self.modelnum, self.f)

        # Updating phi, delta for Nx
        self.phi[i], self.delta[i]= update_phi_delta(self.lam_mat[i, :], self.P, self.delta[i-1], self.a_phi, self.a_delta)

        # Updating phi, delta for Nxy
        self.phi_xy[i], self.delta_xy[i]= update_phi_delta(self.lam_mat_xy[i, :], self.P_xy, self.delta_xy[i-1], self.a_phi, self.a_delta)
        
        # MH step for signal parameters
        update_signal_param(self, i)
        
        self.llike[i], prisum, self.psd_mat[i, :], self.s_s[i, :] = llike_psum_s(self, i=i)
        # Update likelihood and posterior
        self.logpost[i] = lpost(self.llike[i], prisum)