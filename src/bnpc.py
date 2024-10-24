#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nazeela
"""

import numpy as np
from core import dens, lpost, psd, update_phi,update_delta
from compute import llike_psum_s, update_lambda
from my_signal import update_signal_param
from init import init_param


class MCMCResult:
    def __init__(self, phi, delta, llike, logpost, lambda_matrix, knots, splines_psd, noise_psd, sig_psd, tot_psd, b, g, psi, A, E, T, f):
        self.phi = phi
        self.delta = delta
        self.llike = llike
        self.logpost = logpost
        self.lambda_matrix = lambda_matrix
        self.knots = knots
        self.splines_psd = splines_psd
        self.noise_psd = noise_psd
        self.sig_psd = sig_psd
        self.tot_psd = tot_psd
        self.b = b
        self.g = g
        self.psi = psi
        self.A = A
        self.E = E
        self.T = T
        self.f = f
    
    def summary(self):
        """
        Summary of the MCMC results.
        """
        return 
    

def mcmc(T: np.ndarray, n: int, k: int, burnin: int, A=None, E=None, Spar=1, degree=3, modelnum=1, f=None, fs=None, blocked=False, signal_model=None, data_bin_edges=None, data_bin_weights=None, log_data=False):
    
    # Checking if A and signal is provided
    if A is not None and signal_model is None:
        signal_model = 1
    
    params = init_param(n=n, k=k, Spar=Spar, modelnum=modelnum, blocked=blocked, signal_model=signal_model, f=f, A=A, E=E, T=T, degree=degree, data_bin_edges=data_bin_edges, data_bin_weights=data_bin_weights, log_data=log_data)
    
    for i in range(1, n):
        # Updating lambda
        params.lam_mat[i, :], params.sigma, accept_count = update_lambda(params.lam_mat[i-1, :], params.sigma, params.accept_frac, k, signal_model, params.phi[i-1], params.delta[i-1], params.b[i-1], params.g[i-1], params.psi[i-1], params.f, params.P, Spar, modelnum, params.splines, params.s_s, A, E, T)
        
        params.accept_frac = accept_count / k
        params.splines_mat[i, :] = dens(params.lam_mat[i, :], params.splines.T)
        params.s_n[i, :] = psd(params.splines_mat[i, :], Spar=Spar, modelnum=modelnum)

        # Sampling phi
        params.phi[i] = update_phi(params.lam_mat[i, :], params.P, params.delta[i-1], params.a_phi)

        # Sampling delta
        params.delta[i] = update_delta(params.phi[i], params.a_delta)
        
        # MH step for signal parameters
        if signal_model is not None:    
            params.b, params.g, params.psi = update_signal_param(i, params.b, params.g, params.psi, params.phi, params.delta, params.lam_mat[i, :], params.P, k, params.s_n, A, E, T, params.f, signal_model) 
        
        # Update likelihood and posterior
        params.llike[i], prisum, params.psd_mat[i, :], params.s_s[i, :] = llike_psum_s(signal_model, params.lam_mat[i, :], params.phi[i], params.delta[i], params.b[i], params.g[i], params.psi[i], params.f, params.P, k, params.s_n[i, :], A, E, T)
        params.logpost[i] = lpost(params.llike[i], prisum)
                
    result = MCMCResult(
        phi=params.phi[burnin:n],
        delta=params.delta[burnin:n],
        loglike=params.llike[burnin:n],
        logpost=params.logpost[burnin:n],
        lambda_matrix=params.lam_mat[burnin:n, :],
        knots=params.knots,
        splines_psd=params.splines_mat[burnin:n, :],
        noise_psd=params.s_n[burnin:n, :],
        sig_psd=params.s_s[burnin:n, :],
        tot_psd=params.psd_mat[burnin:n, :],
        b=params.b[burnin:n],
        g=params.g[burnin:n],
        psi=params.psi[burnin:n],
        A=A,
        E=E,
        T=T,
        f=params.f
    )

    return result

