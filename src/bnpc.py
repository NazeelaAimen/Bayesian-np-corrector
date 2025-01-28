#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nazeela
"""

import numpy as np
from compute import MCMCloop
from init import init_param


class MCMCParams:
    def __init__(self, T, n, k, burnin, A=None, E=None, Spar=1, Spar_xy=None, degree=3, modelnum=1, f=None, fs=None, blocked=False, signal_model=None, data_bin_edges=None, data_bin_weights=None, log_data=True):
        # Store MCMC input parameters
        self.T = T
        self.n = n
        self.k = k
        self.burnin = burnin
        self.A = A
        self.E = E
        self.Spar = Spar
        self.Spar_xy = Spar_xy
        self.degree = degree
        self.modelnum = modelnum
        self.f = f
        self.fs = fs
        self.blocked = blocked
        self.signal_model = signal_model
        self.data_bin_edges = data_bin_edges
        self.data_bin_weights = data_bin_weights
        self.log_data = log_data
        init_param(self)

             
class MCMCResult:
    def __init__(self, params):
        cond=slice(params.burnin,params.n)
        self.phi = params.phi[cond]
        self.delta = params.delta[cond]
        self.phi_xy = params.phi_xy[cond]
        self.delta_xy = params.delta_xy[cond]
        self.loglike = params.llike[cond]
        self.logpost = params.logpost[cond]
        self.lambda_matrix = params.lam_mat[cond, :]
        self.lambda_matrix_xy = params.lam_mat_xy[cond, :]
        self.knots = params.knots
        self.knots_xy = params.knots_xy
        self.Nx_psd = params.Nx_mat[cond, :]
        self.Nxy_psd = params.Nxy_mat[cond, :]
        self.noise_psd = params.s_n[cond, :]
        self.noise_psd_A = params.s_nA[cond, :]
        self.sig_psd = params.s_s[cond, :]
        self.tot_psd = params.psd_mat[cond, :]
        self.b = params.b[cond]
        self.g = params.g[cond]
        self.psi = params.psi[cond]
        self.A = params.A
        self.E = params.E
        self.T = params.T
        self.f = params.f
    def summary(self):
        """
        Summary of the MCMC results.
        """
        return 
    
def mcmc(T: np.ndarray, n: int, 
         k: int, burnin: int,
         A=None, E=None, Spar=1, 
         Spar_xy=None, degree=3, 
         modelnum=1, f=None, 
         fs=None, blocked=False,
         signal_model=1, 
         data_bin_edges=None, 
         data_bin_weights=None, 
         log_data=True):
    params=MCMCParams(T, n, k, burnin, A, E, Spar, Spar_xy, degree, modelnum, f, fs, blocked, signal_model, data_bin_edges, data_bin_weights, log_data)
    MCMCloop(params)            
    result = MCMCResult(params=params)

    return result
