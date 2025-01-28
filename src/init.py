#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:00:01 2024

@author: naim769
"""

import numpy as np
from utils import get_pdgrm

from knot_allocation import knot_loc
from core import noises,generate_basis_matrix, panelty_datvar, dens, generate_basis_matrix, lpost, psd
from compute import llike_psum_s, update_phi, update_delta
from init_weights import optimize_starting_weights

def init_param(self):
    """
    Initialize parameters directly on `self`.
    Assumes `self` has all required attributes from `MCMCParams.__init__`.
    """
    # Checking for blocked data and whether given one or three channels
    pdgrm = get_pdgrm(self.blocked, self.A, self.T)
    
    # Frequency setup
    if self.f is None:
        self.f = np.linspace(0, self.fs / 2, len(pdgrm + 1))[1:]
    
    # Initialize MCMC variables
    n, k = self.n, self.k
    self.delta = np.zeros(n)
    self.phi = np.zeros(n)
    self.delta_xy = np.zeros(n)
    self.phi_xy = np.zeros(n)
    self.llike = np.zeros(n)
    self.logpost = np.zeros(n)
    self.b = np.zeros(n)
    self.g = np.zeros(n)
    self.psi = np.zeros(n)
    self.lam_mat = np.zeros((n, k))
    self.lam_mat_xy = np.zeros((n, k))
    self.Nx_mat = np.zeros((n, len(pdgrm)))
    self.Nxy_mat = np.zeros((n, len(pdgrm)))
    self.psd_mat = np.zeros((n, len(pdgrm)))
    self.s_s = np.zeros((n, len(pdgrm)))
    self.s_n = np.zeros((n, len(pdgrm)))
    self.s_nA = np.zeros((n, len(pdgrm)))
    
    # Knots over Fourier frequencies
    self.knots = knot_loc(pdgrm=pdgrm, Spar=self.Spar, n_knots=k, degree=self.degree,
                          data_bin_edges=self.data_bin_edges, f=self.f, data_bin_weights=self.data_bin_weights,
                          log_data=self.log_data)
    
    # Basis matrix
    gridp = np.linspace(self.knots[0], self.knots[-1], len(pdgrm))
    self.splines = generate_basis_matrix(self.knots, gridp, 3)
    self.n_gridpoints, self.n_basis = self.splines.shape
    
    # Penalty matrix for data variations
    self.P = panelty_datvar(d=1, knots=self.knots, degree=self.degree)
    
    # Initial lambda values
    
    lam = pdgrm / np.sum(pdgrm)
    lam = lam[np.round(np.linspace(0, len(lam) - 1, k)).astype(int)]
    lam[lam == 0] = 1e-50
    # lam= optimize_starting_weights(
    #     splines= self.splines.T, 
    #     Spar= self.Spar, 
    #     modelnum= self.modelnum,
    #     n_basis= self.n_basis,
    #     data= self.T,
    #     init_x= lam,
    #     n_optimization_steps=100,
    #     bounds=None,
    # )
    self.lam_mat[0, :] = lam
    
    # Initial MCMC parameter values
    self.delta[0] = 1
    self.phi[0] = 1
    self.b[0] = np.log(1e28)
    self.g[0] = -2/3        
    self.a_phi = k / 2 + 1
    self.a_delta = 1 + 1e-4
    self.sigma = 1
    self.accept_frac = 0.4

    if self.Spar_xy is None:
            self.Spar_xy = self.Spar
    self.knots_xy = knot_loc(pdgrm=pdgrm, Spar=self.Spar, n_knots=k, degree=self.degree,
                                data_bin_edges=self.data_bin_edges, f=self.f, data_bin_weights=self.data_bin_weights,
                                log_data=self.log_data)
        
    gridp_A = np.linspace(self.knots_xy[0], self.knots_xy[-1], len(pdgrm))
    self.splines_xy = generate_basis_matrix(self.knots_xy, gridp_A, 3)
        
    self.P_xy = panelty_datvar(d=1, knots=self.knots_xy, degree=self.degree)
        
    # Initial lambda values for Nxy
    lam_xy= lam
    self.lam_mat_xy[0, :] = lam_xy
    self.sigma_xy = 1
    self.accept_frac_xy = 0.4
        
    self.delta_xy[0] = 1
    self.phi_xy[0] = 1
    # Calculate initial log-likelihood and log-posterior
    self.Nx_mat[0,:],self.Nxy_mat[0,:], self.s_n[0, :], self.s_nA[0,:]= noises(self.lam_mat[0, :], self.lam_mat_xy[0, :], self.splines, self.splines_xy, self.Spar, self.Spar_xy, self.modelnum, self.f)

    self.llike[0], prisum, self.psd_mat[0, :], self.s_s[0, :] = llike_psum_s(self, i=0)
    self.logpost[0] = lpost(self.llike[0], prisum)
    ##Some extra code (to be deleted later if does not work)
    self.b_sd = 0.1  # initial proposal stdev for b
    self.g_sd = 0.1  # initial proposal stdev for g

