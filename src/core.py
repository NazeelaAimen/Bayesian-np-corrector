#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:37:58 2024

@author: naim769
"""
import numpy as np
from scipy.stats import gamma, uniform, norm, invgamma
from skfda.representation.basis import BSplineBasis
from scipy.interpolate import interp1d
from skfda.misc.regularization import L2Regularization    
from skfda.misc.operators import LinearDifferentialOperator

def diffMatrix(k, d=2):
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out, axis=0)
    return out

def dens(lam,splines):
    #Spline PSD
    return(np.sum(lam[:, None] * splines, axis=0))


def psd(Snpar, Spar=1, modelnum=0):
    # models for PSD of noise
    if modelnum==0: #Only splines
        S=Snpar 
    elif modelnum==1:
        S =Snpar + np.log(Spar)
    elif modelnum==4:
        S=0.5*Snpar+np.log(Spar)
    else:
        S=np.log(Spar)+Snpar*np.log(10)
    return(S)


   
def tot_psd(s_n,s_s):
    #s_n is the log PSD of noise
    #s_s is the log PSD of signal
    # this function is defined to calculate the log of sums of the PSD which is going to be used in the log likelihood
    sth=s_n > s_s
    S = np.where(sth,
                  s_n + np.log(1+np.exp(s_s - s_n)),
                  s_s + np.log(1+np.exp(s_n - s_s)))
    return S

    
def generate_basis_matrix(knots,grid_points,degree, normalised: bool = True) -> np.ndarray:#slipper pspline psd
        basis = BSplineBasis(knots=knots,order=degree+1).to_basis()
        basis_matrix = basis.to_grid(grid_points).data_matrix.squeeze().T

        if normalised:
            # normalize the basis functions
            knots_with_boundary = np.concatenate(
                [
                    np.repeat(knots[0], degree),
                    knots,
                    np.repeat(knots[-1], degree),
                ]
            )
            n_knots = len(knots_with_boundary)
            mid_to_end_knots = knots_with_boundary[degree + 1 :]
            start_to_mid_knots = knots_with_boundary[
                : (n_knots - degree - 1)
            ]
            bs_int = (mid_to_end_knots - start_to_mid_knots) / (
                degree + 1
            )
            bs_int[bs_int == 0] = np.inf
            basis_matrix = basis_matrix / bs_int
        return basis_matrix   

def panelty_datvar(d,knots,degree=3,epsi=1e-6):
    basis=BSplineBasis(knots=knots,order=degree+1)
    regularization = L2Regularization(
                LinearDifferentialOperator(d)
            )
    p = regularization.penalty_matrix(basis)
    p / np.max(p)
    return p + epsi * np.eye(p.shape[1])

def panelty_linear(k,d):
    #linear
    P = diffMatrix(k, d=2)
    P = np.matmul(np.transpose(P), P)
    return(P)

def loglike1(pdgrm, S):
    #likelihood for one channel
    lnlike = -1* np.sum(S + np.exp(np.log(pdgrm) - S))
    return lnlike


def loglike(A,E,T,S,s_n):
    # log likelihood for three channels
    # this is simply the sum of the likelihoods for the three channels
    # A and E contains the signal. However, T does not contain the signal
    lnlike = loglike1(A,S)+loglike1(E,S)+loglike1(T,s_n)
    return lnlike


def psilprior(psi):
    return norm.logpdf(psi, loc=-4, scale=1)

def b_lprior(b):
    return norm.logpdf(b, loc=-4, scale=1)

def glprior(g):
    return norm.logpdf(g, loc=2, scale=1)

def lamb_lprior(lam, phi, P, k):
    return k * np.log(phi) / 2 - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2

def phi_lprior(phi, delta):
    return gamma.logpdf(phi, a=1, scale=1/delta)

def delta_lprior(delta):
    return gamma.logpdf(delta, a=1e-4, scale=1/1e-4)

def prior_sum(lamb_lpri, phi_lpri, delta_lpri, b_lpri=0 , g_lpri=0, psi_lprior=0):
    return lamb_lpri + phi_lpri + delta_lpri + b_lpri + g_lpri + psi_lprior

def lpost(loglike, lpriorsum):
    return loglike + lpriorsum


def updateCov(X, cov_obj=None):
    if cov_obj is None:
        cov_obj = {'mean': X, 'cov': None, 'n': 1}
        return cov_obj

    cov_obj['n'] += 1  # Update number of observations

    if cov_obj['n'] == 2:
        X1 = cov_obj['mean']
        cov_obj['mean'] = X1 / 2 + X / 2
        dX1 = X1 - cov_obj['mean']
        dX2 = X - cov_obj['mean']
        cov_obj['cov'] = np.outer(dX1, dX1) + np.outer(dX2, dX2)
        return cov_obj

    dx = cov_obj['mean'] - X  # previous mean minus new X
    cov_obj['cov'] = cov_obj['cov'] * (cov_obj['n'] - 2) / (cov_obj['n'] - 1) + np.outer(dx, dx) / cov_obj['n']
    cov_obj['mean'] = cov_obj['mean'] * (cov_obj['n'] - 1) / cov_obj['n'] + X / cov_obj['n']
    return cov_obj

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