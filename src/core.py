#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:37:58 2024

@author: naim769
"""
import numpy as np
from scipy.stats import gamma, uniform, norm, invgamma, multivariate_normal, truncnorm
from skfda.representation.basis import BSplineBasis
from scipy.interpolate import interp1d
from skfda.misc.regularization import L2Regularization    
from skfda.misc.operators import LinearDifferentialOperator
from utils import updata_phi_A, determinant

def diffMatrix(k, d=2):
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out, axis=0)
    return out

def dens(lam,splines):
    #Spline PSD
    return(np.sum(lam[:, None] * splines, axis=0))

#A channel noise
def N_A(Nx, Nxy, f):
    Nx=np.exp(Nx)
    Nxy=np.exp(Nxy)
    return np.log(2/3*(Nx - Nxy))



#T channel noise
def N_T (Nx, Nxy, f):
    Nx=np.exp(Nx)
    Nxy=np.exp(Nxy)
    return np.log(1 / 3 * abs(Nx + 2 * Nxy))



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

def noises(lam, lam_xy, splines, splines_xy, Spar, Spar_xy, modelnum,f):
        Nx=psd(dens(lam, splines.T), Spar=Spar, modelnum=modelnum)
        Nxy=psd(dens(lam_xy, splines_xy.T), Spar=Spar_xy, modelnum=modelnum)
        noise=N_T(Nx,Nxy,f)
        noise_A=N_A(Nx,Nxy,f)
        return Nx, Nxy, noise, noise_A

   
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
    # A and E contains the signal. However, T does not contain the signal
    lnlike = loglike1(A,S)+loglike1(E,S)+loglike1(T,s_n)
    return lnlike


def psilprior(psi):
    return norm.logpdf(psi, loc=-4, scale=0.1)

# def b_lprior(b):
#     return norm.logpdf(b, loc=np.log(1e28), scale=0.1)

# def glprior(g):
#     return norm.logpdf(g, loc=-2/3, scale=0.1)


def b_lprior(b):
    return uniform.logpdf(b, 64.45, 64.5)

def glprior(g):
    return uniform.logpdf(g, -0.65, -0.67)

def lamb_lprior(lam, phi, P, k):
    return k * np.log(phi) / 2 - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2

# def lamb_A_lprior(lam_A, lam_mat, k):
#     sigma_lam=cov_matrix(lam_mat)
#     sigma_inv=inverse(sigma_lam)
#     mean_vec=np.mean(lam_mat, axis=0)
#     return -k * np.log(determinant(sigma_lam)) / 2 -  np.matmul(np.transpose(lam_A-mean_vec), np.matmul(sigma_inv, (lam_A-mean_vec))) / 2

# def lamb_A_lprior(lam_A, lam_mat, P_A, k):
#     phi=updata_phi_A(lam_mat, P_A)
#     mean_vec=np.mean(lam_mat, axis=0)
#     mult=phi @ P_A @ phi 
#     return k * np.log(determinant(mult)) / 2 -  np.matmul(np.transpose(lam_A-mean_vec), np.matmul(mult, (lam_A-mean_vec))) / 2
    
    
def phi_lprior(phi, delta):
    return gamma.logpdf(phi, a=1, scale=1/delta)

def delta_lprior(delta):
    return gamma.logpdf(delta, a=1e-4, scale=1/1e-4)

#noise priors:
def Na_prior(Na):
    return gamma.logpdf(Na, a=0.05, scale=0.05)

#noise priors:
def Np_prior(Np):
    return gamma.logpdf(Np, a=0.05, scale=0.05)

def prior_sum(lamb_lpri, phi_lpri, delta_lpri, lamb_lpri_xy=0, phi_lpri_xy=0, delta_lpri_xy=0, b_lpri=0 , g_lpri=0, psi_lprior=0):
    return lamb_lpri + phi_lpri + delta_lpri + lamb_lpri_xy + phi_lpri_xy + delta_lpri_xy + b_lpri + g_lpri + psi_lprior

def lpost(loglike, lpriorsum):
    return loglike + lpriorsum


