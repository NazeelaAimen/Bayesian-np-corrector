#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nazeela
"""

import numpy as np
from skfda.representation.basis import BSplineBasis
from scipy.stats import gamma, uniform, norm
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.interpolate import interp1d
from skfda.misc.regularization import L2Regularization    
from skfda.misc.operators import LinearDifferentialOperator
from scipy.stats import multivariate_normal
from scipy.stats import median_abs_deviation
from typing import List
from collections import namedtuple
def mad(x):
   return np.median(abs(x - np.median(x)))

def uniformmax(sample):
    median = np.median(sample)
    mad1 = mad(sample)
    abs_deviation = np.abs(sample - median)
    
    normalized_deviation = abs_deviation / mad1
    max_deviation = np.nanmax(normalized_deviation)
    
    return max_deviation

def compute_iae(psd, truepsd,n): #note use PSD not log PSD
    return sum(abs(psd - truepsd)) * 2* np.pi / n

def compute_prop(u05,u95,truepsd):
    v=[]
    for x in range(len(u05)):
        if (truepsd[x] >= u05[x]) and (truepsd[x] <= u95[x]):
                 v.append(1)
        else:
                 v.append(0)
    return(np.mean(v))

def compute_ci(psds):
    CI = namedtuple('CI', ['u05', 'u95', 'med', 'label'])
    psd_help = np.apply_along_axis(uniformmax, 0, psds)
    psd_mad = median_abs_deviation(psds, axis=0)
    c_value = np.quantile(psd_help, 0.9)
    psd_med = np.median(psds, axis=0)
    psd_u95 = psd_med + c_value * psd_mad
    psd_u05 = psd_med - c_value * psd_mad
    return CI(u05=psd_u05, u95=psd_u95, med=psd_med, label='pypsd')


def plot_psd(freq, pdgrm, truepsd, ci):
    plt.plot(freq, np.log(pdgrm), linestyle='-', color='black', alpha=0.5, label='Periodogram')
    plt.plot(freq, truepsd, linestyle='--', color='black', label='truepsd')
    plt.plot(freq, ci.med, color='red')
    plt.fill_between(
        freq, ci.u05, ci.u95, alpha=0.5, linewidth=0.0, color='red', label=ci.label
         )
    plt.legend()




def ar1(n, a, sig):
    #AR1 time series
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = a * y[t-1] + np.random.normal(loc=0, scale=sig)
    return y

def ar2(n, a1, a2, sig):
    #AR2 time series
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = a1 * y[t-1] + a2 * y[t-2] + np.random.normal(loc=0, scale=sig)
    return y

def ar4(n, a1, a2, a3, a4, sig):
    #AR4 time series
    y = np.zeros(n)
    for t in range(4, n):
        y[t] = a1 * y[t-1] + a2 * y[t-2] + a3 * y[t-3] + a4 * y[t-4] + np.random.normal(loc=0, scale=sig)
    return y

def s_ar1(a, sig2, f):
    #AR1 spectrum
    return sig2 / (1 + a**2 - 2 * a * np.cos(2 * np.pi * f))

def s_ar2(a1, a2, sig2, f):
    #AR2 spectrum
    return sig2 / (1 + a1**2 + a2**2 + 2 * a1 * (a2 - 1) * np.cos(2 * np.pi * f) - 2 * a2 * np.cos(4 * np.pi * f))

def s_ar2e(a1, a2, a3, a4, sig2, f):
    psd=sig2/(np.abs(1-a1*np.exp(1j*2*np.pi*f)-a2*np.exp(1j*4*np.pi*f)))**2
    return psd

def s_ar4(a1, a2, a3, a4, sig2, f):
    #AR4 spectrum
    real_part = (1-a1*np.cos(2*np.pi*f)- a2*np.cos(4*np.pi*f)- a3* np.cos(6*np.pi*f)- a4 * np.cos(8*np.pi*f))**2
    imag_part = (a1*np.sin(2*np.pi*f)+ a2*np.sin(4*np.pi*f)+ a3* np.sin(6*np.pi*f)+ a4 * np.sin(8*np.pi*f))**2 
    psd = sig2 / (real_part + imag_part) 
    return psd

def s_ar4e(a1, a2, a3, a4, sig2, f):
    psd=sig2/(np.abs(1-a1*np.exp(1j*2*np.pi*f)-a2*np.exp(1j*4*np.pi*f)-a3*np.exp(1j*6*np.pi*f)-a4*np.exp(1j*8*np.pi*f)))**2
    return psd


def dens(lam,splines):
    #Spline PSD
    return(np.sum(lam[:, None] * splines, axis=0))

def psd(Snpar, Spar=1, alpha=1, modelnum=0):
    if modelnum==0: #Only splines
        S=Snpar 
    elif modelnum==1:
        S =  alpha*(Snpar) + np.log(Spar)
    elif modelnum==2:
        S = alpha * (Snpar - np.log(Spar)) + np.log(Spar)
    elif modelnum==3:
        S =  2* Snpar - alpha*np.log(Spar)
    elif modelnum==4:
        S=0.5*Snpar+0.5*(alpha+1)*np.log(Spar)
    else:
        S=np.log(Spar)+Snpar*np.log(10)
    return(S)

def loglike(pdgrm, S):
    lnlike = -1* np.sum(S + np.exp(np.log(pdgrm) - S))
    return lnlike

def lamb_lprior(lam, phi, P, k):
    return k * np.log(phi) / 2 - phi * np.matmul(np.transpose(lam), np.matmul(P, lam)) / 2

def phi_lprior(phi, delta):
    return gamma.logpdf(phi, a=1, scale=1/delta)

def delta_lprior(delta):
    return gamma.logpdf(delta, a=1e-4, scale=1/1e-4)

def alpha_lprior(alpha):
    return uniform.logpdf(alpha)

def lpost(loglike, lamb_lpri, phi_lpri, delta_lpri, alpha_lpri=0):
    return loglike + lamb_lpri + phi_lpri + delta_lpri + alpha_lpri

def diffMatrix(k, d=2):
    out = np.eye(k)
    for i in range(d):
        out = np.diff(out, axis=0)
    return out


def data_peak_knots(data: np.ndarray, n_knots: int) -> np.ndarray:#based on Patricio's pspline paper
    aux = np.sqrt(data)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(data)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    df = interp1d(
        np.linspace(0, 1, num=n), cumf, kind="linear", fill_value=(0, 1)
    )

    invDf = interp1d(
        df(np.linspace(0, 1, num=n)),
        np.linspace(0, 1, num=n),
        kind="linear",
        fill_value=(0, 1),
        bounds_error=False,
    )
    return invDf(np.linspace(0, 1, num=n_knots))

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

def mcmc(pdgrm,n,k,burnin,Spar=1,degree=3,modelnum=1,alphastart=None,f=None,fs=None):
    delta = np.zeros(n)
    phi = np.zeros(n)
    llike = np.zeros(n)
    logpost = np.zeros(n)
    if f is None:
        f= np.linspace(0,fs/2, len(pdgrm))[1:]
        pdgrm=pdgrm[1:]
        if Spar!=1:
            Spar=Spar[1:]
    lam_mat=np.zeros((n,k))
    splines_mat=np.zeros((n,len(pdgrm)))
    psd_mat=np.zeros((n,len(pdgrm)))
    # Initial values
    lam = pdgrm / np.sum(pdgrm)
    lam = lam[np.round(np.linspace(0, len(lam) - 1, k)).astype(int)]
    lam[lam == 0] = 1e-50
    lam_mat[0,:]=lam

    delta[0] = 1
    phi[0] = 1
    
    # 
    sigma= 1
    accept_frac=0.4
    
    #for variable alpha:
    if alphastart!=None:
        alpha = np.zeros(n)
        alpha[0] = alphastart
    

    # Knots over Fourier frequencies
    K = k - degree + 1
    #knots = np.linspace(min(f), max(f), num=K)#Equidistant
    #difference knots:
    data=abs(Spar-pdgrm)
    knots=data_peak_knots(data,K)
    m = max(f) - min(f)
    c = min(f)
    knots = m * knots + c #Linear translation from [0,1] to fourier frequencies range

    #making basis matrix:
    gridp=np.linspace(knots[0],knots[-1],len(pdgrm))
    splines=generate_basis_matrix(knots,gridp,3)
   
    #Panelty matrix
    #linear
    # P = diffMatrix(k, d=2)
    # P = np.matmul(np.transpose(P), P)
    # data variations:
    P=panelty_datvar(d=1,knots=knots,degree=degree)
     
    #splines:
    splines_mat[0,:]=dens(lam,splines.T)
    
    if alphastart==None:
        #for constant alpha
        S=psd(splines_mat[0,:], Spar=Spar,  modelnum=modelnum)
        llike[0] = loglike(pdgrm=pdgrm, S=S)
        logpost[0] = lpost(llike[0], lamb_lprior(lam, phi[0], P, k),
                            phi_lprior(phi[0], delta[0]),
                            delta_lprior(delta[0]))
    else:
        #for variable alpha
        S=psd(splines_mat[0,:], Spar=Spar, alpha=alpha[0], modelnum=modelnum)
        llike[0] = loglike(pdgrm=pdgrm, S=S)
        logpost[0] = lpost(llike[0], lamb_lprior(lam, phi[0], P, k),
                      phi_lprior(phi[0], delta[0]),
                      delta_lprior(delta[0]),
                      alpha_lprior(alpha[0]))
    psd_mat[0,:]=S
    for i in range(1,n):
        # updating lambda
        # based on Patricio's pspline paper
        if accept_frac < 0.30:  
            sigma = sigma * 0.90 
        elif accept_frac > 0.50:  
            sigma = sigma * 1.1  
        accept_count = 0
        aux = np.arange(0, k)
        np.random.shuffle(aux)
        for g in range(0, len(lam)):
            z = np.random.normal()
            u = np.log(np.random.uniform())
            pos = aux[g]
            lam_p=lam[pos]
            lam_star = lam_p + sigma * z
            
            if alphastart==None:
                ftheta = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar,  modelnum=modelnum)),
                                lamb_lprior(lam, phi[i - 1], P, k),
                                phi_lprior(phi[i - 1], delta[i - 1]),
                                delta_lprior(delta[i - 1]))
                lam[pos] = lam_star
                ftheta_star = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar,  modelnum=modelnum)),
                                    lamb_lprior(lam, phi[i - 1], P, k),
                                    phi_lprior(phi[i - 1], delta[i - 1]),
                                    delta_lprior(delta[i - 1]))
            else:
                ftheta = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar, alpha=alpha[i - 1],  modelnum=modelnum)),
                              lamb_lprior(lam, phi[i - 1], P, k),
                              phi_lprior(phi[i - 1], delta[i - 1]),
                              delta_lprior(delta[i - 1]),
                              alpha_lprior(alpha[i - 1]))
                lam[pos] = lam_star
                ftheta_star = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar, alpha=alpha[i - 1],  modelnum=modelnum)),
                                lamb_lprior(lam, phi[i - 1], P, k),
                                phi_lprior(phi[i - 1], delta[i - 1]),
                                delta_lprior(delta[i - 1]),
                                alpha_lprior(alpha[i - 1]))
            
            
            A = np.min([0, ftheta_star - ftheta])
            if u>A:
                lam[pos]=lam_p
            else:
                accept_count+=1
        accept_frac = accept_count / k
        splines_mat[i,:]=dens(lam,splines.T)
        lam_mat[i,:]=lam

        # Sampling phi
        a_phi = k / 2 + 1
        b_phi = 0.5 * np.matmul(np.transpose(lam), np.matmul(P, lam)) + delta[i - 1]
        phi[i] = gamma.rvs(a=a_phi, scale=1 / b_phi, size=1)

        # Sampling delta
        a_delta = 1 + 1e-4
        b_delta = phi[i] + 1e-4
        delta[i] = gamma.rvs(a=a_delta, scale=1 / b_delta, size=1)
        
        
        # Updating alpha
        if alphastart!=None:
            alpha[i] = np.random.normal(alpha[i - 1], 1)
            #Re-sample until aprop is within the range [0, 1]
            while alpha[i] < 0 or alpha[i] > 1:
                alpha[i] = np.random.normal(alpha[i - 1], 1)
            
            ftheta = lpost(loglike(pdgrm=pdgrm, S=psd(splines_mat[i,:], Spar=Spar, alpha=alpha[i - 1],  modelnum=modelnum)),
                            lamb_lprior(lam, phi[i], P, k),
                            phi_lprior(phi[i], delta[i]),
                            delta_lprior(delta[i]),
                            alpha_lprior(alpha[i-1]))
            ftheta_star = lpost(loglike(pdgrm=pdgrm, S=psd(splines_mat[i,:], Spar=Spar, alpha=alpha[i],  modelnum=modelnum)),
                                lamb_lprior(lam, phi[i], P, k),
                                phi_lprior(phi[i], delta[i]),
                                delta_lprior(delta[i]),
                                alpha_lprior(alpha[i]))
            
            A = min([0, ftheta_star - ftheta])
            
            # Accept or reject the proposed alpha value
            if np.log(np.random.rand()) > A:
                alpha[i] = alpha[i - 1]
            S=psd(splines_mat[i,:], Spar=Spar, alpha=alpha[i], modelnum=modelnum)
            llike[i] = loglike(pdgrm=pdgrm, S=S)
            logpost[i] = lpost(llike[i], lamb_lprior(lam, phi[i], P, k),
                                phi_lprior(phi[i], delta[i]),
                                delta_lprior(delta[i]),
                                alpha_lprior(alpha[i]))
        
        
        else:
            S=psd(splines_mat[i,:], Spar=Spar, modelnum=modelnum)
            llike[i] = loglike(pdgrm=pdgrm, S=S)
            logpost[i] = lpost(llike[i], lamb_lprior(lam, phi[i], P, k),
                                phi_lprior(phi[i], delta[i]),
                                delta_lprior(delta[i]))
        psd_mat[i,:]=S
        
        
    phi = phi[burnin:n]
    delta = delta[burnin:n]
    llike = llike[burnin:n]
    logpost = logpost[burnin:n]
    lam_mat=lam_mat[burnin:n,:]
    psd_mat=psd_mat[burnin:n,:]
    splines_mat=splines_mat[burnin:n,:]

    if alphastart==None:
        result = {'phi': phi,
                  'delta': delta,
                  'llike': llike,
                  'logpost': logpost,
                  'lambda': lam_mat,
                  'knots': knots,
                  'splines_psd':splines_mat,
                  'psd':psd_mat}
    else:
        alpha = alpha[burnin:n]
        result = {'phi': phi,
                  'delta': delta,
                  'llike': llike,
                  'logpost': logpost,
                  'lambda': lam_mat,
                  'knots': knots,
                  'alpha': alpha,
                  'splines_psd':splines_mat,
                  'psd':psd_mat}
    

    return result


def mcmcAMH(pdgrm,n,k,burnin,Spar=1,degree=3,modelnum=1,alphastart=None,f=None,fs=None):
    delta = np.zeros(n)
    phi = np.zeros(n)
    llike = np.zeros(n)
    logpost = np.zeros(n)
    if f is None:
        f= np.linspace(0,fs/2, len(pdgrm))[1:]
        pdgrm=pdgrm[1:]
        if Spar!=1:
            Spar=Spar[1:]
    lam_mat=np.zeros((n,k))
    splines_mat=np.zeros((n,len(pdgrm)))
    psd_mat=np.zeros((n,len(pdgrm)))
    # Initial values
    lam = pdgrm / np.sum(pdgrm)
    lam = lam[np.round(np.linspace(0, len(lam) - 1, k)).astype(int)]
    lam[lam == 0] = 1e-50
    lam_mat[0,:]=lam

    delta[0] = 1
    phi[0] = 1
    
    # 
    sigma= 1
    accept_frac=0.4
    
    
    

    # Knots over Fourier frequencies
    K = k - degree + 1
    data=abs(Spar-pdgrm)
    knots=data_peak_knots(pdgrm,K)
    m = max(f) - min(f)
    c = min(f)
    knots = m * knots + c #Linear translation from [0,1] to fourier frequencies range
    #making basis matrix:
    gridp=np.linspace(knots[0],knots[-1],len(pdgrm))
    splines=generate_basis_matrix(knots,gridp,3)
    P=panelty_datvar(d=1,knots=knots,degree=degree)      
    #P=panelty_datvar(k,degree,1,knots)
     
    #splines:
    splines_mat[0,:]=dens(lam,splines.T)
    Uv_am = np.random.uniform(0, 1, n)
    Uv = np.log(np.random.uniform(0, 1, n))

    if alphastart==None:
        #for constant alpha
        S=psd(splines_mat[0,:], Spar=Spar,  modelnum=modelnum)
        llike[0] = loglike(pdgrm=pdgrm, S=S)
        logpost[0] = lpost(llike[0], lamb_lprior(lam, phi[0], P, k),
                            phi_lprior(phi[0], delta[0]),
                            delta_lprior(delta[0]))
        Ik = (0.1**2) * np.diag(np.ones(k) / k)
        covObj = updateCov(lam, None)
        c_amh = (2.38**2) / k
    else:
        #for variable alpha
        alpha = np.zeros(n)
        alpha[0] = alphastart
        S=psd(splines_mat[0,:], Spar=Spar, alpha=alpha[0], modelnum=modelnum)
        llike[0] = loglike(pdgrm=pdgrm, S=S)
        logpost[0] = lpost(llike[0], lamb_lprior(lam, phi[0], P, k),
                      phi_lprior(phi[0], delta[0]),
                      delta_lprior(delta[0]),
                      alpha_lprior(alpha[0]))
        Ik = (0.1**2) * np.diag(np.ones(k+1) / (k+1))
        covObj = updateCov(np.append(lam,alpha[0]), None)
        c_amh = (2.38**2) / (k+1)
    
    psd_mat[0,:]=S
    count = []
    for i in range(1,n):
        
        # updating lambda
        # based on Patricio's pspline paper
        if (Uv_am[i] < 0.05) or (i <= 2*k):
            if alphastart==None:
                lam_star = multivariate_normal.rvs(mean=lam, cov=Ik)
            else:
                sth = multivariate_normal.rvs(mean=np.append(lam,alpha[i-1]), cov=Ik)
                lam_star=sth[:-1]
                alpha[i]=sth[-1]
        else:
            if alphastart==None:
                lam_star = multivariate_normal.rvs(mean=lam, cov=c_amh * upcov)
            else:
                sth = multivariate_normal.rvs(mean=np.append(lam,alpha[i-1]), cov=c_amh * upcov)
                lam_star=sth[:-1]
                alpha[i]=sth[-1]
        
        if alphastart==None:
            ftheta = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar,  modelnum=modelnum)),
                            lamb_lprior(lam, phi[i - 1], P, k),
                            phi_lprior(phi[i - 1], delta[i - 1]),
                            delta_lprior(delta[i - 1]))
            ftheta_star = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam_star,splines.T), Spar=Spar,  modelnum=modelnum)),
                                lamb_lprior(lam_star, phi[i - 1], P, k),
                                phi_lprior(phi[i - 1], delta[i - 1]),
                                delta_lprior(delta[i - 1]))
        else:
            ftheta = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam,splines.T), Spar=Spar, alpha=alpha[i - 1],  modelnum=modelnum)),
                          lamb_lprior(lam, phi[i - 1], P, k),
                          phi_lprior(phi[i - 1], delta[i - 1]),
                          delta_lprior(delta[i - 1]),
                          alpha_lprior(alpha[i - 1]))
            ftheta_star = lpost(loglike(pdgrm=pdgrm, S=psd(dens(lam_star,splines.T), Spar=Spar, alpha=alpha[i - 1],  modelnum=modelnum)),
                            lamb_lprior(lam, phi[i - 1], P, k),
                            phi_lprior(phi[i - 1], delta[i - 1]),
                            delta_lprior(delta[i - 1]),
                            alpha_lprior(alpha[i]))
        A = np.min([0, ftheta_star - ftheta])
        if Uv[i]<A:
            lam=lam_star
            count.append(1)
        else:
            count.append(0)
            if alphastart!=None:
                alpha[i]=alpha[i-1]
        splines_mat[i,:]=dens(lam,splines.T)
        lam_mat[i,:]=lam
        if alphastart==None:
            covObj = updateCov(lam, covObj)
        else:
            covObj = updateCov(np.append(lam,alpha[i]), covObj)
        upcov=covObj['cov']

        # Sampling phi
        a_phi = k / 2 + 1
        b_phi = 0.5 * np.matmul(np.transpose(lam), np.matmul(P, lam)) + delta[i - 1]
        phi[i] = gamma.rvs(a=a_phi, scale=1 / b_phi, size=1)

        # Sampling delta
        a_delta = 1 + 1e-4
        b_delta = phi[i] + 1e-4
        delta[i] = gamma.rvs(a=a_delta, scale=1 / b_delta, size=1)
        
        
        # Updating alpha
        if alphastart!=None:
            S=psd(splines_mat[i,:], Spar=Spar, alpha=alpha[i], modelnum=modelnum)
            llike[i] = loglike(pdgrm=pdgrm, S=S)
            logpost[i] = lpost(llike[i], lamb_lprior(lam, phi[i], P, k),
                                phi_lprior(phi[i], delta[i]),
                                delta_lprior(delta[i]),
                                alpha_lprior(alpha[i]))
        
        
        else:
            S=psd(splines_mat[i,:], Spar=Spar, modelnum=modelnum)
            llike[i] = loglike(pdgrm=pdgrm, S=S)
            logpost[i] = lpost(llike[i], lamb_lprior(lam, phi[i], P, k),
                                phi_lprior(phi[i], delta[i]),
                                delta_lprior(delta[i]))
        psd_mat[i,:]=S
        
        
    phi = phi[burnin:n]
    delta = delta[burnin:n]
    llike = llike[burnin:n]
    logpost = logpost[burnin:n]
    lam_mat=lam_mat[burnin:n,:]
    psd_mat=psd_mat[burnin:n,:]
    splines_mat=splines_mat[burnin:n,:]

    if alphastart==None:
        result = {'phi': phi,
                  'delta': delta,
                  'llike': llike,
                  'logpost': logpost,
                  'lambda': lam_mat,
                  'knots': knots,
                  'splines_psd':splines_mat,
                  'psd':psd_mat}
    else:
        alpha = alpha[burnin:n]
        result = {'phi': phi,
                  'delta': delta,
                  'llike': llike,
                  'logpost': logpost,
                  'lambda': lam_mat,
                  'knots': knots,
                  'alpha': alpha,
                  'splines_psd':splines_mat,
                  'psd':psd_mat}
    

    return result