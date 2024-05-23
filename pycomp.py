#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:54:43 2024

@author: naim769
"""

import bnpc
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import rpy2
import rpy2.robjects as robjects
from collections import namedtuple
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from scipy.stats import median_abs_deviation

CI = namedtuple('CI', ['u05', 'u95', 'med', 'label', 'iae'])

N_TOTAL = 5000
BURNIN = 1000


# def uniformmax():
def mad(x):
    return np.median(abs(x - np.median(x)))




class MCMCdata:
    def __init__(self):
        self.fz = None
        self.v = None
        self.dblist = None
        self.psds = None
        self.psd_quantiles = None
        self.lnl = None
        self.samples = None

    @classmethod
    def from_r(cls, mcmc):
        obj = cls()
        obj.fz = None
        obj.v = mcmc["V"]
        obj.dblist = mcmc["db.list"]
        obj.psd = mcmc["fpsd.sample"]
        obj.psd_quantiles = np.array(
            [
                np.array(mcmc["psd.median"]),
                np.array(mcmc["psd.p05"]),
                np.array(mcmc["psd.p95"]),
            ]
        )
        obj.lnl = mcmc["ll.trace"]
        obj.samples = np.array([mcmc["phi"], mcmc["delta"], mcmc["tau"]]).T
        return obj


# Load the R script containing the function
random_seed = 0
robjects.r['set.seed'](random_seed)

robjects.r['source']('psd_arma.R')

# Access the R function
psd_arma = robjects.globalenv['psd_arma']

n = 512
freq = np.pi * (2 * (np.arange(1, n // 2 + 1) - 1) / n)[1:]
arex = np.array([0.9, -0.9, 0.9, -0.9])

sigma2_ex = 1
f_r = robjects.FloatVector(freq)
arex_r = robjects.FloatVector(arex)
ma_ex = robjects.NA_Integer

truepsd = np.log(psd_arma(f_r, arex_r, ma_ex, sigma2_ex))

y = robjects.r['arima.sim'](robjects.r['list'](ar=arex_r), n=n)
y_c = y - np.mean(y)
scale = np.std(y)
y_c = y_c / scale
# pdgrm = bnpc.makepdgrm_pi(y_c)
# f=pdgrm['f']
# parametric model:
a1p = 0.1883
sig2p = 7.839
ar1psd = psd_arma(f_r, a1p, ma_ex, sig2p)
mcmcr = importr("psplinePsd")

k = 25
degree = 3

np_cv_rules = default_converter + numpy2ri.converter
with np_cv_rules.context():
    mcmc_r_res = mcmcr.gibbs_pspline(
        y_c,
        burnin=BURNIN,
        Ntotal=N_TOTAL,
        degree=3,
        eqSpacedKnots=False,
        k=k,
    )

pdgrm = mcmc_r_res['pdgrm'][1:-1]
resultpy = bnpc.mcmc(pdgrm=pdgrm, n=N_TOTAL, k=k, burnin=BURNIN, f=freq, Spar=ar1psd, modelnum=1)
log_psds_py = resultpy['psd'] + 2 * np.log(scale)
log_psds_r = np.log(mcmc_r_res['fpsd.sample'] * scale ** 2)[1:-1, :]


pdgrm = pdgrm * scale ** 2


def compute_iae(psd, truepsd):
    return sum(abs(psd - truepsd)) * np.pi / len(psd)


def compute_ci_py(psds):
    psd_help = np.apply_along_axis(bnpc.uniformmax, 0, psds)
    psd_mad = median_abs_deviation(psds, axis=0)
    c_value = np.quantile(psd_help, 0.9)
    psd_med = np.median(psds, axis=0)
    psd_u95 = psd_med + c_value * psd_mad
    psd_u05 = psd_med - c_value * psd_mad
    iae = compute_iae(psd_med, truepsd)
    return CI(u05=psd_u05, u95=psd_u95, med=psd_med, label='pypsd', iae=iae)


def compute_ci_r(psds):
    # Access the R function
    robjects.r['source']('uniform.R')
    unifmax = robjects.globalenv['uniformmax']
    apply = robjects.r['apply']
    mad = robjects.r['mad']
    med = robjects.r['median']
    with np_cv_rules.context():
        psd_help = apply(psds, 1, unifmax)
        psd_mad = apply(psds, 1, mad)
        psd_med = apply(psds, 1, med)
    c_value = np.quantile(psd_help, 0.9)
    psd_u95 = psd_med + c_value * psd_mad
    psd_u05 = psd_med - c_value * psd_mad
    iae = compute_iae(psd_med, truepsd)
    return CI(u05=psd_u05, u95=psd_u95, med=psd_med, label='rpsd', iae=iae)


from typing import List


def plot_psd(freq, pdgrm, truepsd, cis: List[CI], colors: List[str], hatchs):
    plt.plot(freq, np.log(pdgrm), linestyle='-', color='black', alpha=0.5, label='Periodogram')
    plt.plot(freq, truepsd, linestyle='--', color='black', label='truepsd')
    for i, ci in enumerate(cis):
        plt.plot(freq, ci.med, color=colors[i])
        # use shaded lines to show the CI

        plt.fill_between(
            freq, ci.u05, ci.u95, alpha=0.5, linewidth=0.0, color=colors[i], label=ci.label,
            hatch=hatchs[i]
             )
    plt.legend()
    plt.savefig('PSD_comp.png', dpi=300, bbox_inches='tight')


ci_py = compute_ci_py(log_psds_py)
# # relabel the CI
# ci_py
#
# ci_py['label'] = 'py(psd-nazeela)'
ci_py_using_r = compute_ci_r(log_psds_py.T)
# ci_py_using_r['label'] = 'r(psd-nazeela)'
ci_r = compute_ci_r(log_psds_r)
# ci_r['label'] = 'r(psd-r)'

mse = lambda x, y: np.mean((x - y) ** 2)

print(mse(ci_py.u05, ci_py_using_r.u05))
print(mse(ci_py.u95, ci_py_using_r.u95))

plot_psd(freq, pdgrm, truepsd, [ci_py, ci_py_using_r, ci_r], ['blue', 'red', 'green'], hatchs=['\\', '/', ''])
