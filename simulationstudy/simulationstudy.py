
import bnpc
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
import rpy2
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import pickle
import time
import argparse

##Random seed:
parser = argparse.ArgumentParser(description='Simulation script with random seed.')
parser.add_argument('random_seed', type=int, help='Random seed for the simulation')
args = parser.parse_args()
rd = args.random_seed

#R
np_cv_rules = default_converter + numpy2ri.converter
robjects.r['source']('psd_arma.R')
psd_arma = robjects.globalenv['psd_arma']
mcmcr=importr("psplinePsd")

rd = 0  
#AR(4) coefficients
a1=0.9
a2=-0.9
a3=0.9
a4=-0.9
sig=1
#AR(1) coefficients
a1p=0.1883
sig2p=7.839
arex_r = robjects.FloatVector(np.array([0.9, -0.9, 0.9, -0.9]))
ma_ex = robjects.NA_Integer
#number of data points
n=[128,256,512]
series=[]
freq=[]
truepsd=[]
spar=[]



for i in range(0,len(n)):
    random.seed(rd)
    #time series AR(4)
    series.append({f'{n[i]}': bnpc.ar4(n[i], a1, a2, a3, a4, sig)})
    #frequency[0,pi]
    freq.append({f'{n[i]}': np.pi*( 2 * (np.arange(1, n[i] // 2 + 1) - 1) / n[i])[1:]})
    f_r=robjects.FloatVector(freq[i][f'{n[i]}'])
    #truepsd
    truepsd.append({f'{n[i]}':np.log(psd_arma(f_r,arex_r,ma_ex,sig))})
    spar.append({f'{n[i]}':psd_arma(f_r,a1p,ma_ex,sig2p)})


#MCMC:
k=40
degree=3
iterations=50000
burnin=25000
result_r=[]
result=[]
result_t=[]#result when the parametric model is true psd
r_t=[]
p_t=[]
p_t_t=[]
for i in range(0,len(n)): 
    y_c=bnpc.cent_series(series[i][f'{n[i]}'])
    with np_cv_rules.context():
        s_t_r = time.time()
        result_r.append( mcmcr.gibbs_pspline(
                    y_c,
                    burnin=burnin,
                    Ntotal=iterations,
                    degree=3,
                    eqSpacedKnots=False,
                    k=k,
                ))
        e_t_r = time.time()
        r_t.append(e_t_r-s_t_r)
        pdgrm=result_r[i]['pdgrm'][1:-1]
        # AR(1)
        stpy=time.time()
        result.append(bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar[i][f'{n[i]}'], modelnum=1,f=freq[i][f'{n[i]}']))
        etpy=time.time()
        p_t.append(etpy-stpy)
        # AR(4) 
        stpy_t=time.time()
        result_t.append(bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd[i][f'{n[i]}']-2*np.log(np.std(series[i][f'{n[i]}']))), modelnum=1,f=freq[i][f'{n[i]}']))
        etpy_t=time.time()
        p_t_t.append(etpy_t-stpy_t)




####Calculating IAE and prop
iae=[]
iae_t=[]
iae_r=[]
prop=[]
prop_t=[]
prop_r=[]
for i in range(0,len(n)):
    ci_py = bnpc.compute_ci(result[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_py_t = bnpc.compute_ci(result_t[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_r = bnpc.compute_ci((np.log(result_r[i]['fpsd.sample'] * np.std(series[i][f'{n[i]}']) ** 2)[1:-1]).T)
    iae.append(bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iae_t.append(bnpc.compute_iae(np.exp(ci_py_t.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iae_r.append(bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    prop.append(bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd[i][f'{n[i]}']))        
    prop_t.append(bnpc.compute_prop(ci_py_t.u05,ci_py_t.u95,truepsd[i][f'{n[i]}']))
    prop_r.append(bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd[i][f'{n[i]}']))


    

# import os
# os.makedirs(f'sim_res_{n}')

# 

#Outputs:     
col_names = 'py_AR(1)_iae_128 py_AR(1)_iae_256 py_AR(1)_iae_512 py_AR(4)_iae_128 py_AR(4)_iae_256 py_AR(4)_iae_512 r_iae_128 r_iae_256 r_iae_512'
np.savetxt(f'sim_res/{rd}iae.txt', [iae,iae_t,iae_r], header=col_names)

col_names = 'py_AR(1)_prop_128 py_AR(1)_prop_256 py_AR(1)_prop_512 py_AR(4)_prop_128 py_AR(4)_prop_256 py_AR(4)_prop_512 r_prop_128 r_prop_256 r_prop_512'
np.savetxt(f'sim_res/{rd}prop.txt', [prop,prop_t,prop_r], header=col_names)


col_names = 'py_AR(1)_run_t_128 py_AR(1)_run_t_256 py_AR(1)_run_t_512 py_AR(4)_run_t_128 py_AR(4)_run_t_256 py_AR(4)_run_t_512 r_tun_t_128 r_run_t_256 r_run_t_512'
np.savetxt(f'sim_res/{rd}runtime.txt',  [p_t,p_t_t,r_t], header=col_names)

with open(f'sim_res/{rd}resultpy_AR(1)_128_256_512.pkl', 'wb') as f:
     pickle.dump(result, f)

with open(f'sim_res/{rd}resultpy_AR(4)_128_256_512.pkl', 'wb') as f:
     pickle.dump(result_t, f)

with open(f'sim_res/{rd}resultr_128_256_512.pkl', 'wb') as f:
    pickle.dump(result_r, f)


with open(f'sim_res/{rd}tsereis.pkl', 'wb') as f:
    pickle.dump(series, f)
