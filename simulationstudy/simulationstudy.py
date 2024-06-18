
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
import pandas as pd
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


#AR(4) coefficients
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
    series.append({f'{n[i]}': pd.read_csv(f"ts{n[i]}.csv").iloc[rd]})
    #frequency[0,pi]
    freq.append({f'{n[i]}': np.pi*( 2 * (np.arange(1, n[i] // 2 + 1) - 1) / n[i])[1:]})
    f_r=robjects.FloatVector(freq[i][f'{n[i]}'])
    #truepsd
    truepsd.append({f'{n[i]}':np.log(psd_arma(f_r,arex_r,ma_ex,sig))})
    spar.append({f'{n[i]}':psd_arma(f_r,a1p,ma_ex,sig2p)})


#MCMC:
k=40
degree=3
iterations=100000
burnin=50000
result_r=[]
resultMH=[]
resultMH_t=[]#result when the parametric model is true psd while updating lambda using MH
result=[]
result_t=[]#result when the parametric model is true psd while updating lambda using AMH
r_t=[]
p_t=[]
pMH_t=[]
p_t_t=[]
pMH_t_t=[]


####Calculating IAE and prop
iae=[]
iae_t=[]
iaeMH=[]
iaeMH_t=[]
iae_r=[]
prop=[]
prop_t=[]
propMH=[]
propMH_t=[]
prop_r=[]
for i in range(0,len(n)): 
    #MCMC
    y_c=np.array(bnpc.cent_series(series[i][f'{n[i]}']))
    with np_cv_rules.context():
        s_t_r = time.time()
        result_r.append( mcmcr.gibbs_pspline(
                    y_c,
                    burnin=burnin,
                    Ntotal=iterations,
                    degree=3,
                    eqSpacedKnots=False,
                    k=k,
                    printIter =50000
                ))
    e_t_r = time.time()
    r_t.append(e_t_r-s_t_r)
    pdgrm=result_r[i]['pdgrm'][1:-1]
    # AR(1)
    ##AMH
    stpy=time.time()
    result.append(bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar[i][f'{n[i]}'], modelnum=1,f=freq[i][f'{n[i]}']))
    etpy=time.time()
    p_t.append(etpy-stpy)
    #MH
    stpy=time.time()
    resultMH.append(bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar[i][f'{n[i]}'], modelnum=1,f=freq[i][f'{n[i]}']))
    etpy=time.time()
    pMH_t.append(etpy-stpy)
    # AR(4) 
    #AMH
    stpy_t=time.time()
    result_t.append(bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd[i][f'{n[i]}']-2*np.log(np.std(series[i][f'{n[i]}']))), modelnum=1,f=freq[i][f'{n[i]}']))
    etpy_t=time.time()
    p_t_t.append(etpy_t-stpy_t)
    # MH
    stpy_t=time.time()
    resultMH_t.append(bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd[i][f'{n[i]}']-2*np.log(np.std(series[i][f'{n[i]}']))), modelnum=1,f=freq[i][f'{n[i]}']))
    etpy_t=time.time()
    pMH_t_t.append(etpy_t-stpy_t)
    
    #CI
    ci_py = bnpc.compute_ci(result[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_py_t = bnpc.compute_ci(result_t[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_pyMH = bnpc.compute_ci(resultMH[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_py_tMH = bnpc.compute_ci(resultMH_t[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_r = bnpc.compute_ci((np.log(result_r[i]['fpsd.sample'] * np.std(series[i][f'{n[i]}']) ** 2)[1:-1]).T)
    #iae
    iae.append(bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iae_t.append(bnpc.compute_iae(np.exp(ci_py_t.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iaeMH.append(bnpc.compute_iae(np.exp(ci_pyMH.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iaeMH_t.append(bnpc.compute_iae(np.exp(ci_py_tMH.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iae_r.append(bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    #prop
    prop.append(bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd[i][f'{n[i]}']))        
    prop_t.append(bnpc.compute_prop(ci_py_t.u05,ci_py_t.u95,truepsd[i][f'{n[i]}']))
    propMH.append(bnpc.compute_prop(ci_pyMH.u05,ci_pyMH.u95,truepsd[i][f'{n[i]}']))        
    propMH_t.append(bnpc.compute_prop(ci_py_tMH.u05,ci_py_tMH.u95,truepsd[i][f'{n[i]}']))
    prop_r.append(bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd[i][f'{n[i]}']))
    
    
# import os
# os.makedirs(f'sim_res_{n}')
# 

#Outputs:     
col_names = 'pyAMH_AR(1)_iae_128 pyAMH_AR(1)_iae_256 pyAMH_AR(1)_iae_512 pyAMH_AR(4)_iae_128 pyAMH_AR(4)_iae_256 pyAMH_AR(4)_iae_512 pyMH_AR(1)_iae_128 pyMH_AR(1)_iae_256 pyMH_AR(1)_iae_512 pyMH_AR(4)_iae_128 pyMH_AR(4)_iae_256 pyMH_AR(4)_iae_512 r_iae_128 r_iae_256 r_iae_512'
np.savetxt(f'sim_res/{rd}iae.txt', [iae,iae_t,iaeMH,iaeMH_t,iae_r], header=col_names)

col_names = 'pyAMH_AR(1)_prop_128 pyAMH_AR(1)_prop_256 pyAMH_AR(1)_prop_512 pyAMH_AR(4)_prop_128 pyAMH_AR(4)_prop_256 pyAMH_AR(4)_prop_512 pyAMH_AR(1)_prop_128 pyAMH_AR(1)_prop_256 pyAMH_AR(1)_prop_512 pyAMH_AR(4)_prop_128 pyAMH_AR(4)_prop_256 pyAMH_AR(4)_prop_512 r_prop_128 r_prop_256 r_prop_512'
np.savetxt(f'sim_res/{rd}prop.txt', [prop,prop_t,propMH,propMH_t,prop_r], header=col_names)

col_names = 'pyAMH_AR(1)_run_t_128 pyAMH_AR(1)_run_t_256 pyAMH_AR(1)_run_t_512 pyAMH_AR(4)_run_t_128 pyAMH_AR(4)_run_t_256 pyAMH_AR(4)_run_t_512 pyMH_AR(1)_run_t_128 pyMH_AR(1)_run_t_256 pyMH_AR(1)_run_t_512 pyMH_AR(4)_run_t_128 pyMH_AR(4)_run_t_256 pyMH_AR(4)_run_t_512 r_tun_t_128 r_run_t_256 r_run_t_512'
np.savetxt(f'sim_res/{rd}runtime.txt',  [p_t,p_t_t,pMH_t,pMH_t_t,r_t], header=col_names)

with open(f'sim_res/{rd}resultpyAMH_AR(1)_128_256_512.pkl', 'wb') as f:
    pickle.dump(result, f)

with open(f'sim_res/{rd}resultpyAMH_AR(4)_128_256_512.pkl', 'wb') as f:
    pickle.dump(result_t, f)

with open(f'sim_res/{rd}resultpyMH_AR(1)_128_256_512.pkl', 'wb') as f:
    pickle.dump(resultMH, f)

with open(f'sim_res/{rd}resultpyMH_AR(4)_128_256_512.pkl', 'wb') as f:
    pickle.dump(resultMH_t, f)

with open(f'sim_res/{rd}resultr_128_256_512.pkl', 'wb') as f:
    pickle.dump(result_r, f)


