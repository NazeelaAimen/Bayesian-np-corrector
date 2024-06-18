
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
#Random seed:
parser = argparse.ArgumentParser(description='Simulation script with random seed.')
parser.add_argument('random_seed', type=int, help='Random seed for the simulation')
args = parser.parse_args()
rd = args.random_seed


#n
parser = argparse.ArgumentParser(description='Simulation script with ts length.')
parser.add_argument('length', type=int, help='ts length for the simulation')
args = parser.parse_args()
n = args.random_seed

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
series=[]
freq=[]
truepsd=[]
spar=[]

random.seed(rd)
series=pd.read_csv(f"ts{n}.csv").iloc[rd]
sdser=np.std(series)
freq=np.pi*( 2 * (np.arange(1, n // 2 + 1) - 1) / n)[1:]
f_r=robjects.FloatVector(freq)
truepsd=np.log(psd_arma(f_r,arex_r,ma_ex,sig))
spar=psd_arma(f_r,a1p,ma_ex,sig2p)


#MCMC:
k=40
degree=3
iterations=10000
burnin=5000
y_c=np.array(bnpc.cent_series(series))
s_t_r = time.time()
with np_cv_rules.context():    
    result_r= mcmcr.gibbs_pspline(
                y_c,
                burnin=burnin,
                Ntotal=iterations,
                degree=3,
                eqSpacedKnots=False,
                k=k,
                printIter =50000
            )
e_t_r = time.time()
r_t=e_t_r-s_t_r
pdgrm=result_r['pdgrm'][1:-1]

#AR(1)
#AMH
stpy=time.time()
result=bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar, modelnum=1,f=freq)
etpy=time.time()
p_t=etpy-stpy

#MH
stpy=time.time()
resultMH=bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar, modelnum=1,f=freq)
etpy=time.time()
pMH_t=etpy-stpy

# AR(4) 
#AMH
stpy_t=time.time()
result_t=bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd-2*np.log(sdser)), modelnum=1,f=freq)
etpy_t=time.time()
p_t_t=etpy_t-stpy_t
# MH
stpy_t=time.time()
resultMH_t=bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd-2*np.log(sdser)), modelnum=1,f=freq)
etpy_t=time.time()
pMH_t_t=etpy_t-stpy_t

####Calculating IAE and prop 
iae=[]
prop=[]      
#CI
ci_py = bnpc.compute_ci(result['psd']+2*np.log(sdser))
ci_py_t = bnpc.compute_ci(result_t['psd']+2*np.log(sdser))
ci_pyMH = bnpc.compute_ci(resultMH['psd']+2*np.log(sdser))
ci_py_tMH = bnpc.compute_ci(resultMH_t['psd']+2*np.log(sdser))
ci_r = bnpc.compute_ci((np.log(result_r['fpsd.sample'] * sdser ** 2)[1:-1]).T)
#iae
iae.append(bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd),n))
iae.append(bnpc.compute_iae(np.exp(ci_py_t.med),np.exp(truepsd),n))
iae.append(bnpc.compute_iae(np.exp(ci_pyMH.med),np.exp(truepsd),n))
iae.append(bnpc.compute_iae(np.exp(ci_py_tMH.med),np.exp(truepsd),n))
iae.append(bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd),n))
#prop
prop.append(bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd))        
prop.append(bnpc.compute_prop(ci_py_t.u05,ci_py_t.u95,truepsd))
prop.append(bnpc.compute_prop(ci_pyMH.u05,ci_pyMH.u95,truepsd))        
prop.append(bnpc.compute_prop(ci_py_tMH.u05,ci_py_tMH.u95,truepsd))
prop.append(bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd))
    
    
# import os
# os.makedirs(f'sim_res_{n}')
# 

#Outputs:     
col_names = 'pyAMH_AR(1)_iae pyAMH_AR(4)_iae pyMH_AR(1)_iae pyMH_AR(4)_iae r_iae'
np.savetxt(f'sim_res/{rd}iae.txt', iae, header=col_names)

col_names = 'pyAMH_AR(1)_prop pyAMH_AR(4)_prop pyMH_AR(1)_prop pyMH_AR(4)_prop r_prop'
np.savetxt(f'sim_res/{rd}prop.txt', prop, header=col_names)

col_names = 'pyAMH_AR(1)_run_t pyAMH_AR(4)_run_t pyMH_AR(1)_run_t pyMH_AR(4) r_tun_t'
np.savetxt(f'sim_res/{rd}runtime.txt',  [p_t,p_t_t,pMH_t,pMH_t_t,r_t], header=col_names)

with open(f'sim_res/{rd}resultpyAMH_AR(1)_{n}.pkl', 'wb') as f:
    pickle.dump(result, f)

with open(f'sim_res/{rd}resultpyAMH_AR(4)_{n}.pkl', 'wb') as f:
    pickle.dump(result_t, f)

with open(f'sim_res/{rd}resultpyMH_AR(1)_{n}.pkl', 'wb') as f:
    pickle.dump(resultMH, f)

with open(f'sim_res/{rd}resultpyMH_AR(4)_{n}.pkl', 'wb') as f:
    pickle.dump(resultMH_t, f)

with open(f'sim_res/{rd}resultr_{n}.pkl', 'wb') as f:
    pickle.dump(result_r, f)


