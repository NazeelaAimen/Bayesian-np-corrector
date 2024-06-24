
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
import os

# #Random seed:
parser = argparse.ArgumentParser(description='Simulation script with random seed and  ts length')
parser.add_argument('random_seed', type=int, help='Random seed for the simulation')
parser.add_argument('length', type=int, help='ts length for the simulation')
args = parser.parse_args()
rd = args.random_seed
n = args.length

#R
np_cv_rules = default_converter + numpy2ri.converter
robjects.r['source']('psd_arma.R')
psd_arma = robjects.globalenv['psd_arma']
mcmcr=importr("psplinePsd")


#AR(1) coefficients
a1p=0.1883
sig2p=7.839
#AR(4) coefficients
sig=1
arex_r = robjects.FloatVector(np.array([0.9, -0.9, 0.9, -0.9]))
ma_ex = robjects.NA_Integer
#number of data points

random.seed(rd)
series=pd.read_csv(f"ts{n}.csv").iloc[rd,:]
sdser=np.std(series)
freq=np.pi*( 2 * (np.arange(1, n // 2 + 1) - 1) / n)[1:]
f_r=robjects.FloatVector(freq)
truepsd=np.log(psd_arma(f_r,arex_r,ma_ex,sig))
spar=np.log(psd_arma(f_r,a1p,ma_ex,sig2p))


#MCMC:
k=25
degree=3
iterations=200000
burnin=150000
y_c=np.array(bnpc.cent_series(series))
s_t_r = time.time()
with np_cv_rules.context():    
    result_r= mcmcr.gibbs_pspline(
                y_c,
                burnin=50,
                Ntotal=100,
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
result=bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(spar-2*np.log(sdser)), modelnum=1,f=freq)
etpy=time.time()
p_t=etpy-stpy

# AR(4) 
#AMH
stpy_t=time.time()
result_t=bnpc.mcmcAMH(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=np.exp(truepsd-2*np.log(sdser)), modelnum=1,f=freq)
etpy_t=time.time()
p_t_t=etpy_t-stpy_t

####Calculating IAE and prop 
iae=[]
prop=[]      
#CI
ci_py = bnpc.compute_ci(result['psd']+2*np.log(sdser))
ci_py_t = bnpc.compute_ci(result_t['psd']+2*np.log(sdser))
#iae
iae.append(bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd),n))
iae.append(bnpc.compute_iae(np.exp(ci_py_t.med),np.exp(truepsd),n))
#prop
prop.append(bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd))        
prop.append(bnpc.compute_prop(ci_py_t.u05,ci_py_t.u95,truepsd))
    


dire=f'test_sim_{n}'    
if not os.path.exists(dire):
   os.makedirs(dire)


#Outputs:     
col_names = 'pyAMH_AR(1)_iae pyAMH_AR(4)_iae'
np.savetxt(f'sim_res_{n}/{rd}iae.txt', iae, header=col_names)

col_names = 'pyAMH_AR(1)_prop pyAMH_AR(4)_prop'
np.savetxt(f'sim_res_{n}/{rd}prop.txt', prop, header=col_names)

col_names = 'pyAMH_AR(1)_run_t pyAMH_AR(4)_run_t'
np.savetxt(f'sim_res_{n}/{rd}runtime.txt',  [p_t,p_t_t], header=col_names)




