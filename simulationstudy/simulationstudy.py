
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
k=25
degree=3
iterations=10000
burnin=5000
result_r=[]
result=[]
r_t=[]
p_t=[]
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
        stpy=time.time()
        result.append(bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar[i][f'{n[i]}'], modelnum=1,f=freq[i][f'{n[i]}']))
        etpy=time.time()
        p_t.append(etpy-stpy)



####Calculating IAE and prop
iae=[]
iae_r=[]
prop=[]
prop_r=[]
for i in range(0,len(n)):
    ci_py = bnpc.compute_ci(result[i]['psd']+2*np.log(np.std(series[i][f'{n[i]}'])))
    ci_r = bnpc.compute_ci((np.log(result_r[i]['fpsd.sample'] * np.std(series[i][f'{n[i]}']) ** 2)[1:-1]).T)
    iae.append(bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    iae_r.append(bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd[i][f'{n[i]}']),n[i]))
    prop.append(bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd[i][f'{n[i]}']))        
    prop_r.append(bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd[i][f'{n[i]}']))


    



#Outputs:     
col_names = 'py_iae_128 py_iae_256 py_iae_512 r_iae_128 r_iae_256 r_iae_512'
np.savetxt(f'{rd}iae.txt', [iae,iae_r], header=col_names)

col_names = 'py_prop_128 py_prop_256 py_prop_512 r_prop_128 r_prop_256 r_prop_512'
np.savetxt(f'{rd}prop.txt', [prop,prop_r], header=col_names)


col_names = 'py_run_t_128 py_run_t_256 py_run_t_512 r_tun_t_128 r_run_t_256 r_run_t_512'
np.savetxt(f'{rd}runtime.txt',  [p_t,r_t], header=col_names)

with open(f'{rd}resultpy_128_256_512.pkl', 'wb') as f:
     pickle.dump(result, f)

with open(f'{rd}resultr_128_256_512.pkl', 'wb') as f:
    pickle.dump(result_r, f)


