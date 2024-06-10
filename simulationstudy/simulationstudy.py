
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
rd = 0  
robjects.r['source']('psd_arma.R')
psd_arma = robjects.globalenv['psd_arma']
mcmcr=importr("psplinePsd")

#AR(4) coefficients
a1=0.9
a2=-0.9
a3=0.9
a4=-0.9
sig=1
arex_r = robjects.FloatVector(np.array([0.9, -0.9, 0.9, -0.9]))
ma_ex = robjects.NA_Integer
#number of data points
n1=128#number of data points
n2=256#number of data points
n3=512#number of data points

#time series AR(4)
random.seed(rd)
series = bnpc.ar4(n1, a1, a2, a3, a4, sig)
series1 =bnpc.ar4(n2, a1, a2, a3, a4, sig)
series2 = bnpc.ar4(n3, a1, a2, a3, a4, sig)
    
#Frequency: [0,pi]
freq =np.pi*( 2 * (np.arange(1, n1 // 2 + 1) - 1) / n1)[1:]
freq1 =np.pi*( 2 * (np.arange(1, n2 // 2 + 1) - 1) / n2)[1:]
freq2 =np.pi*( 2 * (np.arange(1, n3 // 2 + 1) - 1) / n3)[1:]

f_r = robjects.FloatVector(freq)
f_r1= robjects.FloatVector(freq1)
f_r2= robjects.FloatVector(freq2)

#true model
truepsd=np.log(psd_arma(f_r,arex_r,ma_ex,sig))
truepsd1=np.log(psd_arma(f_r1,arex_r,ma_ex,sig))
truepsd2=np.log(psd_arma(f_r2,arex_r,ma_ex,sig))

#parametric model AR(1)
a1p=0.1883
sig2p=7.839
spar=psd_arma(f_r,a1p,ma_ex,sig2p)
spar1=psd_arma(f_r1,a1p,ma_ex,sig2p)
spar2=psd_arma(f_r2,a1p,ma_ex,sig2p)



#MCMC:
k=25
degree=3
iterations=10000
burnin=5000

#n1=128
y_c=(series-np.mean(series))/np.std(series)
        
with np_cv_rules.context():
    s_t_r = time.time()
    result_r= mcmcr.gibbs_pspline(
                    y_c,
                    burnin=burnin,
                    Ntotal=iterations,
                    degree=3,
                    eqSpacedKnots=False,
                    k=k,
                )
    e_t_r = time.time()
    r_t=e_t_r-s_t_r
                
pdgrm=result_r['pdgrm'][1:-1]
stpy=time.time()
result=bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar, modelnum=1,f=freq)
etpy=time.time()
p_t=etpy-stpy


#n2=256
y_c1=(series1-np.mean(series1))/np.std(series1)
                
with np_cv_rules.context():
       s_t_r = time.time()
       result_r1= mcmcr.gibbs_pspline(
                    y_c1,
                    burnin=burnin,
                    Ntotal=iterations,
                    degree=3,
                    eqSpacedKnots=False,
                    k=k,
                )
       e_t_r = time.time()
       r1_t=e_t_r-s_t_r
                
pdgrm=result_r1['pdgrm'][1:-1]
stpy=time.time()
result1=bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar1, modelnum=1,f=freq1)
etpy=time.time()
p1_t=etpy-stpy



#n3=512
y_c2=(series2-np.mean(series2))/np.std(series2)
with np_cv_rules.context():
        s_t_r = time.time()
        result_r2=mcmcr.gibbs_pspline(
                    y_c2,
                    burnin=burnin,
                    Ntotal=iterations,
                    degree=3,
                    eqSpacedKnots=False,
                    k=k,
                )
        e_t_r = time.time()
        r2_t=e_t_r-s_t_r
pdgrm=result_r2['pdgrm'][1:-1]
stpy=time.time()
result2=bnpc.mcmc(pdgrm=pdgrm, n=iterations, k=k, burnin=burnin, Spar=spar2, modelnum=1,f=freq2)
etpy=time.time()
p2_t=etpy-stpy



####Calculating IAE and prop

#n=128
ci_py = bnpc.compute_ci(result['psd']+2*np.log(np.std(series)))
ci_r = bnpc.compute_ci((np.log(result_r['fpsd.sample'] * np.std(series) ** 2)[1:-1]).T)
##IAE
iae_values=bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd),n1)
iae_values_r=bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd),n1)
        
##prop 
prop=bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd)        
prop_r=bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd)
   

#n=256
ci_py = bnpc.compute_ci(result1['psd']+2*np.log(np.std(series1)))
ci_r = bnpc.compute_ci((np.log(result_r1['fpsd.sample'] * np.std(series1) ** 2)[1:-1]).T)
        
        
##IAE
iae1_values=bnpc.compute_iae(np.exp(ci_py.med),np.exp(truepsd1),n2)
iae1_values_r=bnpc.compute_iae(np.exp(ci_r.med),np.exp(truepsd1),n2)

##prop 
prop1=bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd1)        
prop1_r=bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd1)


#n=512
ci_py =bnpc.compute_ci(result2['psd']+2*np.log(np.std(series2)))
ci_r = bnpc.compute_ci((np.log(result_r2['fpsd.sample'] * np.std(series2) ** 2)[1:-1]).T)
        
        
##IAE
iae2_values=compute_iae(np.exp(ci_py.med),np.exp(truepsd2),n3)
iae2_values_r=compute_iae(np.exp(ci_r.med),np.exp(truepsd2),n3)

##prop 
prop2=bnpc.compute_prop(ci_py.u05,ci_py.u95,truepsd2)        
prop2_r=bnpc.compute_prop(ci_r.u05,ci_r.u95,truepsd2)




#Outputs:     
iae = np.column_stack((iae_values, iae1_values, iae2_values, iae_values_r, iae1_values_r, iae2_values_r))
col_names = 'py_iae_128 py_iae_256 py_iae_512 r_iae_128 r_iae_256 r_iae_512'
np.savetxt(f'{rd}iae.txt', iae, header=col_names)

propor = np.column_stack((prop, prop1, prop2, prop_r, prop1_r, prop2_r))
col_names = 'py_prop_128 py_prop_256 py_prop_512 r_prop_128 r_prop_256 r_prop_512'
np.savetxt(f'{rd}prop.txt', propor, header=col_names)

run_t = np.column_stack((p_t, p1_t, p2_t, r_t, r1_t, r2_t))
col_names = 'py_run_t_128 py_run_t_256 py_run_t_512 r_tun_t_128 r_run_t_256 r_run_t_512'
np.savetxt(f'{rd}runtime.txt', run_t, header=col_names)

with open(f'{rd}resultpy_128_256_512.pkl', 'wb') as f:
     pickle.dump([result,result1,result2], f)

with open(f'{rd}resultr_128_256_512.pkl', 'wb') as f:
    pickle.dump([result_r,result_r1,result_r2], f)


