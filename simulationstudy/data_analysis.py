#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:26:07 2024

@author: naim769
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
num = [128, 256, 512]

c_n = [
    'AMH AR(1)',
    'AMH AR(4)',
    'MH AR(1)',
    'MH AR(4)',
    'R'
]


co = ['red', 'blue', 'black', 'green', 'orange']
fig, axes = plt.subplots(len(num), 1, figsize=(12, 10), sharex=True, sharey=True)
handles=[]
iae_med = np.zeros((len(num), len(c_n)))
prop_med = np.zeros((len(num), len(c_n)))

for j, n in enumerate(num):
    f_iae = glob.glob(f'sim_res_{n}/*iae.txt')
    f_prop = glob.glob(f'sim_res_{n}/*prop.txt')
    
    iae = []
    prop = []
    
    
    for i in range(len(f_iae)):
        iae.append(np.loadtxt(f_iae[i], skiprows=1))
        prop.append(np.loadtxt(f_prop[i], skiprows=1))
    
    iae = np.array(iae)
    prop = np.array(prop)
    
    # Plot iae in current subplot
    for i in range(iae.shape[1]):
        sns.kdeplot(iae[:, i], label=c_n[i], color=co[i], ax=axes[j])
        if j == 0:
            handles.append(axes[j].lines[-1])
        axes[j].axvline(np.median(iae[:, i]), linestyle='--', color=co[i])
    med_i=np.median(iae,axis=0)
    med_p=np.median(prop,axis=0)    
    print(f'Median iae for n= {n}\n {c_n} \n {med_i}')   
    print(f'Median prop for n= {n}\n {c_n} \n {med_p}')
    axes[j].set_title(f'n = {n}')
    axes[j].set_xlabel('IAE')
    axes[j].set_ylabel('Density')
    

fig.legend(handles,c_n, loc='upper right', bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.savefig('iae.png', dpi=300, bbox_inches='tight')
plt.show()