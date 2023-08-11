# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:24:44 2023

@author: Anandita
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime, sys, os, time
from collections import defaultdict

gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import sphere_functions as sf
import pca_lvm_fns as plf
import figure_fns as ff

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')


#loading tensor product of 1d pca data
f1=os.path.abspath('..')
x1=f1+'data/Translation symmetric tuning/'

d_list=[2,3,4]
period=1

eigenval_d_median=defaultdict(list)
eigenval_d_lower=defaultdict(list)
eigenval_d_upper=defaultdict(list)
N_list_by_d=defaultdict(list)
tensor_prod_eigenvals_theory=defaultdict(list)

eigenvals_tensor_d_median=defaultdict(list)
eigenvals_tensor_d_lower=defaultdict(list)
eigenvals_tensor_d_upper=defaultdict(list)


for d in d_list:
    fw=open(x1+'/circular_uniform_data_d={}'.format(d),'rb')
    data_d=pickle.load(fw)
    fw.close()

    sigma_list=data_d['sigma_list']
    N_list=data_d['total_neurons']
    N_list_by_d[d]=N_list
    for sigma in sigma_list:
        for N in N_list:
            eigenval_d_median[(d,sigma,N)]=data_d['eigenvals_d_sim_median'][(sigma,N)][0]
            eigenval_d_lower[(d,sigma,N)]=data_d['eigenvals_d_sim_lower'][(sigma,N)][0]
            eigenval_d_upper[(d,sigma,N)]=data_d['eigenvals_d_sim_upper'][(sigma,N)][0]
            
            tensor_prod_eigenvals_theory[(d,sigma,N)]=data_d['eigenvals_tensor_theory'][(sigma,N)][0]
        
            eigenvals_tensor_d_median[(d,sigma,N)]=data_d['eigenvals_tensor_d_median'][(sigma,N)][0]
            eigenvals_tensor_d_lower[(d,sigma,N)]=data_d['eigenvals_tensor_d_lower'][(sigma,N)][0]
            eigenvals_tensor_d_upper[(d,sigma,N)]=data_d['eigenvals_tensor_d_upper'][(sigma,N)][0]
            
            
            

# This calculates the radius of sphere with volume x
x1=np.arange(1,1001)


radius_given_volume=defaultdict(list)
eigenvalues_for_given_radius=defaultdict(list)

#calculating the smooth interpolation for eigenvalues
for d in d_list:
    for sigma in sigma_list:
        # radius of sphere with volume values in x1
        radius_given_volume[(d,sigma)]=sf.radius_n_sphere(x1,d)
        #eigenvalues corresponding to these eigenvalues
        eigenvalues_for_given_radius[(d,sigma)]=np.exp((-(2*np.pi*sigma/period)**2)*radius_given_volume[(d,sigma)]**2)
        
        
        
#plotting pc vs variance for different d values, sigma=0.2
d_col_dict=ff.get_cols_for_list(d_list,'Blues',x0=0.75,x1=1.0)
fig, ax = plt.subplots(figsize=(8,6))
for d in d_list:
    N=N_list_by_d[-1]
    sigma=0.2
    plt.plot(x1,eigenval_d_median[(d,sigma,N)][0:100],'o',color=d_col_dict[d],label='d dim sim,d={}'.format(d))

    plt.plot(x1,eigenvals_tensor_d_median[(d,sigma,N)][0:100],color=d_col_dict[d],linewidth=4,alpha=0.4,
                                         label='tensorprod_1d_sim,d={}'.format(d))

    plt.plot(x1,eigenvalues_for_given_radius[(d,sigma)][0:100],linewidth=3,alpha=0.5,color=d_col_dict[d])



#plt.legend()
plt.xlabel("PC #")
plt.ylabel('Variance')
ax.tick_params(width=1,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([0.0,0.1,0.2])
#ax.set_yticklabels([])
ax.set_xticks([1,50,100])
#ax.set_xticklabels([])
#plt.savefig(f1+'plots/Translation symmetric tuning'+curr_date+'/Variance_vs_PC#_different_d')



# same as above with loglog and normalising eigenvalues by their sums,
#sigma =0.2

fig, ax = plt.subplots(figsize=(8,6))
for d in d_list:
    N=N_list_by_d[-1]
    sigma=sigma_list[1]
    s1=np.sum(eigenval_d_median[(d,sigma,N)])
    plt.plot(x1,eigenval_d_median[(d,sigma,N)][0:100]/s1,'o',color=d_col_dict[d],label='d dim sim,d={}'.format(d))
    s2=np.sum(eigenval_d_lower[(d,sigma,N)][0:100])
    s3=np.sum(eigenval_d_upper[(d,sigma,N)][0:100])
    plt.fill_between(x1,eigenval_d_lower[(d,sigma,N)][0:100]/s2,eigenval_d_upper[(d,sigma,N)][0:100]/s3,
                                        alpha=0.4,color=d_col_dict[d])
    s4=np.sum(eigenvals_tensor_d_median[(d,sigma,N)])
    plt.plot(x1,eigenvals_tensor_d_median[(d,sigma,N)][0:100]/s4,color=d_col_dict[d],linewidth=1,
                                         label='tensorprod_1d_sim,d={}'.format(d))

    s5=np.sum(eigenvalues_for_given_radius[(d,sigma)])
    plt.plot(x1,eigenvalues_for_given_radius[(d,sigma)][0:100]/s5,'*',color=d_col_dict[d],linewidth=3,alpha=0.5,label='volume_fit')

plt.yscale('log')
plt.xscale('log')
plt.ylim(top=1,bottom=10**(-8))
plt.legend()
plt.xlabel("PC #")
plt.ylabel('Variance')
ax.tick_params(width=1,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#set labels=False if you
ff.ref_axes(ax,labels=True)
#enter path for saving figure here
#plt.savefig(f1+'plots/Translation symmetric tuning/'+curr_date+'/Variance_vs_pc_different_d_1.pdf',transparent=True)


