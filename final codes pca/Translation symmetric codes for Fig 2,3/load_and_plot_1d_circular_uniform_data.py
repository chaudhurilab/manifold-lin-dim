# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:48:30 2021

@author: Anandita
"""
import datetime, sys, os, time
from collections import defaultdict

import numpy as np
from scipy import linalg as sla
import matplotlib.pyplot as plt
import pickle


gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf
import figure_fns as ff
#from general_file_fns import load_pickle_file, save_pickle_file

curr_seed=int((time.time()%1)*(2**31))
np.random.seed(curr_seed)
print('Seed= {}'.format(curr_seed))

#put date of the folder in which data was saved
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')



#opening the file where the data is saved. This can be changed according 
#to the path where you savethe data
f1=os.path.abspath('..') 
fw=open(f1+'/data/Translation symmetric tuning/1d_data_uniform_circular_bc','rb')
uniform_circular_data=pickle.load(fw)
fw.close()

sigma_list=uniform_circular_data['sigma_list']
N_list=uniform_circular_data['n_neurons_list']

sigma_list=np.sort(sigma_list)

linear_dim_lower_by_N=uniform_circular_data['linear_dim_lower_by_N']
linear_dim_median_by_N=uniform_circular_data['linear_dim_median_by_N']
linear_dim_upper_by_N=uniform_circular_data['linear_dim_upper_by_N']
linear_dim_lower_by_sigma=uniform_circular_data['linear_dim_lower_by_sigma']
linear_dim_median_by_sigma=uniform_circular_data['linear_dim_median_by_sigma']
linear_dim_upper_by_sigma=uniform_circular_data['linear_dim_upper_by_sigma']

eigenvals_lower=uniform_circular_data['eigenvals_lower']
eigenvals_median=uniform_circular_data['eigenvals_median']
eigenvals_upper=uniform_circular_data['eigenvals_upper']
linear_dims_theory=uniform_circular_data['linear_dims_theory']

# This has been changed to eigenvalues theory in the current
# version of the file
eigenvals_theory_dict=uniform_circular_data['eigenvals_theory']
period=uniform_circular_data['period']







plt.rcParams.update({'font.size': 16})

# Dictionaries for colors. The different sigmas will be in red and the different N's in blue
col_dict_sigma = ff.get_cols_for_list(sigma_list, 'Reds')
col_dict_N = ff.get_cols_for_list(N_list, 'Blues')

#selecting a few sigma

sigma_list_1=np.array([0.05,0.1,0.15,0.2])

#Plotting linear dimensions vs N
fig, ax = plt.subplots(figsize=(8,6))
for sigma in sigma_list_1:
    ax.plot(N_list,np.asarray(linear_dim_median_by_sigma[sigma]),'o', color=col_dict_sigma[sigma],
                label=sigma,markersize=8)
    ax.fill_between(N_list,linear_dim_lower_by_sigma[sigma],linear_dim_upper_by_sigma[sigma],
      color=col_dict_sigma[sigma],alpha=0.4)

ax.legend()
ax.set_xlabel('N')
ax.set_ylabel('Linear dim')
for axis in ['bottom','left']:
  ax.spines[axis].set_linewidth(1)

ax.set_ylim([0 ,35])

ax.set_xticks([1, 25, 50])
ax.set_yticks([1,5,10,15])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(width=2,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig('1d lvm plots/figure 2/'+'Linear_dim_vs_N_1d_lvm.pdf',transparent=True)


#Choosing N from n_neurons list
N=N_list[-1]

#plotting sigma vs Linear dimensions from simulations and theory. Dots for simulation and solid lines for theory
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(sigma_list,linear_dim_median_by_N[N],'ok',markersize=8,label='simulation')

#ax.fill_between(sigma_list,linear_dim_lower_by_N[N],linear_dim_upper_by_N[N],
      #color='dimgray',alpha=0.4)
ax.plot(sigma_list,np.array(list(linear_dims_theory.values())),'k-',linewidth=2,label='theory')
#ax.set_yticks([0,10])
#ax.set_yticklabels([])
ax.set_xticks([0.05,0.1,0.15,0.2])
#ax.set_xticklabels([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for axis in ['bottom','left']:
  ax.spines[axis].set_linewidth(1)
ax.tick_params(width=1,length=6)
#uncomment below if you want to set the labels and see the legend 
#ax.legend()
#ax.set_xlabel('Sigma')
#ax.set_ylabel('Linear dim')

#uncomment if you want to save figure
#plt.savefig('1d lvm plots/figure 2/'+'linear_dim_vs_sigma_1d_noncircular_uniform.pdf',transparent=True)





#Plotting the first 15 eigenvalues of the covariance matrix for different values of sigma
fig,ax=plt.subplots(figsize=(8,6))
for sigma in sigma_list_1:
    eigenvals_lower_norm=eigenvals_lower[(sigma,N)]/np.sum(eigenvals_lower[(sigma,N)])
    eigenvals_median_norm=eigenvals_median[(sigma,N)]/np.sum(eigenvals_median[(sigma,N)])
    eigenvals_upper_norm=eigenvals_upper[(sigma,N)]/np.sum(eigenvals_upper[(sigma,N)])
    eigenvals_theory=eigenvals_theory_dict[(sigma,N)]
    eigenvals_theory=eigenvals_theory/np.sum(eigenvals_theory)
    ax.plot(np.arange(1,16),eigenvals_median_norm[0:15],'o',label=np.round(sigma,2),color=col_dict_sigma[sigma],markersize=8)
    ax.plot(np.arange(1,16),eigenvals_theory[0:15],color=col_dict_sigma[sigma],linewidth=2)
    ax.fill_between(np.arange(1,16),eigenvals_lower_norm[0:15],eigenvals_upper_norm[0:15],color=col_dict_sigma[sigma],alpha=0.4)
#ax.legend()
for axis in ['bottom','left']:
  ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2,length=6)
ax.set_xticks([5,10,15])
#ax.set_xticklabels([])
ax.set_yticks([0.0,0.5])
#ax.set_yticklabels([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel('PCA component')
ax.set_ylabel('Variance explained')
# plt.savefig('1d lvm plots/figure 2/'+'Variance_explained_1d_lvm_circular_scatter.pdf',transparent=True)










