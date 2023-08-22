# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:02:04 2022

@author: Anandita
"""


import datetime, sys, os, time
from collections import defaultdict

import numpy as np
from scipy import linalg as sla
import matplotlib.pyplot as plt
import pickle


gen_fn_dir = os.path.abspath('..') + '/shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf
import figure_fns as ff
import one_d_tuning_curves as tc

#load and plot sigmoidal response data
curr_date=datetime.datetime.now().strftime('%Y_%m_%d')

f1=os.path.abspath('..')
a=open(f1+'data/Multiplicative tuning/'+curr_date+'/product_gaussian_varying_width_response','rb')
gaussian_varying_width_product=pickle.load(a)
a.close()

d_list=np.array([2,3,4,5,6,7,8])

#loading eigenvalues
#normalized eigenvals from d dimensional sim
eigenvals_d_median_frac=gaussian_varying_width_product['eigenvals_d_median_frac']
eigenvals_d_lower_frac=gaussian_varying_width_product['eigenvals_d_lower_frac']
eigenvals_d_upper_frac=gaussian_varying_width_product['eigenvals_d_upper_frac']

#normalized tensor product of 1d eigenvalues
eigenvals_tensor_d_median_frac=gaussian_varying_width_product['eigenvals_tensor_d_median_frac']
eigenvals_tensor_d_lower_frac=gaussian_varying_width_product['eigenvals_tensor_d_lower_frac']
eigenvals_tensor_d_upper_frac=gaussian_varying_width_product['eigenvals_tensor_d_upper_frac']

#loading linear dims
linear_dims_tensor_prod_lower=gaussian_varying_width_product['linear_dims_tensor_prod_lower']
linear_dims_tensor_prod_median=gaussian_varying_width_product['linear_dims_tensor_prod_median']
linear_dims_tensor_prod_upper=gaussian_varying_width_product['linear_dims_tensor_prod_upper']

linear_dims_tensor_prod_lower_array=np.array(list(linear_dims_tensor_prod_lower.values()))
linear_dims_tensor_prod_median_array=np.array(list(linear_dims_tensor_prod_median.values()))
linear_dims_tensor_prod_upper_array=np.array(list(linear_dims_tensor_prod_upper.values()))

yerr=(linear_dims_tensor_prod_upper_array-linear_dims_tensor_prod_lower_array)/2

#1d eigenvalues
eigenvals_1d_median=gaussian_varying_width_product['eigenvals_1d']
normalised_eigenvalues=eigenvals_1d_median/np.sum(eigenvals_1d_median)

entropy_norm_eigvals=-np.sum(np.multiply(normalised_eigenvalues,np.log2(normalised_eigenvalues)))

delta=0.05

lower_bound=(2**((entropy_norm_eigvals-delta)*d_list))

d_list_1=np.array([2,3,4])
d_col_dict=ff.get_cols_for_list(d_list_1,'Blues',x0=0.8,x1=1.0)

#plotting eigenvalues for d=2,3,4
x1=np.arange(1,51)
fig,ax=plt.subplots()
for d in d_list_1:

    plt.plot(x1,eigenvals_d_median_frac[d][0:50],'.',color=d_col_dict[d],
                                    label='d dim data,d={}'.format(d))
    plt.fill_between(x1,eigenvals_d_lower_frac[d][0:50],eigenvals_d_upper_frac[d][0:50],
                                          color=d_col_dict[d],alpha=0.4)


    plt.plot(x1,eigenvals_tensor_d_median_frac[d][0:50],label='tensor prod 1d,d={}'.format(d),
                                          color=d_col_dict[d])
    plt.fill_between(x1,eigenvals_tensor_d_lower_frac[d][0:50],
                        eigenvals_tensor_d_upper_frac[d][0:50],color=d_col_dict[d],alpha=0.4)

ax.tick_params(width=1,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yscale('log')
plt.xscale('log')
plt.ylim(top=1,bottom=10**(-3))
plt.xlim(left=1,right=50)
ff.ref_axes(ax,labels=False)
#ax.legend()
#ax.set_xlabel('PC #')
#ax.set_ylabel('Variance explained')
#ax.set_title('Variance explained vs PC# product of gaussian with varying width')
#plt.savefig(f1+'plots/'+curr_date+'/variance_exp_vs_pc_sigmoidal_curves_d=2,3,4.pdf',transparent=True)
plt.show()
plt.close()




#plotting linear dimension
plt.rcParams.update({'font.size': 16})
fig,ax=plt.subplots()
plt.errorbar(d_list,linear_dims_tensor_prod_median_array,yerr,fmt='.',color='k',label='data')

plt.plot(d_list,lower_bound,'-k',label='delta=0.05')


ax.tick_params(width=1,length=10)
#ax.set_xlabel('d')
#ax.set_ylabel('linear dim')
#plt.legend()
plt.yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([2,8])
#plt.ylim(10,10**6)
#ax.set_yticks([10,10**2,10**3])

ff.ref_axes(ax,labels=True)
#plt.savefig(f1+'plots/'+curr_date+'/linear_dim_vs_true_dim_gaussian_varying_widths_cutoff_09.pdf',transparent=True)
#plt.legend()
plt.show()
plt.close()



