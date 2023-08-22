# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:06:41 2023

@author: Anandita
"""

import datetime, sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import matplotlib as mpl  

gen_fn_dir = os.path.abspath('..') + '/shared_scripts/'
sys.path.append(gen_fn_dir)

import figure_fns as ff

d_list=np.array([2,4,6,8])


d_list_1=np.array([2,4])

eigenvals_d_median_by_d=defaultdict(list)
eigenvals_d_lower_by_d=defaultdict(list)
eigenvals_d_upper_by_d=defaultdict(list)

eigenvals_d_median_frac_by_d=defaultdict(list)
eigenvals_d_lower_frac_by_d=defaultdict(list)
eigenvals_d_upper_frac_by_d=defaultdict(list)


eigenvals_tensor_d_median_by_d=defaultdict(list)
eigenvals_tensor_d_lower_by_d=defaultdict(list)
eigenvals_tensor_d_upper_by_d=defaultdict(list)

eigenvals_tensor_d_frac_median_by_d=defaultdict(list)
eigenvals_tensor_d_frac_lower_by_d=defaultdict(list)
eigenvals_tensor_d_frac_upper_by_d=defaultdict(list)


linear_dims_d_median_by_d=defaultdict(list)
linear_dims_d_lower_by_d=defaultdict(list)
linear_dims_d_upper_by_d=defaultdict(list)


linear_dims_tensor_d_median_by_d=defaultdict(list)
linear_dims_tensor_d_lower_by_d=defaultdict(list)
linear_dims_tensor_d_upper_by_d=defaultdict(list)

multiplicative_data=defaultdict(list)

for d in d_list:
    fw=open('data/2023_06_25/product_sigmoid_gaussian_data_d={}'.format(d),'rb')
    multiplicative_data[d]=pickle.load(fw)
    fw.close()
    
    if d<=4:
        eigenvals_d_median_by_d[d]=multiplicative_data[d]['eigenvals_d_median']
        eigenvals_d_lower_by_d[d]=multiplicative_data[d]['eigenvals_d_lower']
        eigenvals_d_upper_by_d[d]=multiplicative_data[d]['eigenvals_d_upper']
        
        eigenvals_d_median_frac_by_d[d]=eigenvals_d_median_by_d[d]/np.sum(eigenvals_d_median_by_d[d])
        eigenvals_d_lower_frac_by_d[d]=eigenvals_d_lower_by_d[d]/np.sum(eigenvals_d_lower_by_d[d])
        eigenvals_d_upper_frac_by_d[d]=eigenvals_d_upper_by_d[d]/np.sum(eigenvals_d_upper_by_d[d])
        
        linear_dims_d_median_by_d[d]=multiplicative_data[d]['linear_dims_d_median'][d]
        linear_dims_d_lower_by_d[d]=multiplicative_data[d]['linear_dims_d_lower'][d]
        linear_dims_d_upper_by_d[d]=multiplicative_data[d]['linear_dims_d_upper'][d]
        
    eigenvals_tensor_d_median_by_d[d]=multiplicative_data[d]['eigenvals_tensor_d_median']
    eigenvals_tensor_d_lower_by_d[d]=multiplicative_data[d]['eigenvals_tensor_d_lower']
    eigenvals_tensor_d_upper_by_d[d]=multiplicative_data[d]['eigenvals_tensor_d_upper']
    
    eigenvals_tensor_d_frac_median_by_d[d]=eigenvals_tensor_d_median_by_d[d]/np.sum(eigenvals_tensor_d_median_by_d[d])
    eigenvals_tensor_d_frac_lower_by_d[d]=eigenvals_tensor_d_lower_by_d[d]/np.sum(eigenvals_tensor_d_lower_by_d[d])
    eigenvals_tensor_d_frac_upper_by_d[d]=eigenvals_tensor_d_upper_by_d[d]/np.sum(eigenvals_tensor_d_upper_by_d[d])
    
    linear_dims_tensor_d_median_by_d[d]=multiplicative_data[d]['linear_dims_tensor_prod_median'][d]
    linear_dims_tensor_d_lower_by_d[d]=multiplicative_data[d]['linear_dims_tensor_prod_lower'][d]
    linear_dims_tensor_d_upper_by_d[d]=multiplicative_data[d]['linear_dims_tensor_prod_upper'][d]


linear_dim_gaussian_1d=multiplicative_data[d]['linear_dims_1d_gaussian']
linear_dim_sigmoid_1d=multiplicative_data[d]['linear_dims_1d_sigmoid']

product_1d_tunings=np.zeros(len(d_list))

for d_idx,d in enumerate(d_list):
    product_1d_tunings[d_idx]=(linear_dim_gaussian_1d**(d/2))*(linear_dim_sigmoid_1d**(d/2))

eigenvals_1d_sigmoid=multiplicative_data[d]['eigenvals_1d_sigmoid_median']
eigenvals_1d_gaussian=multiplicative_data[d]['eigenvals_1d_gaussian_median']

eigenvals_1d_sigmoid=eigenvals_1d_sigmoid/np.sum(eigenvals_1d_sigmoid)
eigenvals_1d_gaussian=eigenvals_1d_gaussian/np.sum(eigenvals_1d_gaussian)

#finding lower bound from entropies
entropy_sigmoid=np.sum(np.multiply(eigenvals_1d_sigmoid,-np.log2(eigenvals_1d_sigmoid)))
entropy_gaussian=np.sum(np.multiply(eigenvals_1d_gaussian,-np.log2(eigenvals_1d_gaussian)))

lower_bound=2**(((entropy_gaussian+entropy_sigmoid-0.05)*(d_list/2)))

#setting some font and size parameters for plots
plt.rcParams.update({'font.size':16})

mpl.rc('font',family='Arial')



#plotting eigenvalues 
plt.figure()
for d in d_list_1:
    x=np.arange(1,len(eigenvals_d_median_by_d[d])+1)
    plt.plot(x[0:20],eigenvals_d_median_frac_by_d[d][0:20],'^',color='crimson',label='eigenvals d sim,d={}'.format(d))
    plt.fill_between(x[0:20],eigenvals_d_lower_frac_by_d[d][0:20],eigenvals_d_upper_frac_by_d[d][0:20],color='crimson',alpha=0.2)
    
    x1=np.arange(1,len(eigenvals_tensor_d_median_by_d[d]+1))
    plt.plot(x1[0:20],eigenvals_tensor_d_frac_median_by_d[d][0:20],'o',color='dodgerblue',label='eigenvals tensor prod,d={}'.format(d))
    plt.fill_between(x1[0:20],eigenvals_tensor_d_frac_lower_by_d[d][0:20],eigenvals_tensor_d_frac_upper_by_d[d][0:20],color='dodgerblue',alpha=0.3)


plt.legend()
plt.xlabel('PC#')
plt.ylabel('Eigenvalues')
#plt.savefig('1d lvm plots/2023_06_26/Eigenvals_vs_PC_product_two_diff_tuning')
plt.show()
plt.close()

d_col_dict=ff.get_cols_for_list(d_list_1,'Blues',x0=0.8,x1=1.0)

#plotting eigenvalues in logscale
plt.figure(figsize=(8,6))
ax=plt.axes()
for d in d_list_1:
    x=np.arange(1,len(eigenvals_d_median_by_d[d])+1)
    plt.plot(x[0:50],eigenvals_d_median_frac_by_d[d][0:50],'.',color=d_col_dict[d],
             label='eigenvals d sim,d={}'.format(d))
    # plt.fill_between(x[0:50],eigenvals_d_lower_frac_by_d[d][0:50],eigenvals_d_upper_frac_by_d[d][0:50],
    #                  color='crimson',alpha=0.2)
    
    x1=np.arange(1,len(eigenvals_tensor_d_median_by_d[d]+1))
    plt.plot(x1[0:50],eigenvals_tensor_d_frac_median_by_d[d][0:50],
             color=d_col_dict[d],label='eigenvals tensor prod,d={}'.format(d),alpha=0.6)
    plt.fill_between(x1[0:50],eigenvals_tensor_d_frac_lower_by_d[d][0:50],
                      eigenvals_tensor_d_frac_upper_by_d[d][0:50],color=d_col_dict[d],alpha=0.3)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1,50)
plt.ylim(10**-3,1)
ax.set_xticklabels([])
ax.set_yticklabels([])
ff.ref_axes(ax,labels=False)
#plt.legend()
#plt.xlabel('PC#')
#plt.ylabel('Frac.var.exp')
#ax.tick_params(width=1,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('1d lvm plots/2023_06_26/Eigenvals_vs_PC_product_two_diff_tuning.pdf',transparent=True,dpi=288)
plt.show()
plt.close()


linear_dims_d_median_array=np.array(list(linear_dims_d_median_by_d.values()))
linear_dims_d_lower_array=np.array(list(linear_dims_d_lower_by_d.values()))
linear_dims_d_upper_array=np.array(list(linear_dims_d_upper_by_d.values()))

linear_dims_tensor_median_array=np.array(list(linear_dims_tensor_d_median_by_d.values()))
linear_dims_tensor_lower_array=np.array(list(linear_dims_tensor_d_lower_by_d.values()))
linear_dims_tensor_upper_array=np.array(list(linear_dims_tensor_d_upper_by_d.values()))



linear_dims_d_error=[linear_dims_d_lower_array-linear_dims_d_median_array,
                     linear_dims_d_upper_array-linear_dims_d_median_array]

linear_dims_tensor_error=[linear_dims_tensor_lower_array-linear_dims_tensor_lower_array
                          ,linear_dims_tensor_upper_array-linear_dims_tensor_lower_array]

#plotting linear dimensions
plt.figure(figsize=(8,6))
plt.plot(d_list_1,linear_dims_d_median_array,'d',color='k',label='d sim',zorder=2.5)
plt.fill_between(d_list_1,linear_dims_d_lower_array,linear_dims_d_upper_array,alpha=0.4)

plt.plot(d_list,linear_dims_tensor_median_array,'k',label='tensor d')
plt.fill_between(d_list,linear_dims_tensor_lower_array,linear_dims_tensor_upper_array,
                 alpha=0.4)

plt.plot(d_list,lower_bound,color='k',label='lower bound')
plt.yscale('log')
#plt.plot(d_list,product_1d_tunings,'v',label='product_1d')

plt.legend()
plt.xlabel('d')
plt.ylabel('Linear dim')
#plt.savefig('Linear_dim_vs_d_product_sigmoid_gaussian')
plt.show()
plt.close()

#plotting linear dimension with errorbar
fig,ax=plt.subplots(figsize=(8,6))
ax.errorbar(d_list_1,linear_dims_d_median_array,yerr=linear_dims_d_error,fmt='dk',
            label='d sim',zorder=2.5)


ax.errorbar(d_list,linear_dims_tensor_median_array,yerr=linear_dims_tensor_error,
            fmt='ok',label='tensor d')

plt.plot(d_list,lower_bound,color='k',label='lower bound')
ax.set_yscale('log')
#plt.plot(d_list,product_1d_tunings,'v',label='product_1d')
plt.xticks(d_list)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(width=1,length=6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ff.ref_axes(ax,labels=False)
#plt.legend()
#plt.xlabel('d')
#plt.ylabel('Linear dimension')
plt.savefig('1d lvm plots/2023_06_26/Linear_dim_vs_d_product_sigmoid_gaussian.pdf',transparent=True,dpi=288)
plt.show()
plt.close()

#making heatmap of product of tuning curves
 
n_neurons_per_dim=8
 
sigmoid_centers=np.linspace(0,2,n_neurons_per_dim,endpoint=False)
#slopes of the sigmoids are uniformly distributed between -5 and 5
slopes=np.linspace(-5,5,n_neurons_per_dim)
    
sigma_list=np.linspace(0.1,0.5,n_neurons_per_dim,endpoint=False)   
    
mu=sigmoid_centers[2]

sigma=sigma_list[2]

slope=slopes[2]

x=np.arange(0,2,0.01)
  
y1=1/(1+np.exp(-slope*(x-mu))) 

y2=np.exp(-(x-mu)**2/(2*sigma**2))

plt.figure()
plt.plot(x,y1,'k')
plt.axis('off')
plt.savefig('1d lvm plots/2023_06_26/1d sigmoid.pdf',transparent=True,dpi=288,bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
plt.plot(x,y2,'k')
plt.axis('off')
plt.savefig('1d lvm plots/2023_06_26/1d gaussian.pdf',transparent=True,dpi=288,bbox_inches='tight')
plt.show() 
plt.close()

prod=np.outer(y1,y2)
  
plt.figure(figsize=(6,6))
ax=plt.axes()
plt.imshow(prod)
ax.set_xticks([0,200])
ax.set_yticks([0,200])
ax.tick_params(width=1,length=10)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.savefig('1d lvm plots/2023_06_26/2d tuning.pdf',transparent=True,dpi=288,bbox_inches='tight')
plt.show()
plt.close()






   