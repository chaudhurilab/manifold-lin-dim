# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:58:05 2023

@author: Anandita
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime, sys, os, time
from collections import defaultdict

gen_fn_dir = os.path.abspath('..') + '/shared_scripts/'
sys.path.append(gen_fn_dir)

import sphere_functions as sf
import pca_lvm_fns as plf
import figure_fns as ff

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')


#loading tensor product of 1d pca data
f1=os.path.abspath('..')
x1=f1+'data/Translation symmetric tuning/'+curr_date

d_list=[2,3,4,5,6,7,8]
period=1

linear_dims_d_sim_median=defaultdict(list)
linear_dims_d_sim_lower=defaultdict(list)
linear_dims_d_sim_upper=defaultdict(list)

linear_dims_tensor_d_median=defaultdict(list)
linear_dims_tensor_d_lower=defaultdict(list)
linear_dims_tensor_d_upper=defaultdict(list)

linear_dims_tensor_theory=defaultdict(list)

N_list_by_d=defaultdict(list)

for d in d_list:
    fw=open(x1+'/circular_uniform_data_d={}'.format(d),'rb')
    data_d=pickle.load(fw)
    fw.close()

    sigma_list=data_d['sigma_list']
    N_list=data_d['total_neurons']
    N_list_by_d[d]=N_list
    for sigma in sigma_list:
        for N in N_list:
            if d<=4:
                linear_dims_d_sim_median[(d,sigma,N)]=data_d['linear_dims_d_sim_median'][(sigma,N)]
                linear_dims_d_sim_lower[(d,sigma,N)]=data_d['linear_dims_d_sim_lower'][(sigma,N)]
                linear_dims_d_sim_upper[(d,sigma,N)]=data_d['linear_dims_d_sim_upper'][(sigma,N)]
                
            linear_dims_tensor_d_median[(d,sigma,N)]=data_d['linear_dims_tensor_d_median'][(sigma,N)]
            linear_dims_tensor_d_lower[(d,sigma,N)]=data_d['linear_dims_tensor_d_lower'][(sigma,N)]
            linear_dims_tensor_d_upper[(d,sigma,N)]=data_d['linear_dims_tensor_d_upper'][(sigma,N)]

#finding the linear dims from chi distribution
radius_as_by_sigma=defaultdict(list)
radius_as_by_d=defaultdict(list)

linear_dims_as_theory_by_d=defaultdict(list)
linear_dims_as_theory_by_sigma=defaultdict(list)

for sigma in sigma_list:
    for d in d_list:
        sigma_1=2*np.pi*sigma
        #this is the asymptotic mean of  scaled chi distribution( scale= (2^(0.5)*sigma))

        median_as=(1/sigma_1)*np.sqrt((d/2)-(1/3))

        #asymptotic stddev of the chi distribution
        stdev_as=1/(2*sigma_1)

        #asymptotic radius
        #the co-efficient of the std. dev was calculated by fitting a curve
        r= median_as +1.71* stdev_as

        radius_as_by_d[d].append(r)
        radius_as_by_sigma[sigma].append(r)
        linear_dims_as_theory_by_d[d].append(sf.volume_n_sphere(r,d))
        linear_dims_as_theory_by_sigma[sigma].append(sf.volume_n_sphere(r,d))

#lower bounds on linear dims
linear_dims_lb_by_d=defaultdict(list)
linear_dims_lb_by_sigma=defaultdict(list)
#calculating radius as lower bound from median
for sigma in sigma_list:
    for d in d_list:


        #asymptotic radius
        r=(0.8/(np.sqrt(8)*np.pi))* (np.sqrt(d)/sigma)

        radius_as_by_d[d].append(r)
        radius_as_by_sigma[sigma].append(r)
        linear_dims_lb_by_d[d].append(sf.volume_n_sphere(r,d))
        linear_dims_lb_by_sigma[sigma].append(sf.volume_n_sphere(r,d))

        
#fix sigma and N to plot linear dim vs d                
linear_dims_d_sim_median_by_sigma=defaultdict(list)
linear_dims_d_sim_lower_by_sigma=defaultdict(list)
linear_dims_d_sim_upper_by_sigma=defaultdict(list)

linear_dims_tensor_d_median_by_sigma=defaultdict(list)
linear_dims_tensor_d_lower_by_sigma=defaultdict(list)
linear_dims_tensor_d_upper_by_sigma=defaultdict(list)

for d in d_list:
    N=N_list_by_d[d][-1]
    for sigma in sigma_list:
        if d<=4:
            linear_dims_d_sim_median_by_sigma[sigma].append(linear_dims_d_sim_median[(d,sigma,N)])
            linear_dims_d_sim_lower_by_sigma[sigma].append(linear_dims_d_sim_lower[(d,sigma,N)])
            linear_dims_d_sim_upper_by_sigma[sigma].append(linear_dims_d_sim_lower[(d,sigma,N)])
            
        linear_dims_tensor_d_median_by_sigma[sigma].append(linear_dims_tensor_d_median[(d,sigma,N)])
        linear_dims_tensor_d_lower_by_sigma[sigma].append(linear_dims_tensor_d_lower[(d,sigma,N)])
        linear_dims_tensor_d_upper_by_sigma[sigma].append(linear_dims_tensor_d_upper[(d,sigma,N)])
           
                
#plotting linear dim vs d

sigma_col_dict=ff.get_cols_for_list(sigma_list,'Reds')

d_list_1=np.array([2,3,4])

plt.figure()                
for sigma in sigma_list:
    plt.plot(d_list_1,linear_dims_d_sim_median_by_sigma[sigma],'d',color=sigma_col_dict[sigma],
             label='linear dims d sim, sigma={}'.format(sigma))
    plt.fill_between(d_list_1,linear_dims_d_sim_lower_by_sigma[sigma],
                     linear_dims_d_sim_upper_by_sigma[sigma],color=sigma_col_dict[sigma],alpha=0.4)
    
    plt.plot(d_list,linear_dims_tensor_d_median_by_sigma[sigma],'o',color=sigma_col_dict[sigma],
             label='linear dims tensor d, sigma={}'.format(sigma))
    plt.fill_between(d_list,linear_dims_tensor_d_lower_by_sigma[sigma],
                     linear_dims_tensor_d_upper_by_sigma[sigma],color=sigma_col_dict[sigma],alpha=0.4)
   
    plt.plot(d_list,linear_dims_as_theory_by_sigma[sigma],color=sigma_col_dict[sigma]
             ,label='theory')
    
    plt.plot(d_list,linear_dims_lb_by_sigma[sigma],color=sigma_col_dict[sigma]
             ,label='theory')

plt.xlabel('d')
plt.ylabel('Linear dim')
plt.yscale('log')
plt.savefig(f1+'plots/Translation symmetric tuning/'+curr_date+'Linear dim vs d')
plt.show()
plt.close()


#fixing d,N for plotting linear dim vs sigma
linear_dims_d_sim_median_by_d=defaultdict(list)
linear_dims_d_sim_lower_by_d=defaultdict(list)
linear_dims_d_sim_upper_by_d=defaultdict(list)

linear_dims_tensor_d_median_by_d=defaultdict(list)
linear_dims_tensor_d_lower_by_d=defaultdict(list)
linear_dims_tensor_d_upper_by_d=defaultdict(list)

for d in d_list:
    N=N_list_by_d[d][-1]
    for sigma in sigma_list:
        if d<=4:
            linear_dims_d_sim_median_by_d[d].append(linear_dims_d_sim_median[(d,sigma,N)])
            linear_dims_d_sim_lower_by_d[d].append(linear_dims_d_sim_lower[(d,sigma,N)])
            linear_dims_d_sim_upper_by_d[d].append(linear_dims_d_sim_lower[(d,sigma,N)])
            
        linear_dims_tensor_d_median_by_d[d].append(linear_dims_tensor_d_median[(d,sigma,N)])
        linear_dims_tensor_d_lower_by_d[d].append(linear_dims_tensor_d_lower[(d,sigma,N)])
        linear_dims_tensor_d_upper_by_d[d].append(linear_dims_tensor_d_upper[(d,sigma,N)])



#plotting linear dim vs sigma
d_list_1=np.array([2,4])

d_col_dict=ff.get_cols_for_list(sigma_list,'Reds')

plt.figure()
for d in d_list_1:
    if d<=4:
        plt.plot(sigma_list,linear_dims_d_sim_median_by_d[d],'d',color=d_col_dict[d],
             label='linear dims d sim, d={}'.format(d))
    plt.fill_between(d_list,linear_dims_d_sim_lower_by_d[d],
                     linear_dims_d_sim_upper_by_d[d],color=d_col_dict[d],alpha=0.4)
    
    plt.plot(d_list,linear_dims_tensor_d_median_by_d[d],'o',color=d_col_dict[d],
             label='linear dims tensor d, d={}'.format(d))
    plt.fill_between(d_list,linear_dims_tensor_d_lower_by_d[d],
                     linear_dims_tensor_d_upper_by_d[d],color=d_col_dict[d],alpha=0.4)
   
    plt.plot(d_list,linear_dims_as_theory_by_d[d],color=d_col_dict[d]
             ,label='theory')
    
    plt.plot(d_list,linear_dims_lb_by_d[d],color=d_col_dict[d]
             ,label='theory')
    
plt.xlabel('sigma')
plt.ylabel('Linear dim')
plt.yscale('log')
plt.savefig(f1+'plots/Translation symmetric tuning/'+curr_date+'Linear dim vs sigma')
plt.show()
plt.close()

#For plotting for linear dim vs N fix sigma and d
linear_dims_d_sim_median_by_d,sigma=defaultdict(list)
linear_dims_d_sim_lower_by_d,sigma=defaultdict(list)
linear_dims_d_sim_upper_by_d,sigma=defaultdict(list)

linear_dims_tensor_d_median_by_d,sigma=defaultdict(list)
linear_dims_tensor_d_lower_by_d,sigma=defaultdict(list)
linear_dims_tensor_d_upper_by_d,sigma=defaultdict(list)

for d in d_list:
    for sigma in sigma_list:
        if d<=4:
            linear_dims_d_sim_median_by_d,sigma[(d,sigma)].append(linear_dims_d_sim_median[(d,sigma,N)])
            linear_dims_d_sim_lower_by_d,sigma[(d,sigma)].append(linear_dims_d_sim_lower[(d,sigma,N)])
            linear_dims_d_sim_upper_by_sigma[(d,sigma)].append(linear_dims_d_sim_lower[(d,sigma,N)])
            
        linear_dims_tensor_d_median_by_d,sigma[(d,sigma)].append(linear_dims_tensor_d_median[(d,sigma,N)])
        linear_dims_tensor_d_lower_by_d,sigma[(d,sigma)].append(linear_dims_tensor_d_lower[(d,sigma,N)])
        linear_dims_tensor_d_upper_by_d,sigma[(d,sigma)].append(linear_dims_tensor_d_upper[(d,sigma,N)])

#plotting linear dim vs N

#put the d you want to plot it for here
d=2
N_list=N_list_by_d[d]

for sigma in sigma_list:
    if d<=4:
        plt.plot(N_list,linear_dims_d_sim_median_by_d,sigma[(d,sigma)],'d',color=sigma_col_dict[sigma],
             label='linear dims d sim, sigma={}'.format(sigma))
    plt.fill_between(N_list,linear_dims_d_sim_lower_by_d,sigma[(d,sigma)],
                     linear_dims_d_sim_upper_by_d,sigma[(d,sigma)],color=sigma_col_dict[sigma],alpha=0.4)
    
    plt.plot(N_list,linear_dims_tensor_d_median_by_d,sigma[(d,sigma)],'o',color=sigma_col_dict[sigma],
             label='linear dims tensor d, sigma={}'.format(sigma))
    plt.fill_between(N_list,linear_dims_tensor_d_lower_by_d,sigma[(d,sigma)],
                     linear_dims_tensor_d_upper_by_d,sigma[(d,sigma)],color=sigma_col_dict[(d,sigma)],alpha=0.4)
plt.legend()
plt.xlabel('N')
plt.ylabel('Linear dim')
plt.title('Linear dim vs N,d={}'.format(d))
plt.savefig(f1+'plots/Translation symmetric tuning/'+curr_date+'Linear dim vs N,d={}'.format(d))
plt.show()
plt.close()  
    




    
                
                
                
                