# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:40:53 2023

@author: Anandita
"""

import sys
import datetime, sys, os, time
from collections import defaultdict
from scipy.special import gamma
from scipy import linalg as sla
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg as sla
#import matplotlib.pyplot as plt
import pickle

gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf

curr_date='2023_06_19'

d_list=np.array([1,2,3,4])
circular_uniform_tc_data=defaultdict(list)
circular_random_tc_data=defaultdict(list)
noncircular_random_tc_data=defaultdict(list)
noncircular_uniform_tc_data=defaultdict(list)

for d in d_list:
    f1=open('circular_multid_data/'+curr_date+'/linear_dims_data_random_centers_d={}'.format(d),'rb')
    circular_random_tc_data[d]=pickle.load(f1)
    f1.close()
    
    f1=open('circular_multid_data/'+curr_date+'/linear_dims_data_uniform_centers_d={}'.format(d),'rb')
    circular_uniform_tc_data[d]=pickle.load(f1)
    f1.close()
    

    f2=open('non_circular_multid_data/'+curr_date+'/linear_dims_data_random_centers_d={}'.format(d),'rb')
    noncircular_random_tc_data[d]=pickle.load(f2)
    f2.close()

    f3= open('non_circular_multid_data/'+curr_date+'/linear_dims_data_uniform_centers_d={}'.format(d),'rb')
    noncircular_uniform_tc_data[d]=pickle.load(f3)
    f3.close()


sigma_list=circular_random_tc_data[1]['sigma_list']

linear_dim_circular_random=defaultdict(list)
linear_dim_noncircular_random=defaultdict(list)
linear_dim_noncircular_uniform=defaultdict(list)
linear_dim_circular_uniform=defaultdict(list)

for d in d_list:
    for sigma in sigma_list:
        N=circular_random_tc_data[d]['total_neurons_list'][0]
        k=(sigma,N)
        circular_random=circular_random_tc_data[d]['linear_dims_d_sim'][k]
        circular_random.extend(np.array(circular_random)+1)
        linear_dim_circular_random[(d,sigma)]=circular_random
        
        N=circular_uniform_tc_data[d]['total_neurons'][0]
        k=(sigma,N)
        circular_uniform=circular_uniform_tc_data[d]['linear_dims_tensor_cov_1d'][k]
        circular_uniform.extend(np.array(circular_uniform)+1)
        linear_dim_circular_uniform[(d,sigma)]=circular_uniform
        
        N=noncircular_random_tc_data[d]['total_neurons'][0]
        k=(sigma,N)
        noncircular_random=noncircular_random_tc_data[d]['linear_dims_d_sim'][k]
        noncircular_random.extend(np.array(noncircular_random)+1)
        linear_dim_noncircular_random[(d,sigma)]=noncircular_random
        
        N=noncircular_uniform_tc_data[d]['total_neurons'][0]
        k=(sigma,N)
        noncircular_uniform=noncircular_uniform_tc_data[d]['linear_dims_tensor_cov_1d'][k]
        noncircular_uniform.extend(np.array(noncircular_uniform)+1)
        linear_dim_noncircular_uniform[(d,sigma)]=noncircular_uniform
        
        
        
        
        
        

linear_dim_circular_random_median_by_sigma=defaultdict(list)
linear_dim_circular_random_lower_by_sigma=defaultdict(list)
linear_dim_circular_random_upper_by_sigma=defaultdict(list)

linear_dim_noncircular_random_median_by_sigma=defaultdict(list)
linear_dim_noncircular_random_lower_by_sigma=defaultdict(list)
linear_dim_noncircular_random_upper_by_sigma=defaultdict(list)

linear_dim_noncircular_uniform_median_by_sigma=defaultdict(list)
linear_dim_noncircular_uniform_lower_by_sigma=defaultdict(list)
linear_dim_noncircular_uniform_upper_by_sigma=defaultdict(list)

linear_dim_circular_uniform_median_by_sigma=defaultdict(list)
linear_dim_circular_uniform_lower_by_sigma=defaultdict(list)
linear_dim_circular_uniform_upper_by_sigma=defaultdict(list)



for sigma in sigma_list:
    for d in d_list:
       k=(d,sigma)
       linear_dim_circular_random_median_by_sigma[sigma].append(np.median(linear_dim_circular_random[k]))
       linear_dim_circular_random_lower_by_sigma[sigma].append(np.percentile(linear_dim_circular_random[k],10))
       linear_dim_circular_random_upper_by_sigma[sigma].append(np.percentile(linear_dim_circular_random[k],90))

       linear_dim_noncircular_random_median_by_sigma[sigma].append(np.median(linear_dim_noncircular_random[k]))
       linear_dim_noncircular_random_lower_by_sigma[sigma].append(np.percentile(linear_dim_noncircular_random[k],10))
       linear_dim_noncircular_random_upper_by_sigma[sigma].append(np.percentile(linear_dim_noncircular_random[k],90))

       linear_dim_noncircular_uniform_median_by_sigma[sigma].append(np.median(linear_dim_noncircular_uniform[k]))
       linear_dim_noncircular_uniform_lower_by_sigma[sigma].append(np.percentile(linear_dim_noncircular_uniform[k],10))
       linear_dim_noncircular_uniform_upper_by_sigma[sigma].append(np.percentile(linear_dim_noncircular_uniform[k],90))

       linear_dim_circular_uniform_median_by_sigma[sigma].append(np.median(linear_dim_circular_uniform[k]))
       linear_dim_circular_uniform_lower_by_sigma[sigma].append(np.percentile(linear_dim_circular_uniform[k],10))
       linear_dim_circular_uniform_upper_by_sigma[sigma].append(np.percentile(linear_dim_circular_random[k],90))






circular_random_error=defaultdict(list)
noncircular_random_error=defaultdict(list)
noncircular_uniform_error=defaultdict(list)

for sigma in sigma_list:
    circular_random_error[sigma]=np.vstack((linear_dim_circular_random_lower_by_sigma[sigma],
                                            linear_dim_circular_random_upper_by_sigma[sigma]))
    noncircular_random_error[sigma]=np.vstack((linear_dim_noncircular_random_lower_by_sigma[sigma],
                                               linear_dim_noncircular_random_upper_by_sigma[sigma]))
    noncircular_uniform_error[sigma]=np.vstack((linear_dim_noncircular_uniform_lower_by_sigma[sigma],
                                               linear_dim_noncircular_uniform_upper_by_sigma[sigma]))

plt.rcParams.update({'font.size':14})

sigma=sigma_list[0]
#plots with errorbars

# plt.figure()
# plt.errorbar(d_list,linear_dim_circular_random_median_by_sigma[sigma],yerr=circular_random_error[sigma],fmt='o',
#              color='blue',label='ciruclar random')
# # plt.fill_between(d_list,linear_dim_circular_random_lower_by_sigma[sigma],
# #                   linear_dim_circular_random_upper_by_sigma[sigma],alpha=0.4,
# #                   color='blue')
# plt.errorbar(d_list,linear_dim_noncircular_random_median_by_sigma[sigma],yerr=noncircular_random_error[sigma],fmt='o',color='magenta',
#           label='nonciruclar random')
# # plt.fill_between(d_list,linear_dim_noncircular_random_lower_by_sigma[sigma],
# #                   linear_dim_noncircular_random_upper_by_sigma[sigma],alpha=0.4,
# #                   color='magenta')

# plt.errorbar(d_list,linear_dim_noncircular_uniform_median_by_sigma[sigma],yerr=noncircular_uniform_error[sigma],fmt='o',color='limegreen'
#           ,label='nonciruclar uniform')
# # plt.fill_between(d_list,linear_dim_noncircular_uniform_lower_by_sigma[sigma],
# #                   linear_dim_noncircular_uniform_upper_by_sigma[sigma],alpha=0.4,
# #                   color='limegreen')

# plt.plot(d_list,linear_dim_circular_uniform_median_by_sigma[sigma],marker='o',color='firebrick',
#           label='circular uniform')



# plt.legend()
# plt.yscale('log')
# plt.xlabel('d')
# plt.ylabel('Linear dim')
# plt.title('Linear dim vs d, sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/Linear_dim_vs_d_sigma=01_lines_with_error_bars')
# plt.show()
# plt.close()

import matplotlib as mpl  
mpl.rc('font',family='Arial')

plt.rcParams.update({'font.size':16})

hfont = {'fontname':'Arial'}

sigma=sigma_list[0]
#lines without error bars
plt.figure(figsize=(8,6))
plt.plot(d_list,linear_dim_circular_random_median_by_sigma[sigma],'o',
             color='blue',label='circular random')
plt.fill_between(d_list,linear_dim_circular_random_lower_by_sigma[sigma],
                  linear_dim_circular_random_upper_by_sigma[sigma],alpha=0.4,
                  color='blue')
plt.plot(d_list,linear_dim_noncircular_random_median_by_sigma[sigma],'o',color='magenta',
          label='noncircular random')
plt.fill_between(d_list,linear_dim_noncircular_random_lower_by_sigma[sigma],
                  linear_dim_noncircular_random_upper_by_sigma[sigma],alpha=0.4,
                  color='magenta')

plt.plot(d_list,linear_dim_noncircular_uniform_median_by_sigma[sigma],'o',color='limegreen'
          ,label='noncircuclar uniform')
plt.fill_between(d_list,linear_dim_noncircular_uniform_lower_by_sigma[sigma],
                  linear_dim_noncircular_uniform_upper_by_sigma[sigma],alpha=0.4,
                  color='limegreen')

plt.plot(d_list,linear_dim_circular_uniform_median_by_sigma[sigma],marker='o',color='firebrick',
          label='circular uniform')
plt.fill_between(d_list,linear_dim_circular_uniform_lower_by_sigma[sigma],
                  linear_dim_circular_uniform_upper_by_sigma[sigma],alpha=0.4,
                  color='firebrick')

plt.legend()
plt.xticks([1,2,3,4])
plt.yscale('log')
plt.xlabel('d')
plt.ylabel('Linear dimension')
plt.title('sigma={}'.format(sigma))
plt.savefig('plots/2023_06_21/Linear_dim_vs_d_sigma=01_lines.pdf',transparent=True,dpi=288)
plt.show()
plt.close()

# #making bar plots


# plt.figure(figsize=(8,6))
# plt.bar(d_list-0.4,linear_dim_noncircular_random_median_by_sigma[sigma],color='magenta'
#           ,yerr=circular_random_error[sigma],width=0.2,label='noncircuclar random',log=True)

# plt.bar(d_list-0.2,linear_dim_noncircular_uniform_median_by_sigma[sigma],color='limegreen'
#           ,yerr=noncircular_random_error[sigma],width=0.2,label='noncircuclar uniform',log=True)

# plt.bar(d_list,linear_dim_circular_random_median_by_sigma[sigma],color='blue'
#           ,yerr=noncircular_uniform_error[sigma],width=0.2,label='circuclar random',log=True)

# plt.bar(d_list+0.2,linear_dim_circular_uniform_by_sigma[sigma],color='firebrick'
#           ,width=0.2,label='circular uniform',log=True)

# plt.legend()
# #plt.yscale('log')
# plt.xlabel('d')
# plt.ylabel('Linear dim')
# plt.title('Linear dim vs d, sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/Linear_dim_vs_d_sigma=01_bars')
# plt.show()
# plt.close()

sigma=sigma_list[1]

#lines with errorbars
# plt.figure()
# plt.errorbar(d_list,linear_dim_circular_random_median_by_sigma[sigma],yerr=circular_random_error[sigma],fmt='o',
#              color='blue',label='ciruclar random')
# # plt.fill_between(d_list,linear_dim_circular_random_lower_by_sigma[sigma],
# #                   linear_dim_circular_random_upper_by_sigma[sigma],alpha=0.4,
# #                   color='blue')
# plt.errorbar(d_list,linear_dim_noncircular_random_median_by_sigma[sigma],yerr=noncircular_random_error[sigma],fmt='o',color='magenta',
#           label='nonciruclar random')
# # plt.fill_between(d_list,linear_dim_noncircular_random_lower_by_sigma[sigma],
# #                   linear_dim_noncircular_random_upper_by_sigma[sigma],alpha=0.4,
# #                   color='magenta')

# plt.errorbar(d_list,linear_dim_noncircular_uniform_median_by_sigma[sigma],yerr=noncircular_uniform_error[sigma],fmt='o',color='limegreen'
#           ,label='nonciruclar uniform')
# # plt.fill_between(d_list,linear_dim_noncircular_uniform_lower_by_sigma[sigma],
# #                   linear_dim_noncircular_uniform_upper_by_sigma[sigma],alpha=0.4,
# #                   color='limegreen')

# plt.plot(d_list,linear_dim_circular_uniform_by_sigma[sigma],marker='o',color='firebrick',
#           label='circular uniform')

# plt.legend()
# plt.yscale('log')
# plt.xlabel('d')
# plt.ylabel('Linear dim')
# plt.title('Linear dim vs d, sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/Linear_dim_vs_d_sigma=02_lines_with_error_bars')
# plt.show()
# plt.close()


sigma=sigma_list[1]
#lines
plt.figure(figsize=(8,6))
plt.plot(d_list,linear_dim_circular_random_median_by_sigma[sigma],'o',
             color='blue',label='circular random')
plt.fill_between(d_list,linear_dim_circular_random_lower_by_sigma[sigma],
                  linear_dim_circular_random_upper_by_sigma[sigma],alpha=0.4,
                  color='blue')
plt.plot(d_list,linear_dim_noncircular_random_median_by_sigma[sigma],'o',color='magenta',
          label='noncircuclar random')
plt.fill_between(d_list,linear_dim_noncircular_random_lower_by_sigma[sigma],
                  linear_dim_noncircular_random_upper_by_sigma[sigma],alpha=0.4,
                  color='magenta')

plt.plot(d_list,linear_dim_noncircular_uniform_median_by_sigma[sigma],'o',color='limegreen'
          ,label='noncircular uniform')
plt.fill_between(d_list,linear_dim_noncircular_uniform_lower_by_sigma[sigma],
                  linear_dim_noncircular_uniform_upper_by_sigma[sigma],alpha=0.4,
                  color='limegreen')

plt.plot(d_list,linear_dim_circular_uniform_median_by_sigma[sigma],marker='o',color='firebrick',
          label='circular uniform')
plt.fill_between(d_list,linear_dim_circular_uniform_lower_by_sigma[sigma],
                  linear_dim_circular_uniform_upper_by_sigma[sigma],alpha=0.4,
                  color='firebrick')
plt.xticks([1,2,3,4])
plt.legend()
plt.yscale('log')
plt.xlabel('d')
plt.ylabel('Linear dimension')
plt.title(' sigma={}'.format(sigma))
plt.savefig('plots/2023_06_21/Linear_dim_vs_d_sigma=02_lines.pdf',transparent=True,dpi=288)
plt.show()
plt.close()



#making bar plots

# plt.figure(figsize=(8,6))
# plt.bar(d_list-0.4,linear_dim_noncircular_random_median_by_sigma[sigma],color='magenta'
#           ,yerr=circular_random_error[sigma],width=0.2,label='noncircuclar random')

# plt.bar(d_list-0.2,linear_dim_noncircular_uniform_median_by_sigma[sigma],color='limegreen'
#           ,yerr=noncircular_random_error[sigma],width=0.2,label='noncircuclar uniform')

# plt.bar(d_list,linear_dim_circular_random_median_by_sigma[sigma],color='blue'
#           ,yerr=noncircular_uniform_error[sigma],width=0.2,label='circuclar random')

# plt.bar(d_list+0.2,linear_dim_circular_uniform_by_sigma[sigma],color='firebrick'
#           ,width=0.2,label='circular uniform')

# plt.legend()
# plt.yscale('log')
# plt.xlabel('d')
# plt.ylabel('Linear dim')
# plt.title('Linear dim vs d, sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/Linear_dim_vs_d_sigma=02_bars')
# plt.show()
# plt.close()


