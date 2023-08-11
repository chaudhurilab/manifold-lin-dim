# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:05:29 2021

@author: Anandita
"""


''' 1. Generate multi-d data with noncircular b.c and lattice tuning curve centers and do pca on it and look at eigenvalues of covariance matrix. 
    2. Generate 1d data and take tensor product of one-d data and look at the eigenvalues of covariance matrix.
    3. Take tensor product of eigenvalues of one-d 
'''

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

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')



def gen_data_1d(fn, n_data, n_neurons,period,tuning_centers='random'):
    # Get TC centers
    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=n_neurons)
    elif tuning_centers=='uniform':
        mu_list=np.linspace(0,period,n_neurons,endpoint=False)

    alpha_vals = np.random.uniform(0,period,size=n_data)
    #alpha_vals=np.linspace(0,period,n_data,endpoint=False)

    data = plf.outer_fn(fn, alpha_vals, mu_list, vectorized='cols')
    return data


def gen_data_multid(fn,n_data, n_neurons,n_dims,period,neurons_per_dimension,tuning_centers='random'):

    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=(n_neurons, n_dims))
    elif tuning_centers=='uniform':
        mu_list=plf.create_d_dim_grid(neurons_per_dimension,period, n_dims)

    #n_data_per_dimension=np.ceil((n_data)**(1/n_dims))
    alpha_vals = np.random.uniform(0,period,size=(n_data, n_dims))
    #alpha_vals=plf.create_d_dim_grid(n_data_per_dimension,period, n_dims)

    data = plf.outer_fn(fn, alpha_vals, mu_list, vectorized='cols')
    return data

#true dimensions
d=4

#period
L=1


#tc widths
#put in sigma list you want here
sigma_list=[0.1,0.2]

#put in neurons per dimension list here
n_neurons_per_dim_list=np.array([10])

total_neurons=n_neurons_per_dim_list**d


#no of runs for each parameter combination
no_of_runs=20

#eigenvalues of covariance matrix form d dimensional simulation
eigenvals_d_sim=defaultdict(list)

#eigenvalues from taking tensor product of eigenvalues from 1d theory
tensor_prod_eigenvals_theory=defaultdict(list)

#eigenvalues from taking tensor product of 1d simulations
eigenvals_tensor_prod_1d_data=defaultdict(list)

#linear dimensions from d dimensional simulation
linear_dims_d_sim=defaultdict(list)

#linear dimension from tensor product of 1d eigenvalues
linear_dims_tensor_cov_1d=defaultdict(list)

#linear dimensions from tensor product of 1d eigenvalues predicted from theory
linear_dims_tensor_1d_theory=defaultdict(list)
#no of data points for d dimensional simulation
n_data_pts=50000

#no of data points for one dimensional simulation
n_data_pts_1d=10000

#set cut-off for finding linear dimensions here
cut_off=0.9


for sigma in sigma_list:
    #d-dimensional response function with circular b.c
    resp_fn_d = lambda alpha, tuning_curve_center: plf.gaussian(
                alpha, tuning_curve_center,np.eye(d)*(sigma**2),norm='distr')

    #1-d response function with noncircular boundary condition
    resp_fn_1d= lambda alpha, tuning_curve_center: plf.gaussian(
              alpha, tuning_curve_center, sigma, norm='distr')

    
    for n_idx,n_neurons in enumerate(total_neurons):
        # each row of this matrix is the eigenvalues of d dimensional 
        #covariance matrix from a single run
        eigenvals_d_matrix=np.zeros((no_of_runs,n_neurons))
        
        #each row of this matrix is tensor product of eigenvalues of a
        #1d covariance matrix
        eigenvals_prod_matrix=np.zeros((no_of_runs,n_neurons))
        n_neurons_per_dim=n_neurons_per_dim_list[n_idx]
        for run_id in range(no_of_runs):
            start=time.time()

            #generating multid data and performing pca
            # for d>4, this part becomes a lot of data in my laptop
            #uncomment below if you want to create d dimensional data
            # if d>1:
            #     data_d=gen_data_multid(resp_fn_d,n_data_pts,n_neurons,d,L,n_neurons_per_dim[n_idx],
            #                 tuning_centers='uniform')
            # elif d==1:
            #     data_d=gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim,L,tuning_centers='uniform')
            # #normalising the data matrix
            # data_d=data_d/np.sqrt(n_data_pts*n_neurons)
            

            # #covariance matrix for d dimensional data
            # cov_d=np.matmul(np.transpose(data_d),data_d)

            # #eigenvalue of covariance matrix
            # eigenvals_cov_d,_=np.linalg.eig(cov_d)
            # eigenvals_cov_d=np.flip(np.sort(eigenvals_cov_d.real))

            # #each row of this matrix contains the eigenvalues for this run
            # eigenvals_d_matrix[run_id]=eigenvals_cov_d



            # generating 1d data and taking tensor product of covariance matrix
            # comment out below if you only want to run for d dimensional data
            data_1d=gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim,L,tuning_centers='uniform')
            data_1d=data_1d/np.sqrt(n_data_pts_1d*n_neurons_per_dim)

            #calculating the 1d covariance matrix
            cov_1d=np.matmul(np.transpose(data_1d),data_1d)

            #calculating eigenvalues of 1d covariance matrix
            eigenvals_1d_sim,_=np.linalg.eig(cov_1d)

            #taking tensor product of eigenvalues
            eigenvals_tensor_d=plf.vector_tens_product(eigenvals_1d_sim.real,d)

            eigenvals_tensor_d=np.flip(np.sort(eigenvals_tensor_d))
            eigenvals_prod_matrix[run_id]=eigenvals_tensor_d



            #linear dims from multi d simulation
            #uncomment if calculating eigenvalues from d dimensional data
            #linear_dims_d_sim[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_cov_d,cut_off))

            # linear dims from tensor product of 1d covariance data matrices
            #comment if not using tensor product of 1d eigenvalues
            linear_dims_tensor_cov_1d[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_tensor_d,cut_off))

        print('Sigma:',sigma ,'Total neurons:',n_neurons,'d',d,'time:',time.time()-start)

        eigenvals_d_sim[(sigma,n_neurons)].append(eigenvals_d_matrix)
        eigenvals_tensor_prod_1d_data[(sigma,n_neurons)].append(eigenvals_prod_matrix)

        #calculating the 1d eigenvalues predicted by theory and taking their tensor product
        n1=int(np.ceil(n_neurons_per_dim/2))
        x1=np.arange(-n1,n1)
        eigenvals_theory_1d=np.exp(-((2*np.pi*sigma/L)**2)*x1**2)
        eigenvals_theory_d=plf.vector_tens_product(eigenvals_theory_1d,d)
        eigenvals_theory_d=(np.flip(np.sort(eigenvals_theory_d)))

        #linear dims from tensor product of 1d theory
        linear_dims_tensor_1d_theory[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_theory_d,cut_off))

        tensor_prod_eigenvals_theory[(sigma,n_neurons)].append(eigenvals_theory_d)


# uncomment if finding eigenvalues from d dimensional data

# eigenvals_d_median=defaultdict(list)
# eigenvals_d_lower=defaultdict(list)
# eigenvals_d_upper=defaultdict(list)

# comment if not using tensor product of eigenvalues
eigenvals_tensor_d_median=defaultdict(list)

eigenvals_tensor_d_lower=defaultdict(list)

eigenvals_tensor_d_upper=defaultdict(list)

for sigma in sigma_list:
    for n_neurons in total_neurons:
        k = (sigma, n_neurons)
        #uncomment below if using eigenvalues of d dimensional covariance matrix
        
        # eigenvals_d_median[k].append(np.median(eigenvals_d_sim[k][0],axis=0))
        # eigenvals_d_lower[k].append(np.percentile(eigenvals_d_sim[k][0],10,axis=0))
        # eigenvals_d_upper[k].append(np.percentile(eigenvals_d_sim[k][0],90,axis=0))
        
        #comment below if not using median
        eigenvals_tensor_d_median[k].append(np.median(eigenvals_tensor_prod_1d_data[k][0],axis=0))
        eigenvals_tensor_d_lower[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],10,axis=0))
        eigenvals_tensor_d_upper[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],90,axis=0))


#uncomment if using d dimensional simulations

# linear_dims_d_sim_median_by_sigma=defaultdict(list)
# linear_dims_d_sim_lower_by_sigma=defaultdict(list)
# linear_dims_d_sim_upper_by_sigma=defaultdict(list)

# linear_dims_d_sim_median_by_N=defaultdict(list)
# linear_dims_d_sim_lower_by_N=defaultdict(list)
# linear_dims_d_sim_upper_by_N=defaultdict(list)


#comment if not using tensor product of 1d eigenvalues

linear_dims_tensor_cov_1d_median_by_N=defaultdict(list)
linear_dims_tensor_cov_1d_lower_by_N=defaultdict(list)
linear_dims_tensor_cov_1d_upper_by_N=defaultdict(list)

linear_dims_tensor_cov_1d_median_by_sigma=defaultdict(list)
linear_dims_tensor_cov_1d_lower_by_sigma=defaultdict(list)
linear_dims_tensor_cov_1d_upper_by_sigma=defaultdict(list)


linear_dims_tensor_theory_by_N=defaultdict(list)
linear_dims_tensor_theory_by_sigma=defaultdict(list)


for N in total_neurons:
    for sigma in sigma_list:
        k=(sigma,N)
        #uncomment if using d dimensional sims
        
        # linear_dims_d_sim_median_by_sigma[sigma].append(np.median(linear_dims_d_sim[k]))
        # linear_dims_d_sim_lower_by_sigma[sigma].append(np.percentile(linear_dims_d_sim[k],10))
        # linear_dims_d_sim_upper_by_sigma[sigma].append(np.percentile(linear_dims_d_sim[k],90))
        
        # linear_dims_d_sim_median_by_N[N].append(np.median(linear_dims_d_sim[k]))
        # linear_dims_d_sim_lower_by_N[N].append(np.percentile(linear_dims_d_sim[k],10))
        # linear_dims_d_sim_upper_by_N[N].append(np.percentile(linear_dims_d_sim[k],90))
        
        # comment if not using tensor product of 1d eigenvalues
        
        linear_dims_tensor_cov_1d_median_by_sigma[sigma].append(np.median(linear_dims_tensor_cov_1d[k]))
        linear_dims_tensor_cov_1d_lower_by_sigma[sigma].append(np.percentile(linear_dims_tensor_cov_1d[k],10))
        linear_dims_tensor_cov_1d_upper_by_sigma[sigma].append(np.percentile(linear_dims_tensor_cov_1d[k],90))

        linear_dims_tensor_cov_1d_median_by_N[N].append(np.median(linear_dims_tensor_cov_1d[k]))
        linear_dims_tensor_cov_1d_lower_by_N[N].append(np.percentile(linear_dims_tensor_cov_1d[k],10))
        linear_dims_tensor_cov_1d_upper_by_N[N].append(np.percentile(linear_dims_tensor_cov_1d[k],90))

        linear_dims_tensor_theory_by_N[N].append(linear_dims_tensor_1d_theory[k])
        linear_dims_tensor_theory_by_sigma[sigma].append(linear_dims_tensor_1d_theory[k])







#Uncomment if you want to save data

to_save={'d':d,'sigma_list':sigma_list,'total_neurons':total_neurons,
          
          'eigenvals_tensor_d':eigenvals_tensor_prod_1d_data,'eigenvals_tensor_d_median':eigenvals_tensor_d_median,
          'eigenvals_tensor_d_lower':eigenvals_tensor_d_lower,'eigenvals_tensor_d_upper':eigenvals_tensor_d_upper,
          'eigenvals_tensor_theory':tensor_prod_eigenvals_theory,
           'linear_dims_tensor_cov_1d':linear_dims_tensor_cov_1d,
          'linear_dims_tensor_cov_1d_lower_by_N':linear_dims_tensor_cov_1d_lower_by_N,
          'linear_dims_tensor_cov_1d_median_by_N':linear_dims_tensor_cov_1d_median_by_N,
          'linear_dims_tensor_cov_1d_upper_by_N':linear_dims_tensor_cov_1d_upper_by_N,
          'linear_dims_tensor_cov_1d_lower_by_sigma':linear_dims_tensor_cov_1d_lower_by_sigma,
          'linear_dims_tensor_cov_1d_median_by_sigma':linear_dims_tensor_cov_1d_median_by_sigma,
          'linear_dims_tensor_cov_1d_upper_by_sigma':linear_dims_tensor_cov_1d_upper_by_sigma,

          'linear_dims_tensor_1d_theory':linear_dims_tensor_1d_theory,
          'linear_dims_tensor_theory_by_N':linear_dims_tensor_theory_by_N,
          'linear_dims_tensor_theory_by_sigma':linear_dims_tensor_theory_by_sigma}


#Put file path where you want to save here
fw=open('non_circular_multid_data/'+'2023_06_19'+'/linear_dims_data_uniform_centers_d={}'.format(d),'wb')
pickle.dump(to_save,fw)
fw.close()

# 'eigenvals_cov_d':eigenvals_d_sim,'eigenvals_cov_d_median':eigenvals_d_median,
# 'eigenvals_cov_d_lower':eigenvals_d_lower,'eigenvals_cov_d_upper':eigenvals_d_upper,



# plt.figure()
# plt.plot(eigenvals_tensor_d_median[(0.2,50)][0],marker='.',label='non_circular_bc')
# plt.plot(tensor_prod_eigenvals_theory[(0.2,50)][0],marker='.',label='circular bc')
# plt.legend()
# plt.xlabel('PC#')
# plt.ylabel('Var exp')
# plt.title('Var exp vs PC# sigma=0.2')
# plt.savefig('plots/2023_06_20/Var exp vs PC for sigma=02')
# plt.show()
# plt.close()

