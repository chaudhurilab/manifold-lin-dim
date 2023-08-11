# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:03:14 2021

@author: Anandita
"""


''' 1. Generate multi-d data with circular b.c and random tuning curve centers and do pca on it and look at eigenvalues of covariance matrix. 
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
d=2

L=1


#tc widths
#put the list of sigma you want to use here
sigma_list=[0.1,0.2]

#put the list of total no of neurons you want to use here
total_neurons_list=[900]

#no of runs for each parameter combination
no_of_runs=10

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


#For random tuning centers it does not make much sense to take tensor product 
#of linear dimensions 

for sigma in sigma_list:
    #d-dimensional response function with circular b.c
    resp_fn_d = lambda alpha, tuning_curve_center: plf.circular_gaussian_multid(alpha, tuning_curve_center, sigma, L)

    #1-d response function with circular boundary condition
    resp_fn_1d= lambda alpha, tuning_curve_center: plf.circular_gaussian(alpha, tuning_curve_center, sigma, L)
    
    for n_idx,n_neurons in enumerate(total_neurons_list):
        # each row of this matrix is the eigenvalues of d dimensional 
        #covariance matrix from a single run
        eigenvals_d_matrix=np.zeros((no_of_runs,n_neurons))
        
        #each row of this matrix is tensor product of eigenvalues of a
        #1d covariance matrix
        eigenvals_prod_matrix=np.zeros((no_of_runs,n_neurons))
        
        n_neurons_per_dim=int(np.round(n_neurons**(1/d)))
        for run_id in range(no_of_runs):
            start=time.time()

            #generating multid data and performing pca
            #for d>4, this part becomes a lot of data in my laptop
            
            
            if d>1:
                data_d=gen_data_multid(resp_fn_d,n_data_pts,n_neurons,d,L,n_neurons_per_dim,
                            tuning_centers='random')
            elif d==1:
                data_d=gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim,L,tuning_centers='random')
            #normalising the data matrix
            data_d=data_d/np.sqrt(n_data_pts*n_neurons)
            

            #covariance matrix for d dimensional data
            cov_d=np.matmul(np.transpose(data_d),data_d)

            #eigenvalue of covariance matrix
            eigenvals_cov_d,_=np.linalg.eig(cov_d)
            eigenvals_cov_d=np.flip(np.sort(eigenvals_cov_d.real))

            #each row of this matrix contains the eigenvalues for this run
            eigenvals_d_matrix[run_id]=eigenvals_cov_d



            # generating 1d data and taking tensor product of covariance matrix
            #uncomment below if you want to take tensor product of d dimensional 
            #eigenvals
            
            # data_1d=gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim,L,tuning_centers='uniform')
            # data_1d=data_1d/np.sqrt(n_data_pts_1d*n_neurons_per_dim)

            # #calculating the 1d covariance matrix
            # cov_1d=np.matmul(np.transpose(data_1d),data_1d)

            # #calculating eigenvalues of 1d covariance matrix
            # eigenvals_1d_sim,_=np.linalg.eig(cov_1d)

            # #taking tensor product of eigenvalues
            # eigenvals_tensor_d=plf.vector_tens_product(eigenvals_1d_sim.real,d)

            # eigenvals_tensor_d=np.flip(np.sort(eigenvals_tensor_d))
            # eigenvals_prod_matrix[run_id]=eigenvals_tensor_d



            #linear dims from multi d simulation
            linear_dims_d_sim[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_cov_d,cut_off))

            # linear dims from tensor product of 1d covariance data matrices
            #uncomment below if you want to calculate linear dims from tensor product
            #linear_dims_tensor_cov_1d[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_tensor_d,cut_off))

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



eigenvals_d_median=defaultdict(list)

eigenvals_d_lower=defaultdict(list)

eigenvals_d_upper=defaultdict(list)



for sigma in sigma_list:
    for n_neurons in total_neurons_list:
        k = (sigma, n_neurons)
        eigenvals_d_median[k].append(np.median(eigenvals_d_sim[k][0],axis=0))
        eigenvals_d_lower[k].append(np.percentile(eigenvals_d_sim[k][0],10,axis=0))
        eigenvals_d_upper[k].append(np.percentile(eigenvals_d_sim[k][0],90,axis=0))
        


linear_dims_d_sim_median_by_sigma=defaultdict(list)
linear_dims_d_sim_lower_by_sigma=defaultdict(list)
linear_dims_d_sim_upper_by_sigma=defaultdict(list)

linear_dims_d_sim_median_by_N=defaultdict(list)
linear_dims_d_sim_lower_by_N=defaultdict(list)
linear_dims_d_sim_upper_by_N=defaultdict(list)




linear_dims_tensor_theory_by_N=defaultdict(list)
linear_dims_tensor_theory_by_sigma=defaultdict(list)


for N in total_neurons_list:
    for sigma in sigma_list:
        k=(sigma,N)
        linear_dims_d_sim_median_by_sigma[sigma].append(np.median(linear_dims_d_sim[k]))
        linear_dims_d_sim_lower_by_sigma[sigma].append(np.percentile(linear_dims_d_sim[k],10))
        linear_dims_d_sim_upper_by_sigma[sigma].append(np.percentile(linear_dims_d_sim[k],90))
        
        linear_dims_d_sim_median_by_N[N].append(np.median(linear_dims_d_sim[k]))
        linear_dims_d_sim_lower_by_N[N].append(np.percentile(linear_dims_d_sim[k],10))
        linear_dims_d_sim_upper_by_N[N].append(np.percentile(linear_dims_d_sim[k],90))

        

        linear_dims_tensor_theory_by_N[N].append(linear_dims_tensor_1d_theory[k])
        linear_dims_tensor_theory_by_sigma[sigma].append(linear_dims_tensor_1d_theory[k])







#Uncomment if you want to save data

to_save={'d':d,'sigma_list':sigma_list,'total_neurons_list':total_neurons_list,
          'eigenvals_cov_d':eigenvals_d_sim,'eigenvals_cov_d_median':eigenvals_d_median,
          'eigenvals_cov_d_lower':eigenvals_d_lower,'eigenvals_cov_d_upper':eigenvals_d_upper,
          
          'eigenvals_tensor_theory':tensor_prod_eigenvals_theory,
          'linear_dims_d_sim':linear_dims_d_sim,
          'linear_dims_d_sim_median_by_sigma':linear_dims_d_sim_median_by_sigma,
          'linear_dims_d_sim_lower_by_sigma':linear_dims_d_sim_lower_by_sigma,
          'linear_dims_d_sim_upper_by_sigma':linear_dims_d_sim_upper_by_sigma,
            
          'linear_dims_tensor_1d_theory':linear_dims_tensor_1d_theory,
          'linear_dims_tensor_theory_by_N':linear_dims_tensor_theory_by_N,
          'linear_dims_tensor_theory_by_sigma':linear_dims_tensor_theory_by_sigma}


#Put file path where you want to save here
fw=open('circular_multid_data/'+'2023_06_19'+'/linear_dims_data_random_centers_d={}'.format(d),'wb')
pickle.dump(to_save,fw)
fw.close()

