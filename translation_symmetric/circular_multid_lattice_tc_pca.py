# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:58:23 2021

@author: Anandita
"""
'''
This file generates data for a fixed d and a range of values for sigma and N
for Gaussian tuning curves with uniformly spaced centers and circular boundary
conditions. In this case it is easier and faster to find d dimensional eigenvalues of
covariance matrix from tensor product of eigenvalues of 1d covariance matrix.
Code has both options, creating d dimensional data matrix from d dimensional
data or taking tensor product of eigenvalues of 1d covariance matrix. Uncommenting 
the last section will save the data in data/Translation symmetric data. The 
load_and_plot codes can then be used to load and plot this data and generate
plots in Figure 3 of the paper.

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

gen_fn_dir = os.path.abspath('..') + '/shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf
import figure_fns as ff

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')


#true dimensions
d=4

#period
L=1


#tc widths
#change according to tuning widths you want to consider
sigma_list=[0.1]

#change according to number of neurons per dimension you want to consider
n_neurons_per_dim_list=np.array([8])


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


for sigma in sigma_list:
    #d-dimensional response function with circular b.c
    resp_fn_d = lambda alpha, tuning_curve_center: plf.circular_gaussian_multid(alpha, tuning_curve_center, sigma, L)

    #1-d response function with circular boundary condition
    resp_fn_1d= lambda alpha, tuning_curve_center: plf.circular_gaussian(alpha, tuning_curve_center, sigma, L)
    
    #total no of neurons
    total_neurons=n_neurons_per_dim_list**d
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
            #for d>4, this part becomes a lot of data in my laptop
            #uncomment below if you want to create d dimensional data
            
            if d>1:
                data_d=plf.gen_data_multid(resp_fn_d,n_data_pts,n_neurons,d,L,n_neurons_per_dim,
                            tuning_centers='uniform')
            elif d==1:
                data_d=plf.gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim,L,tuning_centers='uniform')
            #normalising the data matrix
            data_d=data_d/np.sqrt(n_data_pts*n_neurons)
            
            #uncomment if you want to subtract mean
            #data_d=data_d-np.mean(data_d)

            #covariance matrix for d dimensional data
            cov_d=np.matmul(np.transpose(data_d),data_d)

            #eigenvalue of covariance matrix
            eigenvals_cov_d,_=np.linalg.eig(cov_d)
            eigenvals_cov_d=np.flip(np.sort(eigenvals_cov_d.real))

            #each row of this matrix contains the eigenvalues for this run
            eigenvals_d_matrix[run_id]=eigenvals_cov_d



            # generating 1d data and taking tensor product of covariance matrix
            # comment out below if you only want to run for d dimensional data
            # and do not want tensor product of 1d eigenvalues
            
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
            #uncomment if calculating eigenvalues from d dimensional data
            linear_dims_d_sim[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_cov_d,cut_off))

            # linear dims from tensor product of 1d covariance data matrices
            #comment if not using tensor product of 1d eigenvalues
            #linear_dims_tensor_cov_1d[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_tensor_d,cut_off))

        print('Sigma:',sigma ,'Total neurons:',n_neurons,'d',d,'time:',time.time()-start)
        
        #uncomment if using eigenvals from d dim sims
        eigenvals_d_sim[(sigma,n_neurons)].append(eigenvals_d_matrix)
        
        #comment if not using tensor product of eigenvalues
        eigenvals_tensor_prod_1d_data[(sigma,n_neurons)].append(eigenvals_prod_matrix)

        #calculating the 1d eigenvalues predicted by theory and taking their tensor product
        n1=int(np.ceil(n_neurons_per_dim/2))
        x1=np.arange(-n1,n1)
        eigenvals_theory_1d=np.exp(-((2*np.pi*sigma/L)**2)*x1**2)
        eigenvals_theory_d=plf.vector_tens_product(eigenvals_theory_1d,d)
        eigenvals_theory_d=(np.flip(np.sort(eigenvals_theory_d)))[1:]

        #linear dims from tensor product of 1d theory
        linear_dims_tensor_1d_theory[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_theory_d,cut_off))
        #eigenvalues from theory
        tensor_prod_eigenvals_theory[(sigma,n_neurons)].append(eigenvals_theory_d)


# uncomment if finding eigenvalues from d dimensional data
eigenvals_d_median=defaultdict(list)
eigenvals_d_lower=defaultdict(list)
eigenvals_d_upper=defaultdict(list)

# comment if not using tensor product of eigenvalues
eigenvals_tensor_d_median=defaultdict(list)
eigenvals_tensor_d_lower=defaultdict(list)
eigenvals_tensor_d_upper=defaultdict(list)

for sigma in sigma_list:
    for n_neurons in total_neurons:
        k = (sigma, n_neurons)
        #uncomment below if using eigenvalues of d dimensional covariance matrix
        eigenvals_d_median[k].append(np.median(eigenvals_d_sim[k][0],axis=0))
        eigenvals_d_lower[k].append(np.percentile(eigenvals_d_sim[k][0],10,axis=0))
        eigenvals_d_upper[k].append(np.percentile(eigenvals_d_sim[k][0],90,axis=0))
        
        #comment below if not using median
        eigenvals_tensor_d_median[k].append(np.median(eigenvals_tensor_prod_1d_data[k][0],axis=0))
        eigenvals_tensor_d_lower[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],10,axis=0))
        eigenvals_tensor_d_upper[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],90,axis=0))



#uncomment if using d dimensional simulations
linear_dims_d_sim_median=defaultdict(list)
linear_dims_d_sim_lower=defaultdict(list)
linear_dims_d_sim_upper=defaultdict(list)



#comment if not using tensor product of 1d eigenvalues

# linear_dims_tensor_cov_1d_median=defaultdict(list)
# linear_dims_tensor_cov_1d_lower=defaultdict(list)
# linear_dims_tensor_cov_1d_upper=defaultdict(list)

# linear_dims_tensor_cov_1d_median=defaultdict(list)
# linear_dims_tensor_cov_1d_lower=defaultdict(list)
# linear_dims_tensor_cov_1d_upper=defaultdict(list)


linear_dims_tensor_theory=defaultdict(list)



for N in total_neurons:
    for sigma in sigma_list:
        k=(sigma,N)
        #uncomment if using d dimensional sims
        
        linear_dims_d_sim_median[k].append(np.median(linear_dims_d_sim[k]))
        linear_dims_d_sim_lower[k].append(np.percentile(linear_dims_d_sim[k],10))
        linear_dims_d_sim_upper[k].append(np.percentile(linear_dims_d_sim[k],90))
        
        
        # comment if not using tensor product of 1d eigenvalues
        
        # linear_dims_tensor_cov_1d_median[k].append(np.median(linear_dims_tensor_cov_1d[k]))
        # linear_dims_tensor_cov_1d_lower[k].append(np.percentile(linear_dims_tensor_cov_1d[k],10))
        # linear_dims_tensor_cov_1d_upper[k].append(np.percentile(linear_dims_tensor_cov_1d[k],90))


        linear_dims_tensor_theory[k].append(linear_dims_tensor_1d_theory[k])
        







#Uncomment if you want to save data

# to_save={'d':d,'sigma_list':sigma_list,'total_neurons':total_neurons,
#             'eigenvals_d_sim':eigenvals_d_sim,'eigenvals_d_sim_median':eigenvals_d_median,
#             'eigenvals_d_sim_lower':eigenvals_d_lower,'eigenvals_d_sim_upper':eigenvals_d_upper,
#             'eigenvals_tensor_theory':tensor_prod_eigenvals_theory,
#             'linear_dims_d_sim':linear_dims_d_sim,
#             'linear_dims_d_sim_median':linear_dims_d_sim_median,
#             'linear_dims_d_sim_lower':linear_dims_d_sim_lower,
#             'linear_dims_d_sim_upper':linear_dims_d_sim_upper,
#           'linear_dims_tensor_1d_theory':linear_dims_tensor_1d_theory,
#           'linear_dims_tensor_theory':linear_dims_tensor_theory,
#           'linear_dims_tensor_theory':linear_dims_tensor_theory}


# #Put file path where you want to save here
# f1=os.path.abspath('..')
# fw=open(f1+'data/Translation symmetric tuning/circular_uniform_data_d={}'.format(d),'wb')
# pickle.dump(to_save,fw)
# fw.close()

#include in to_save if you want the results from d dimensional sim
# 'eigenvals_d_sim':eigenvals_d_sim,'eigenvals_d_sim_median':eigenvals_d_median,
#           'eigenvals_d_sim_lower':eigenvals_d_lower,'eigenvals_d_sim_upper':eigenvals_d_upper,
# 'linear_dims_d_sim':linear_dims_d_sim


#include in to_save if you want data from tensor product of 1d data
# 'eigenvals_tensor_d':eigenvals_tensor_prod_1d_data,'eigenvals_tensor_d_median':eigenvals_tensor_d_median,
# 'eigenvals_tensor_d_lower':eigenvals_tensor_d_lower,'eigenvals_tensor_d_upper':eigenvals_tensor_d_upper,
# 'eigenvals_tensor_theory':tensor_prod_eigenvals_theory,
#   'linear_dims_tensor_cov_1d':linear_dims_tensor_cov_1d,
# 'linear_dims_tensor_cov_1d_lower':linear_dims_tensor_cov_1d_lower,
# 'linear_dims_tensor_cov_1d_median':linear_dims_tensor_cov_1d_median,
# 'linear_dims_tensor_cov_1d_upper':linear_dims_tensor_cov_1d_upper,













