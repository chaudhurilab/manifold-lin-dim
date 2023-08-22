# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:00:46 2020

@author: Anandita
"""
''' This file generates  data with uniform tuning curve centers and random stimulus values in a given period 
    in one dimension with periodic boundary conditions and then performs pca on this data. 
    The linear dimension of the data and the eigenvalues of the covariance matrix
    are found both analytically and from simulations. If you want to save the data
    uncomment the last part. This will save the data in data/Translation symmetric data.
    The load and plot codes can then be used to load and plot this data which is
    shown in Figure 2 of the paper.
'''

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

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')

# Simulation parameters
#List of tuning curve widths
sigma_list=np.array([0.05,0.1,0.15,0.2])

#List of number of neurons, change to see how linear dimension changes with
#no of neurons
N_list = [100]


# Multiple tests b/c slight variability from random latent variable (data pt) draws
n_tests_per_param_combo = 10

#No of latent variables, change to see how linear dimension depends on no of data
#points
n_data_pts = 50000

# Tuning curve centers and latent variables lie in range [0,period]
period=1


# dictionaries to store linear dimension and eigenvalues of covariance matrix from
#simulations for different parameter values
linear_dim_sims = defaultdict(list)
cov_eigenvalues_sims=defaultdict(list)

# dictionaries to store linear dimension and eigenvalues of covariance matrix 
#calculated analytically
linear_dims_theory=defaultdict(list)

eigenvals_theory=defaultdict(list)

#fraction of variance  explained by linear dim no of components
cutoff=0.95


for sigma in sigma_list:
    t0 = time.time()
    # Make the data gen function specific to this sigma
    resp_fn = lambda data_point, tuning_curve_center: plf.circular_gaussian(
        data_point, tuning_curve_center, sigma=sigma, period=period)
    for n_neurons in N_list:
        #Each row of this matrix stores the singular values for the given parameter combination
        #ie the (sigma,#of neurons)
        eigen_val_matrix=np.zeros((n_tests_per_param_combo,min(n_neurons,n_data_pts)))
        for curr_sample in range(n_tests_per_param_combo):
            #data is the neural data matrix with the parameter combination.
            data = plf.gen_data_1d(resp_fn, n_data_pts, n_neurons,period,tc_centers='uniform')
            
            #subtracting the mean from each column
            #data=data-np.mean(data)
            
            #normalizing the data matrix by diving by no of neurons
            #and no of neurons
            data=1/(np.sqrt(n_neurons*n_data_pts))*data
            
            #Finding the covariance matrix
            cov_1d=np.matmul(np.transpose(data),data)
            
            #Finding the eigenvalues of the covariance matrix
            eigen_val_1d,_=np.linalg.eig(cov_1d)
            
            #Making sure that the eigenvalues are in decreasing order of
            #magnitude
            eigen_val_1d=np.flip(np.sort(eigen_val_1d.real))
           
            #n_runs \times n_eigenvals matrix which stores the eigenvalues
            #for the different runs of a given parameter combination
            eigen_val_matrix[curr_sample]=eigen_val_1d
            
            #Finding the linear dimension as the number of eigenvalues
            #that contain a large fraction of the total mass of eigenvalues
            linear_dim_sims[(sigma, n_neurons)].append(plf.get_cutoff(
                eigen_val_1d, cutoff))
        
        #Storing the eigenvalues for each parameter combinations as n_run 
        #\times n_neurons matrix
        cov_eigenvalues_sims[(sigma,n_neurons)].append(eigen_val_matrix)
        
        #Calculating the theoretical eigenvalues
        n=int(n_neurons/2)
        y=np.arange(-n,n)
        y1=np.exp((-(2*np.pi*sigma/period)**2)*y**2)
        y1=np.flip(np.sort(y1))
        eigenvals_theory[(sigma,n_neurons)]=y1
        
    #linear dimensions from semi-analytical calculation
    linear_dims_theory[sigma]=(2*(1.96*period/(2*np.pi*np.sqrt(2)*sigma))-1)
    #linear_dims_theory[(sigma,n_neurons)]=plf.get_cutoff(y1,cutoff)
    print('Sigma {}, time {} '.format(sigma, time.time()-t0))



# Grouping eigenvalues and linear dimension by sigma and N
# and finding the median, 10 percentile and 90 percentile for
# both linear dimension and eigenvalues from simulations

linear_dim_median_by_sigma = defaultdict(list)
linear_dim_median_by_N = defaultdict(list)
linear_dim_upper_by_sigma = defaultdict(list)
linear_dim_upper_by_N = defaultdict(list)
linear_dim_lower_by_sigma = defaultdict(list)
linear_dim_lower_by_N = defaultdict(list)




eigenvals_median=defaultdict(list)

eigenvals_lower=defaultdict(list)

eigenvals_upper=defaultdict(list)


for sigma in sigma_list:
    for n_neurons in N_list:
        k = (sigma, n_neurons)
        linear_dim_median_by_sigma[sigma].append(np.median(linear_dim_sims[k]))
        linear_dim_median_by_N[n_neurons].append(np.median(linear_dim_sims[k]))
        linear_dim_lower_by_sigma[sigma].append(np.percentile(linear_dim_sims[k],10))
        linear_dim_lower_by_N[n_neurons].append(np.percentile(linear_dim_sims[k],10))
        linear_dim_upper_by_sigma[sigma].append(np.percentile(linear_dim_sims[k],90))
        linear_dim_upper_by_N[n_neurons].append(np.percentile(linear_dim_sims[k],90))




for sigma in sigma_list:
    for n_neurons in N_list:
        k = (sigma, n_neurons)
        eigenvals_median[k]=np.median(cov_eigenvalues_sims[k][0],axis=0)
        eigenvals_lower[k]=(np.percentile(cov_eigenvalues_sims[k][0],10,axis=0))
        eigenvals_upper[k]=(np.percentile(cov_eigenvalues_sims[k][0],90,axis=0))




#uncomment for saving the data

# to_save={'period':period,'sigma_list':sigma_list,'n_neurons_list':N_list,
#           'linear_dim_lower_by_N':linear_dim_lower_by_N,
#           'linear_dim_median_by_N':linear_dim_median_by_N,
#           'linear_dim_upper_by_N':linear_dim_upper_by_N,
#           'linear_dim_lower_by_sigma':linear_dim_lower_by_sigma,
#           'linear_dim_median_by_sigma':linear_dim_median_by_sigma,
#           'linear_dim_upper_by_sigma':linear_dim_upper_by_sigma,
#           'eigenvals_lower':eigenvals_lower,
#           'eigenvals_median':eigenvals_median,
#           'eigenvals_upper':eigenvals_upper,
#           'linear_dims_theory':linear_dims_theory,
#           'eigenvals_theory':eigenvals_theory}

# # put in filepath name for where you want to save data, I have put in filepath name 
# # in the way I have setup my directory system. However you want to change this in
# # any way convenient

# f1=os.path.abspath('..') 
# fw=open(f1+'/data/Translation symmetric tuning/1d_data_uniform_circular_bc','wb')
# pickle.dump(to_save,fw)
# fw.close()

