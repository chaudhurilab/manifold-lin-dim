# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:29:51 2021

@author: Anandita
"""
''' Generate data from Gaussians with varying widths along each dimension. 
    The Gaussians along each dimension has the same set of widths so that the
    data matrix can be expressed as the tensor product of 1d data matrices.
    Uncomment the last part to save the linear dimension and the eigenvalues
    of the covariance matrices calculated from this data in data/Multiplicative
    tuning data. The corresponding load and plot can be used to make the plots
    corresponding to Gaussian tuning curves with varying widths in Figure 4 of the
    paper.'''

import datetime, sys, os, time
from collections import defaultdict

import numpy as np
from scipy import linalg as sla
import matplotlib.pyplot as plt
import pickle
import itertools


gen_fn_dir = os.path.abspath('..') + '/shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf
import figure_fns as ff
import one_d_tuning_curves as tc
#from general_file_fns import load_pickle_file, save_pickle_file


curr_date=datetime.datetime.now().strftime('%Y_%m_%d')

# list of intrinsic dimensions
#d_list=[2,3,4,5,6,7,8]
d_list=[2]

#no of neurons per dim
n_neurons_per_dim=8

#n data per dim
n_data_per_dim=1000

#no_of_data_points
n_data=10000

# we do multiple runs for each parameter combination because the latent 
#variables x are sampled randomly
no_of_runs=10

#the random variables x lie in a range[0,period]
period=1

#cutoff to calculate the linear dimension
cut_off=0.9

#dictionary to store linear dimensions from d dimensional simulations
linear_dims_d = defaultdict(list)

#dictionary to store linear dimensions from the tensor product of 1d eigenvalues
linear_dims_tensor_prod=defaultdict(list)


#list to store eigenvalues of covariance matrix from d dimensional data
eigenvals_d_list=defaultdict(list)

#list to store eigenvalues from covariance matrix from tensor prod
#of 1d data
eigenvals_tensor_d=defaultdict(list)

eigenvals_1d_list=defaultdict(list)


sigma_list=np.abs(np.linspace(-0.2,0.2,n_neurons_per_dim))

for d in d_list:
   
    
    #1d centers for sigmoids
    mu_list_1d=np.linspace(0,period,n_neurons_per_dim)
    
    #d_dim mu_list_for
    mu_list_d=plf.create_d_dim_points(mu_list_1d,d)
    
    #params list for 1d data from sigmoid tuning curves
    params_list_1d=[[params] for params in zip(mu_list_1d,sigma_list)]
    
    #params list for d dimensional data from sigmoid tuning curves
    prod_params=list(itertools.product(params_list_1d,repeat=d))
    params_list_d=[sum(prod_params[i],[]) for i in range(len(prod_params))]

    #fn_list has one element because same function along each dimension
    fn_list=[tc.gaussian_response]

    #eigenvalue matrix where each row is the eigenvalues from each run 
    #from d dimensional sim
    eigenvalues_matrix_d=np.zeros((no_of_runs,n_neurons_per_dim**d))
    
    #eigenvalue matrix where each row is the eigenvalues from each run 
    #from tensor prod of 1d sim
    eigenvalues_matrix_tens_prod=np.zeros((no_of_runs,n_neurons_per_dim**d))
    
    #eigenvalues of 1d covariance matrix
    eigenvalues_matrix_1d=np.zeros((no_of_runs,n_neurons_per_dim))

    for run_idx in range(no_of_runs):
        start=time.time()
        
        #1d latent variables
        #x_1d=np.random.uniform(0,period,n_data_per_dim)
        x_1d=np.linspace(0,period,n_data_per_dim)
        
        #d dim latent variables
        x_d=np.random.uniform(0,period,size=(n_data,d))
        #generating d dimensional data and finding the eigenvalues of the covariance matrix
        if d<=4:
             #creating the d dimensional data matrix with sigmoidal response
             data_d=tc.product_tuning_curves_matrix(fn_list, params_list_d, x_d)
             #calculating the covariance matrix
             cov_d= (1/n_data)*np.matmul(np.transpose(data_d),data_d)

             #eigenvalues of covariance matrix
             eigenvals_d,_=np.linalg.eig(cov_d)

             eigenvalues_matrix_d[run_idx]=eigenvals_d.real

             linear_dims_d[d].append(plf.get_cutoff(eigenvals_d.real,cut_off))
        
        #creating 1d data and finding tensor product
        data_1d=tc.product_tuning_curves_matrix(fn_list,params_list_1d,x_1d)
        
        #calculating the covariance matrix
        cov_1d= (1/n_data_per_dim)*np.matmul(np.transpose(data_1d),data_1d)
        #eigenvalues of covariance matrix
        eigenvals_1d,_=np.linalg.eig(cov_1d)
        eigenvals_tensor_prod_1=np.flip(np.sort(eigenvals_1d.real))

        eigenvalues_matrix_1d[run_idx]=eigenvals_tensor_prod_1

        #assuming that the slopes along each dimension is the same, covariance of the d dimensional
        eigenvals_tensor_prod=np.flip(np.sort(plf.vector_tens_product(eigenvals_1d.real,d)))
                                     
        eigenvalues_matrix_tens_prod[run_idx]=eigenvals_tensor_prod

        linear_dims_tensor_prod[d].append(plf.get_cutoff(eigenvals_tensor_prod,cut_off))
        print('Run:',run_idx,'d:',d,'Time:',time.time()-start)
    #eigenvals_1d_list[(n_neurons_per_dim)].append(eigenvalues_matrix_1d)
    if d<=4:
        eigenvals_d_list[d]=eigenvalues_matrix_d

    eigenvals_tensor_d[d]=eigenvalues_matrix_tens_prod
    
    

#1d eigenvalues
eigenvals_1d_median=np.median(eigenvalues_matrix_1d,axis=0)
eigenvals_1d_lower=np.percentile(eigenvalues_matrix_1d,10,axis=0)
eigenvales_1d_upper=np.percentile(eigenvalues_matrix_1d,90,axis=0)


#d dimensional eigenvalues
eigenvals_d_median=defaultdict(list)
eigenvals_d_lower=defaultdict(list)
eigenvals_d_upper=defaultdict(list)

eigenvals_d_median_frac=defaultdict(list)
eigenvals_d_lower_frac=defaultdict(list)
eigenvals_d_upper_frac=defaultdict(list)

eigenvals_tensor_d_median=defaultdict(list)
eigenvals_tensor_d_lower=defaultdict(list)
eigenvals_tensor_d_upper=defaultdict(list)

eigenvals_tensor_d_median_frac=defaultdict(list)
eigenvals_tensor_d_lower_frac=defaultdict(list)
eigenvals_tensor_d_upper_frac=defaultdict(list)

#linear dimension
linear_dims_d_median=defaultdict(list)
linear_dims_d_lower=defaultdict(list)
linear_dims_d_upper=defaultdict(list)

linear_dims_tensor_prod_median=defaultdict(list)
linear_dims_tensor_prod_lower=defaultdict(list)
linear_dims_tensor_prod_upper=defaultdict(list)



for d in d_list:
    if d<=4:
        linear_dims_d_median[d]=np.median(linear_dims_d[d])
        linear_dims_d_lower[d]=np.percentile(linear_dims_d[d],10)
        linear_dims_d_upper[d]=np.percentile(linear_dims_d[d],90)

        eigenvals_d_median[d]=np.median(eigenvals_d_list[d],axis=0)
        eigenvals_d_lower[d]=np.percentile(eigenvals_d_list[d],10,axis=0)
        eigenvals_d_upper[d]=np.percentile(eigenvals_d_list[d],90,axis=0)
        
        #normalising the eigenvalues matrix
        sum_eigenvalues=np.sum(eigenvals_d_list[d],axis=1)
        
        sum_eigenvalues=np.reshape(sum_eigenvalues,(len(sum_eigenvalues),1))

        eigenvals_normalised_matrix_d=np.divide(eigenvals_d_list[d],sum_eigenvalues)

        eigenvals_d_median_frac[d]=np.median(eigenvals_normalised_matrix_d,axis=0)
        eigenvals_d_lower_frac[d]=np.percentile(eigenvals_normalised_matrix_d,10,axis=0)
        eigenvals_d_upper_frac[d]=np.percentile(eigenvals_normalised_matrix_d,90,axis=0)
        
    #tensor product of1d eigenvalues
    eigenvals_tensor_d_median[d]=np.median(eigenvals_tensor_d[d],axis=0)
    eigenvals_tensor_d_lower[d]=np.percentile(eigenvals_tensor_d[d],10,axis=0)
    eigenvals_tensor_d_upper[d]=np.percentile(eigenvals_tensor_d[d],90,axis=0)
        
    #normalizing eigenvalues by the total sum
    sum_eigenvalues_tensor=np.sum(eigenvals_tensor_d[d],axis=1)
    sum_eigenvalues_tensor=np.reshape(sum_eigenvalues,(len(sum_eigenvalues_tensor),1))
    eigenvals_normalised_matrix_tensor=np.divide(eigenvals_tensor_d[d],sum_eigenvalues_tensor)

    eigenvals_tensor_d_median_frac[d]=np.median(eigenvals_normalised_matrix_tensor,axis=0)
    eigenvals_tensor_d_lower_frac[d]=np.percentile(eigenvals_normalised_matrix_tensor,10,axis=0)
    eigenvals_tensor_d_upper_frac[d]=np.percentile(eigenvals_normalised_matrix_tensor,90,axis=0)

    #tensor product of linear dimension
    linear_dims_tensor_prod_median[d]=np.median(linear_dims_tensor_prod[d])
    linear_dims_tensor_prod_lower[d]=np.percentile(linear_dims_tensor_prod[d],10)
    linear_dims_tensor_prod_upper[d]=np.percentile(linear_dims_tensor_prod[d],90)


#saving the data
#uncomment if you want to sace data
# to_save={'linear_dims_d':linear_dims_d,'linear_dims_tensor_prod':linear_dims_tensor_prod,
#           'eigenvals_d_median':eigenvals_d_median,'eigenvals_d_lower':eigenvals_d_lower,
#           'eigenvals_d_upper':eigenvals_d_upper,
#           'eigenvals_d_median_frac':eigenvals_d_median_frac,
#           'eigenvals_d_upper_frac':eigenvals_d_upper_frac,
#           'eigenvals_d_lower_frac':eigenvals_d_lower_frac,
#           'eigenvals_tensor_d_median':eigenvals_tensor_d_median,
#             'eigenvals_tensor_d_lower':eigenvals_tensor_d_lower, 
#            'eigenvals_tensor_d_upper':eigenvals_tensor_d_upper,
#             'eigenvals_tensor_d_median_frac':eigenvals_tensor_d_median_frac,
#            'eigenvals_tensor_d_lower_frac':eigenvals_tensor_d_lower_frac,
#             'eigenvals_tensor_d_upper_frac':eigenvals_tensor_d_upper_frac,
#           'linear_dims_d_lower':linear_dims_d_lower,'linear_dims_d_median':linear_dims_d_median,
#         'linear_dims_d_upper':linear_dims_d_upper,'linear_dims_tensor_prod_median':linear_dims_tensor_prod_median,
#         'linear_dims_tensor_prod_lower':linear_dims_tensor_prod_lower,
#           'linear_dims_tensor_prod_upper':linear_dims_tensor_prod_upper,
#           'eigenvals_1d':eigenvals_1d_median}

##put filepath where you want to save here or create a folder with the current date
##as its name in the folder data/Multiplicative tuning
# f1=os.path.abspath('..')
# a=open(f1+'/data/Multiplicative tuning/'+curr_date+'/product_gaussian_varying_width_response','wb')
# pickle.dump(to_save,a)
# a.close()


