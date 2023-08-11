# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:09:32 2023

@author: Anandita
"""
'''In this file we consider product of two different tuning curves, sigmoidal and
 Gaussian. We consider an even no. of dimensions and assume tuning along half of
 the dimensions are sigmoidal and along half of the dimensions are Gaussian. 
 Uncomment the last part if you want to save the linear dimension and the 
 eigenvalues of the covariance matrix calculated from this data. The load and plot
 codes can then be used to make the plots in the fourth row of Figure 4 in the paper.'''

import datetime, sys, os, time
from collections import defaultdict

import numpy as np
from scipy import linalg as sla
import matplotlib.pyplot as plt
import pickle
import itertools


gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf
import figure_fns as ff
import one_d_tuning_curves as tc

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')
# of neurons per dimension
n_neurons_per_dim=8

#total no. of dimensions (should be even)
d=8

#half of the dimensions have sigmoidal tuning and the other half has Gaussian tuning
half_d=int(d/2)

no_of_runs=10
#total neurons
#n_neurons=n_neurons_per_dim_list**d

# of data points
n_data_per_dim=1000
n_data=10**4

period=2

# the stimulus values are picked randomly so we run each simulation multiple times
no_of_runs=10

#cutoff to calculate linear dimensions
cut_off=0.95

#list to store linear dimensions for linear response function
linear_dims_d = defaultdict(list)
linear_dims_tensor_prod=defaultdict(list)
linear_dims_1d_gaussian=defaultdict(list)
linear_dims_1d_sigmoid=defaultdict(list)

#list to store eigenvalues of covariance matrix for linear response function
eigenvals_tensor_d=defaultdict(list)
eigenvals_d_list=defaultdict(list)

#parameters for sigmoidal tuning curves
sigmoid_centers=np.linspace(0,2,n_neurons_per_dim,endpoint=False)
#slopes of the sigmoids are uniformly distributed between -5 and 5
slopes=np.linspace(-5,5,n_neurons_per_dim)
#creating a list of parameters for the sigmoid tuning curve
sigmoid_params_list=[[params] for params in zip(sigmoid_centers,slopes)]

#parameters for gaussian response
#one can also choose to have different number of neurons along this dimension
tuning_centers=np.linspace(0,2,n_neurons_per_dim,endpoint=False)
sigma_list=np.linspace(0.1,0.5,n_neurons_per_dim,endpoint=False)
#creating a list of parameters for the gaussian tuning curves
gaussian_params_list=[[params] for params in zip(tuning_centers,sigma_list)]

#putting the parameters for these two functions together
params_list=[[params1,params2] for params1 in zip(sigmoid_centers,slopes) 
             for params2 in zip(tuning_centers,sigma_list)]

#for d=4,6 etc,the parameters are all possible combinations or cartesian
#products of the above parameters, and then massaging them to have the 
#right form. For example, for d=4, the following will be a list n_neurons
# elements, where each element contains a list of d parameter lists or tuples
if d>2:
    prod_params=list(itertools.product(params_list,repeat=half_d))
    params_list=[sum(prod_params[i],[]) for i in range(len(prod_params))]
    

#functions along each dimension
fns_list=[tc.sigmoidal_response,tc.gaussian_response]
fns_list=fns_list*(half_d)

#fns_list for 1d data
fn_list_1=[tc.sigmoidal_response]
fn_list_2=[tc.gaussian_response]


#eigenvalue matrix where each row is the eigenvalues from each run from d 
#d dimensional sim
eigenvals_matrix_d=np.zeros((no_of_runs,n_neurons_per_dim**d))
#eigenvalue matrix where each row is tensor product of eigenvalues
# from a single run
eigenvals_matrix_tensor_prod=np.zeros((no_of_runs,n_neurons_per_dim**d))

#eigenvalue matrix where each row is 1d eigenvalues from respective tuning curve
eigenvals_matrix_1d_gaussian=np.zeros((no_of_runs,n_neurons_per_dim))
eigenvals_matrix_1d_sigmoid=np.zeros((no_of_runs,n_neurons_per_dim))

for run_id in range(no_of_runs):
    start=time.time()
    #1d latent variables
    x_1d=np.random.uniform(0,period,size=n_data_per_dim)
    #creating d dimensional data if d<=4
    if d<=4:
        #d dimensional latent variables
        x_d=np.random.uniform(0,period,size=(n_data,d))
    
        #2d multiplicative tuning curve with sigmoidal tuning along 1d and 
        #gaussian tuning along another dimension(look at fns_list above)
        data_d=tc.product_tuning_curves_matrix(fns_list, params_list, x_d)
        
        #calculating the covariance matrix
        cov_d= (1/n_data)*np.matmul(np.transpose(data_d),data_d)
        
        #eigenvalues of covariance matrix
        eigenvals_d,_=np.linalg.eig(cov_d)
         
        eigenvals_matrix_d[run_id]=eigenvals_d.real
    
        #finding the linear dimension from the d dimensional data
        linear_dims_d[d].append(plf.get_cutoff(eigenvals_d.real,0.95))
    
    #creating 1d data from sigmoidal tuning curves
    data_1d_sigmoid=tc.product_tuning_curves_matrix(fn_list_1,sigmoid_params_list,x_1d)
    
    cov_1d_sigmoid=(1/n_data_per_dim)*np.matmul(np.transpose(data_1d_sigmoid),data_1d_sigmoid)
    
    eigenvals_1d_sigmoid,_=np.linalg.eig(cov_1d_sigmoid)
    
    eigenvals_matrix_1d_sigmoid[run_id]=eigenvals_1d_sigmoid
    
    linear_dims_1d_sigmoid[d].append(plf.get_cutoff(eigenvals_1d_sigmoid,cut_off))
    
    
    #creating 1d data from gaussian tuning curves
    data_1d_gaussian=tc.product_tuning_curves_matrix(fn_list_2,gaussian_params_list,x_1d)
    
    cov_1d_gaussian=(1/n_data_per_dim)*np.matmul(np.transpose(data_1d_gaussian),data_1d_gaussian)
    
    eigenvals_1d_gaussian,_=np.linalg.eig(cov_1d_gaussian)
    
    eigenvals_matrix_1d_gaussian[run_id]=eigenvals_1d_gaussian
    
    linear_dims_1d_gaussian[d].append(plf.get_cutoff(eigenvals_1d_gaussian,cut_off))
    
    #taking the tensor product of the eigenvalues of the two covariances above
    eigenvals_tensor_prod=np.kron(eigenvals_1d_sigmoid,eigenvals_1d_gaussian)
    eigenvals_tensor_prod=np.flip(np.sort(plf.vector_tens_product(eigenvals_tensor_prod,half_d)))
    
    eigenvals_matrix_tensor_prod[run_id]=eigenvals_tensor_prod
    
    #finding the linear dimension from the tensor product
    linear_dims_tensor_prod[d].append(plf.get_cutoff(eigenvals_matrix_tensor_prod[run_id],cut_off))
    
    print('Run:',run_id,'time taken',time.time()-start)




#eigenvalues from d dimensional data
if d<=4:
    eigenvals_d_median=np.median(eigenvals_matrix_d,axis=0)
    eigenvals_d_lower=np.percentile(eigenvals_matrix_d,10,axis=0)
    eigenvals_d_upper=np.percentile(eigenvals_matrix_d,90,axis=0)
    
    # normalising eigenvalues by the total sum
    sum_eigenvalues=np.sum(eigenvals_matrix_d,axis=1)

    sum_eigenvalues=np.reshape(sum_eigenvalues,(len(sum_eigenvalues),1))

    eigenvals_normalised_matrix_d=np.divide(eigenvals_matrix_d,sum_eigenvalues)

    eigenvals_d_median_frac=np.median(eigenvals_normalised_matrix_d,axis=0)
    eigenvals_d_lower_frac=np.percentile(eigenvals_normalised_matrix_d,10,axis=0)
    eigenvals_d_upper_frac=np.percentile(eigenvals_normalised_matrix_d,90,axis=0)

#eigenvalues 1d
eigenvals_1d_gaussian_median=np.median(eigenvals_matrix_1d_gaussian,axis=0)
eigenvals_1d_sigmoid_median=np.median(eigenvals_matrix_1d_sigmoid,axis=0)


#tensor product of 1d eigenvalues
eigenvals_tensor_d_median=np.median(eigenvals_matrix_tensor_prod,axis=0)
eigenvals_tensor_d_lower=np.percentile(eigenvals_matrix_tensor_prod,10,axis=0)
eigenvals_tensor_d_upper=np.percentile(eigenvals_matrix_tensor_prod,90,axis=0)

#normalizing eigenvalues by the total sum
sum_eigenvalues_tensor=np.sum(eigenvals_matrix_tensor_prod,axis=1)

sum_eigenvalues_tensor=np.reshape(sum_eigenvalues_tensor,(len(sum_eigenvalues_tensor),1))

eigenvals_normalised_matrix_tensor=np.divide(eigenvals_matrix_tensor_prod,sum_eigenvalues_tensor)

eigenvals_tensor_d_median_frac=np.median(eigenvals_normalised_matrix_tensor,axis=0)
eigenvals_tensor_d_lower_frac=np.percentile(eigenvals_normalised_matrix_tensor,10,axis=0)
eigenvals_tensor_d_upper_frac=np.percentile(eigenvals_normalised_matrix_tensor,90,axis=0)


#linear dimension calculated from d dimensional data
if d<=4:
    linear_dims_d_median=defaultdict(list)
    linear_dims_d_lower=defaultdict(list)
    linear_dims_d_upper=defaultdict(list)
    
    linear_dims_d_median[d]=np.median(linear_dims_d[d])
    linear_dims_d_lower[d]=np.percentile(linear_dims_d[d],10)
    linear_dims_d_upper[d]=np.percentile(linear_dims_d[d],90)


#linear dimension from tensor product of 1d data
linear_dims_tensor_prod_median=defaultdict(list)
linear_dims_tensor_prod_lower=defaultdict(list)
linear_dims_tensor_prod_upper=defaultdict(list)


linear_dims_tensor_prod_median[d]=np.median(linear_dims_tensor_prod[d])
linear_dims_tensor_prod_lower[d]=np.percentile(linear_dims_tensor_prod[d],10)
linear_dims_tensor_prod_upper[d]=np.percentile(linear_dims_tensor_prod[d],90)






#uncomment if you want to save data
# if d<=4:
#     to_save={'linear_dims_d':linear_dims_d,'linear_dims_tensor_prod':linear_dims_tensor_prod,
#           'eigenvals_d_median':eigenvals_d_median,'eigenvals_d_lower':eigenvals_d_lower,
#           'eigenvals_d_upper':eigenvals_d_upper,
#           'eigenvals_d_median_frac':eigenvals_d_median_frac,
#           'eigenvals_d_upper_frac':eigenvals_d_upper_frac,
#           'eigenvals_d_lower_frac':eigenvals_d_lower_frac,
#           'eigenvals_tensor_d_median':eigenvals_tensor_d_median,
#             'eigenvals_tensor_d_lower':eigenvals_tensor_d_lower, 'eigenvals_tensor_d_upper':eigenvals_tensor_d_upper,
#           'linear_dims_d_lower':linear_dims_d_lower,
#           'eigenvals_tensor_d_median_frac':eigenvals_tensor_d_median_frac,
#           'eigenvals_tensor_d_lower_frac':eigenvals_tensor_d_lower_frac,
#           'eigenvals_tensor_d_upper_frac':eigenvals_tensor_d_upper_frac,
#           'linear_dims_d_median':linear_dims_d_median,
#           'linear_dims_d_lower':linear_dims_d_lower,
#           'linear_dims_d_upper':linear_dims_d_upper,
#           'linear_dims_tensor_prod_median':linear_dims_tensor_prod_median,
#           'linear_dims_tensor_prod_lower':linear_dims_tensor_prod_lower,
#           'linear_dims_tensor_prod_upper':linear_dims_tensor_prod_upper,
#           'eigenvals_1d_gaussian_median':eigenvals_1d_gaussian_median,
#           'eigenvals_1d_sigmoid_median':eigenvals_1d_sigmoid_median,
#           'linear_dims_1d_gaussian':np.median(linear_dims_1d_gaussian[d]),
#           'linear_dims_1d_sigmoid':np.median(linear_dims_1d_sigmoid[d])}

#     f1=os.path.abspath('..')
#     a=open(f1+'data/Multiplicative tuning/'+curr_date+'/product_sigmoid_gaussian_data_d={}'.format(d),'wb')
#     pickle.dump(to_save,a)
#     a.close()
# else:
#     to_save={
#           'eigenvals_tensor_d_median':eigenvals_tensor_d_median,
#             'eigenvals_tensor_d_lower':eigenvals_tensor_d_lower, 'eigenvals_tensor_d_upper':eigenvals_tensor_d_upper,

#           'eigenvals_tensor_d_median_frac':eigenvals_tensor_d_median_frac,
#           'eigenvals_tensor_d_lower_frac':eigenvals_tensor_d_lower_frac,
#           'eigenvals_tensor_d_upper_frac':eigenvals_tensor_d_upper_frac,
#           'linear_dims_tensor_prod_median':linear_dims_tensor_prod_median,
#           'linear_dims_tensor_prod_lower':linear_dims_tensor_prod_lower,
#           'linear_dims_tensor_prod_upper':linear_dims_tensor_prod_upper,
#           'eigenvals_1d_gaussian_median':eigenvals_1d_gaussian_median,
#           'eigenvals_1d_sigmoid_median':eigenvals_1d_sigmoid_median,
#           'linear_dims_1d_gaussian':np.median(linear_dims_1d_gaussian[d]),
#           'linear_dims_1d_sigmoid':np.median(linear_dims_1d_sigmoid[d])}

##put filepath where you want to save here or create a folder with the current date
##as its name in the folder data/Multiplicative tuning
#     f1=os.path.abspath('..')
#     a=open(f1+'/data/Multiplicative tuning/'+curr_date+'/product_sigmoid_gaussian_data_d={}'.format(d),'wb')
#     pickle.dump(to_save,a)
#     a.close()
    


