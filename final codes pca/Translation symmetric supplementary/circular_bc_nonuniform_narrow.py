# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:57:06 2023

@author: Anandita
"""
import sys
import datetime, sys, os, time
from collections import defaultdict
from scipy.special import gamma
from scipy import linalg as sla
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import numpy as np
from scipy import linalg as sla
#import matplotlib.pyplot as plt
import pickle

gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')

def sampling_tc_around_locations_1d(locations,n_neurons_per_location,radii):
    tc_list=[(np.random.normal(loc=locations[i],scale=radii[i],
                                     size=int(n_neurons_per_location[i]))%1)
                                     for i in range(len(locations))]
    tc_array=np.concatenate(tc_list)
    return tc_array

def gen_data_1d(fn1,fn2, n_data, n_neurons,period,*args,tuning_centers='random'):
    # Get TC centers
    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=n_neurons)
    elif tuning_centers=='uniform':
        mu_list=np.linspace(0,period,n_neurons,endpoint=False)
    elif tuning_centers=='nonuniform':
        mu_list_1=np.linspace(0,period,n_neurons,endpoint=False)
        mu_list_2=sampling_tc_around_locations_1d(*args)
        mu_list=np.concatenate((mu_list_1,mu_list_2))
        
    alpha_vals = np.random.uniform(0,period,size=n_data)
    #alpha_vals=np.linspace(0,period,n_data,endpoint=False)

    data_1 = plf.outer_fn(fn1, alpha_vals, mu_list_1, vectorized='cols')
    data_2=plf.outer_fn(fn2,alpha_vals,mu_list_2,vectorized='cols')
    
    data=np.hstack((data_1,data_2))
    return data



def sampling_tc_around_locations(locations,n_neurons_per_location,radii):
    tc_list=[(multivariate_normal.rvs(mean=locations[i],cov=radii[i]**2,
                                     size=int(n_neurons_per_location[i]))%1)
                                     for i in range(len(locations))]
    tc_array=np.vstack(tc_list)
    return tc_array

def gen_data_multid(fn1,fn2,n_data, n_neurons,n_dims,period,neurons_per_dimension,*args,tuning_centers='random'):

    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=(n_neurons, n_dims))
    elif tuning_centers=='uniform':
        mu_list=plf.create_d_dim_grid(neurons_per_dimension,period, n_dims)
    elif tuning_centers=='nonuniform':
        mu_list_1=plf.create_d_dim_grid(neurons_per_dimension,period, n_dims)
        mu_list_2=sampling_tc_around_locations(*args)
        mu_list=np.vstack((mu_list_1,mu_list_2))
        

    #n_data_per_dimension=np.ceil((n_data)**(1/n_dims))
    alpha_vals = np.random.uniform(0,period,size=(n_data, n_dims))
    #alpha_vals=plf.create_d_dim_grid(n_data_per_dimension,period, n_dims)

    data_1 = plf.outer_fn(fn1, alpha_vals, mu_list_1, vectorized='cols')
    data_2=plf.outer_fn(fn2,alpha_vals,mu_list_2,vectorized='cols')
    data=np.hstack((data_1,data_2))
    return data



#true dimensions
d=1

L=1



#number of neurons per reward location
n_per_rl=5*np.ones(2)

#radius of each reward location
radii=0.05*np.ones(2)

#tc widths
#sigma_list=0.01*np.arange(10,21)

sigma_list=[0.1,0.2]

#sigma

#n_neurons_per_dim_list=np.arange(2,31)
n_neurons_per_dim_list=np.array([50])

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
    #tc_width of added population
    sigma_2=sigma/2
    #d-dimensional response function with circular b.c
    resp_fn_d_1 = lambda alpha, tuning_curve_center: plf.circular_gaussian_multid(alpha, tuning_curve_center, sigma, L)
    
    resp_fn_d_2 = lambda alpha, tuning_curve_center: plf.circular_gaussian_multid(alpha, tuning_curve_center, sigma_2, L)

    #1-d response function with circular boundary condition
    resp_fn_1d_1= lambda alpha, tuning_curve_center: plf.circular_gaussian(alpha, tuning_curve_center, sigma, L)
    
    resp_fn_1d_2= lambda alpha, tuning_curve_center: plf.circular_gaussian(alpha, tuning_curve_center, sigma_2, L)
    
    total_neurons=n_neurons_per_dim_list**d
    
    for n_idx,n_neurons in enumerate(total_neurons):
        n_neurons_per_dim_uniform=n_neurons_per_dim_list[n_idx]
        n_neurons=n_neurons+np.sum(n_per_rl)
        eigenvals_d_matrix=np.zeros((no_of_runs,int(n_neurons)))
        eigenvals_prod_matrix=np.zeros((no_of_runs,int(n_neurons)))
        n_neurons_per_dim=int(n_neurons**(1/d))
        
        for run_id in range(no_of_runs):
            start=time.time()
            #reward locations
            rl=np.random.uniform(0,1,size=2)
            #generating multid data and performing pca
            #for d>4, this part becomes a lot of data in my laptop
            if d>1:
                data_d=gen_data_multid(resp_fn_d_1,resp_fn_d_2,n_data_pts,n_neurons,d,L,n_neurons_per_dim_uniform,
                            rl,n_per_rl,radii,tuning_centers='nonuniform')
            elif d==1:
                data_d=gen_data_1d(resp_fn_1d_1,resp_fn_1d_2,n_data_pts_1d,n_neurons_per_dim_uniform,L,rl,n_per_rl,
                                   radii,tuning_centers='nonuniform')
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
            #linear_dims_tensor_cov_1d[(sigma,n_neurons)].append(plf.get_cutoff(eigenvals_tensor_d,cut_off))

        print('Sigma:',sigma ,'Total neurons:',n_neurons,'d',d,'time:',time.time()-start)

        eigenvals_d_sim[(sigma,n_neurons)].append(eigenvals_d_matrix)
        #eigenvals_tensor_prod_1d_data[(sigma,n_neurons)].append(eigenvals_prod_matrix)

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

# eigenvals_tensor_d_median=defaultdict(list)

# eigenvals_tensor_d_lower=defaultdict(list)

# eigenvals_tensor_d_upper=defaultdict(list)

for sigma in sigma_list:
    for n_neurons in total_neurons+np.sum(n_per_rl):
        
        k = (sigma, n_neurons)
        eigenvals_d_median[k].append(np.median(eigenvals_d_sim[k][0],axis=0))
        eigenvals_d_lower[k].append(np.percentile(eigenvals_d_sim[k][0],10,axis=0))
        eigenvals_d_upper[k].append(np.percentile(eigenvals_d_sim[k][0],90,axis=0))
        # eigenvals_tensor_d_median[k].append(np.median(eigenvals_tensor_prod_1d_data[k][0],axis=0))
        # eigenvals_tensor_d_lower[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],10,axis=0))
        # eigenvals_tensor_d_upper[k].append(np.percentile(eigenvals_tensor_prod_1d_data[k][0],90,axis=0))



linear_dims_d_sim_median=defaultdict(list)
linear_dims_d_sim_lower=defaultdict(list)
linear_dims_d_sim_upper=defaultdict(list)


# linear_dims_tensor_cov_1d_median_by_N=defaultdict(list)
# linear_dims_tensor_cov_1d_lower_by_N=defaultdict(list)
# linear_dims_tensor_cov_1d_upper_by_N=defaultdict(list)

# linear_dims_tensor_cov_1d_median_by_sigma=defaultdict(list)
# linear_dims_tensor_cov_1d_lower_by_sigma=defaultdict(list)
# linear_dims_tensor_cov_1d_upper_by_sigma=defaultdict(list)


linear_dims_tensor_theory_by_N=defaultdict(list)
linear_dims_tensor_theory_by_sigma=defaultdict(list)


for N in total_neurons+np.sum(n_per_rl):
    for sigma in sigma_list:
        
        k=(sigma,N)
        linear_dims_d_sim_median[sigma].append(np.median(linear_dims_d_sim[k]))
        linear_dims_d_sim_lower[sigma].append(np.percentile(linear_dims_d_sim[k],10))
        linear_dims_d_sim_upper[sigma].append(np.percentile(linear_dims_d_sim[k],90))

        # linear_dims_tensor_cov_1d_median_by_sigma[sigma].append(np.median(linear_dims_tensor_cov_1d[k]))
        # linear_dims_tensor_cov_1d_lower_by_sigma[sigma].append(np.percentile(linear_dims_tensor_cov_1d[k],10))
        # linear_dims_tensor_cov_1d_upper_by_sigma[sigma].append(np.percentile(linear_dims_tensor_cov_1d[k],90))

        # linear_dims_tensor_cov_1d_median_by_N[N].append(np.median(linear_dims_tensor_cov_1d[k]))
        # linear_dims_tensor_cov_1d_lower_by_N[N].append(np.percentile(linear_dims_tensor_cov_1d[k],10))
        # linear_dims_tensor_cov_1d_upper_by_N[N].append(np.percentile(linear_dims_tensor_cov_1d[k],90))

        linear_dims_tensor_theory_by_N[N].append(linear_dims_tensor_1d_theory[k])
        linear_dims_tensor_theory_by_sigma[sigma].append(linear_dims_tensor_1d_theory[k])


to_save={'d':d,'reward_locations':rl,'neurons_per_reward_loc':n_per_rl,
          'radii_reward_loc':radii,'sigma_list':sigma_list,
          'eignevals_d_sim':eigenvals_d_sim,
          'eigenvals_d_sim_median':eigenvals_d_median,
          'eigenvals_d_sim_lower':eigenvals_d_lower,
          'eigenvals_d_sim_upper':eigenvals_d_upper,
          'eigenvals_theory':tensor_prod_eigenvals_theory,
          'linear_dims_d_sim_median':linear_dims_d_sim_median,
          'linear_dims_d_sim_lower':linear_dims_d_sim_lower,
          'linear_dims_d_sim_upper':linear_dims_d_sim_upper,
          'linear_dims_tensor_theory_by_sigma':linear_dims_tensor_theory_by_sigma}

fw=open('circular_multid_data/2023_06_19/linear_dim_data_nonuniform_narrow1_tc_d={}'.format(d),'wb')
pickle.dump(to_save,fw)
fw.close()