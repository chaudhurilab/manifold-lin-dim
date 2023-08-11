# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:41:26 2023

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

curr_date=datetime.datetime.now().strftime('%Y_%m_%d')

#true dimensions
d_list=[1]

L=1


#tc widths
#sigma_list=0.01*np.arange(10,21)

sigma_list=np.array([0.1])

n_neurons_per_dim=np.array([50])


#no of runs for each parameter combination
no_of_runs=1



def gen_data_1d(fn, n_data, n_neurons,period,tuning_centers='random'):
    # Get TC centers
    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=n_neurons)
    elif tuning_centers=='uniform':
        mu_list=np.linspace(0,period,n_neurons,endpoint=False)

    #alpha_vals = np.random.uniform(0,period,size=n_data)
    alpha_vals=np.linspace(0,period,n_data,endpoint=False)

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

#no of data points for d dimensional simulation
n_data_pts=50000

#no of data points for one dimensional simulation
n_data_pts_1d=100

#set cut-off for finding linear dimensions here
cut_off=0.9

noise_parameter=0.2

#eigenvalues of covariance matrix form d dimensional simulation
eigenvals_d_sim=defaultdict(list)

eigenvals_d_sim_plus_noise=defaultdict(list)

#eigenvalues from taking tensor product of eigenvalues from 1d theory
tensor_prod_eigenvals_theory=defaultdict(list)

#eigenvalues from taking tensor product of 1d simulations
eigenvals_tensor_prod_1d_data=defaultdict(list)

#linear dimensions from d dimensional simulation
linear_dims_d_sim=defaultdict(list)

linear_dims_d_sim_plus_noise=defaultdict(list)


#linear dimension from tensor product of 1d eigenvalues
linear_dims_tensor_cov_1d=defaultdict(list)

#linear dimensions from tensor product of 1d eigenvalues predicted from theory
linear_dims_tensor_1d_theory=defaultdict(list)


for d in d_list:
    for sigma in sigma_list:
        #d-dimensional response function with circular b.c
        resp_fn_d = lambda alpha, tuning_curve_center: plf.circular_gaussian_multid(alpha, tuning_curve_center, sigma, L)
        
        resp_fn_d_noise=lambda alpha, tuning_curve_center: plf.tuning_curve_plus_noise(plf.circular_gaussian_multid, noise_parameter, alpha, tuning_curve_center, sigma, L)

        #1-d response function with circular boundary condition
        resp_fn_1d= lambda alpha, tuning_curve_center: plf.circular_gaussian(alpha, tuning_curve_center, sigma, L)
        
        resp_fn_1d_noise= lambda alpha, tuning_curve_center:plf.tuning_curve_plus_noise(plf.circular_gaussian,noise_parameter,alpha, tuning_curve_center, sigma, L)
        
        total_neurons=n_neurons_per_dim**d
        for n_idx,n_neurons in enumerate(total_neurons):
            eigenvals_d_matrix=np.zeros((no_of_runs,n_neurons))
            noise_eigenvals_d_matrix=np.zeros((no_of_runs,n_neurons))
            
            for run_id in range(no_of_runs):
                start=time.time()

                #generating multid data and performing pca
                # for d>4, this part becomes a lot of data in my laptop
                if d>1:
                    data_d=gen_data_multid(resp_fn_d,n_data_pts,n_neurons,d,L,n_neurons_per_dim[n_idx],
                                tuning_centers='uniform')
                    data_d_noise=gen_data_multid(resp_fn_d_noise,n_data_pts,n_neurons,d,L,n_neurons_per_dim[n_idx],
                                tuning_centers='uniform')
                elif d==1:
                    data_d=gen_data_1d(resp_fn_1d,n_data_pts_1d,n_neurons_per_dim[n_idx],L,tuning_centers='uniform')
                    
                    data_d_noise=gen_data_1d(resp_fn_1d_noise,n_data_pts_1d,n_neurons_per_dim[n_idx],L,tuning_centers='uniform')
                #normalising the data matrix
                if d==1:
                      data_d=data_d/np.sqrt(n_data_pts_1d*n_neurons)
                
                      data_d_noise=data_d_noise/np.sqrt(n_data_pts_1d*n_neurons)
                elif d>1:
                    data_d=data_d/np.sqrt(n_data_pts*n_neurons)
              
                    data_d_noise=data_d_noise/np.sqrt(n_data_pts*n_neurons)
                
                #covariance matrix for d dimensional data
                cov_d=np.matmul(np.transpose(data_d),data_d)
                
                cov_d_noise=np.matmul(np.transpose(data_d_noise),data_d_noise)
                #eigenvalue of covariance matrix
                eigenvals_cov_d,_=np.linalg.eig(cov_d)
                eigenvals_cov_d=np.flip(np.sort(eigenvals_cov_d.real))
                eigenvals_d_matrix[run_id]=eigenvals_cov_d
                
                eigenvals_cov_d_noise,_=np.linalg.eig(cov_d_noise)
                eigenvals_cov_d_noise=np.flip(np.sort(eigenvals_cov_d_noise.real))
                noise_eigenvals_d_matrix[run_id]=eigenvals_cov_d_noise

                

                #linear dims from multi d simulation
                linear_dims_d_sim[(d,sigma)].append(plf.get_cutoff(eigenvals_cov_d,cut_off))

                linear_dims_d_sim_plus_noise[(d,sigma)].append(plf.get_cutoff(eigenvals_cov_d_noise,cut_off))

        print('Sigma:',sigma ,'Total neurons:',n_neurons,'d',d,'time:',time.time()-start)

        eigenvals_d_sim[(d,sigma)]=eigenvals_d_matrix
        eigenvals_d_sim_plus_noise[(d,sigma)]=noise_eigenvals_d_matrix

        #calculating the 1d eigenvalues predicted by theory and taking their tensor product
        n1=int(np.ceil(n_neurons_per_dim[n_idx]/2))
        x1=np.arange(-n1,n1)
        eigenvals_theory_1d=np.exp(-((2*np.pi*sigma/L)**2)*x1**2)
        eigenvals_theory_d=plf.vector_tens_product(eigenvals_theory_1d,d)
        eigenvals_theory_d=(np.flip(np.sort(eigenvals_theory_d)))

        #linear dims from tensor product of 1d theory
        linear_dims_tensor_1d_theory[(d,sigma)].append(plf.get_cutoff(eigenvals_theory_d,cut_off))

        tensor_prod_eigenvals_theory[(d,sigma)].append(eigenvals_theory_d)


linear_dims_d_sim_median=defaultdict(list)
linear_dims_d_sim_lower=defaultdict(list)
linear_dims_d_sim_upper=defaultdict(list)

linear_dims_noise_median=defaultdict(list)
linear_dims_noise_upper=defaultdict(list)
linear_dims_noise_lower=defaultdict(list)



for sigma in sigma_list:
    k=(d_list[0],sigma)
    linear_dims_d_sim_median[sigma].append(np.median(linear_dims_d_sim[k]))
    linear_dims_d_sim_lower[sigma].append(np.percentile(linear_dims_d_sim[k],10))
    linear_dims_d_sim_upper[sigma].append(np.percentile(linear_dims_d_sim[k],90))
    
    linear_dims_noise_median[sigma].append(np.median(linear_dims_d_sim_plus_noise[k]))
    linear_dims_noise_lower[sigma].append(np.percentile(linear_dims_d_sim_plus_noise[k],10))
    linear_dims_noise_upper[sigma].append(np.percentile(linear_dims_d_sim_plus_noise[k],90))
    


eigenvals_d_sim_median=defaultdict(list)
eigenvals_d_sim_upper=defaultdict(list)
eigenvals_d_sim_lower=defaultdict(list)

eigenvals_d_sim_plus_noise_median=defaultdict(list)
eigenvals_d_sim_plus_noise_upper=defaultdict(list)
eigenvals_d_sim_plus_noise_lower=defaultdict(list)


for sigma in sigma_list:
    k=(d_list[0],sigma)
    eigenvals_d_sim_median[sigma]=np.median(eigenvals_d_sim[k],axis=0)
    eigenvals_d_sim_lower[sigma]=np.percentile(eigenvals_d_sim[k],10,axis=0)
    eigenvals_d_sim_upper[sigma]=np.percentile(eigenvals_d_sim[k],90,axis=0)
    
    eigenvals_d_sim_plus_noise_median[sigma]=np.median(eigenvals_d_sim_plus_noise[k],axis=0)
    eigenvals_d_sim_plus_noise_lower[sigma]=np.percentile(eigenvals_d_sim_plus_noise[k],10,axis=0)
    eigenvals_d_sim_plus_noise_upper[sigma]=np.percentile(eigenvals_d_sim_plus_noise[k],90,axis=0)
    

#saving data
to_save={'d_list':d_list,'sigma_list':sigma_list,'n_neurons_per_dim':n_neurons_per_dim,
         'noise_parameter':noise_parameter,
         'linear_dims_d_sim':linear_dims_d_sim,
         'linear_dims_d_sim_median':linear_dims_d_sim_median,
         'linear_dims_d_sim_lower':linear_dims_d_sim_lower,
         'linear_dims_d_sim_upper':linear_dims_d_sim_upper,
         'linear_dims_d_sim_plus_noise':linear_dims_d_sim_plus_noise,
         'linear_dims_d_sim_plus_noise_median':linear_dims_noise_median,
         'linear_dims_d_sim_plus_noise_lower':linear_dims_noise_lower,
         'linear_dims_d_sim_plus_noise_upper':linear_dims_noise_upper,
         'eigenvals_d_sim':eigenvals_d_sim,
         'eigenvals_d_sim_median':eigenvals_d_sim_median,
         'eigenvals_d_sim_lower':eigenvals_d_sim_lower,
         'eigenvals_d_sim_upper':eigenvals_d_sim_upper,
         'eigenvals_d_sim_plus_noise':eigenvals_d_sim_plus_noise,
         'eigenvals_d_sim_plus_noise_median':eigenvals_d_sim_plus_noise_median,
         'eigenvals_d_sim_plus_noise_lower':eigenvals_d_sim_plus_noise_lower,
         'eigenvals_d_sim_plus_noise_upper':eigenvals_d_sim_plus_noise_upper}

fw=open('circular_multid_data/2023_06_19/pca_cirular_bc_plus_noise_d={}'.format(d_list[0]),'wb')
pickle.dump(to_save,fw)
fw.close()


# plt.rcParams.update({'font.size':14})    
# plt.figure(figsize=(8,6))
# plt.plot(sigma_list, linear_dims_d_sim_median[d_list[0]],marker='o',label='no noise')
# plt.plot(sigma_list, linear_dims_noise_median[d_list[0]],marker='o',label='with noise')
# plt.legend()
# plt.xlabel('sigma')
# plt.ylabel('linear dim')
# plt.title('Linear dim vs sigma for d=4 ,epsilon_noise=0.2')
# #plt.savefig('plots/2023_06_19/linear_dim_vs_sigma_d=4,noise=02')
# plt.show()
# plt.close()

# plt.figure(figsize=(8,6))
# plt.plot(cov_d[15]+0.0001)
# plt.plot(cov_d_noise[15],label='cov_plus_noise')
# plt.legend()
# plt.title("covariance matrixrow with and without noise,d=4,noise=0.2")
# #plt.savefig('plots/2023_06_19/covariance_matrix_rows_d=4,noise=02')
# plt.show()
# plt.close()

# sigma=sigma_list[0]
# plt.figure(figsize=(8,6))
# plt.plot(eigenvals_d_sim_median[sigma][0:50],'o',label='eigenvals_without_noise')
# plt.plot(eigenvals_d_sim_plus_noise_median[sigma][0:50],'^',label='eigenvals_with_noise')
# plt.legend()
# plt.xlabel('PC#')
# plt.ylabel('exp variance')
# plt.title('Var exp. vs PC#, sigma={},d={},epsilon_noise=0.2'.format(sigma,d_list[0]))
# #plt.savefig('plots/2023_06_19/covariance_matrix_rows_d=4,noise=02,sigma=01')
# plt.show()
# plt.close()

x=np.linspace(0,1,n_data_pts_1d,endpoint=False)

import matplotlib as mpl  
mpl.rc('font',family='Arial')

plt.rcParams.update({'font.size':16})
#csfont = {'fontname':'Arial'}
hfont = {'fontname':'Arial'}

indices=np.array([0,9,19,29,39,49])
plt.figure()
for i in indices:
    plt.plot(x, data_d_noise[:,i]*(np.sqrt(n_data_pts*n_neurons)))
plt.title('Tuning curves of different neurons, sigma=0.2, epsilon_noise=0.1',**hfont)
plt.savefig('plots/2023_06_20/Approx_translation_inv_diff_neurons_sigma=01')
plt.show()
plt.close()

