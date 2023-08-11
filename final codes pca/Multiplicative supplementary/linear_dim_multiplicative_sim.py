# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:25:26 2023

@author: Anandita
"""

'''Create x's or lambda's from a dirichlet distribution, such that the components
   of x sum to 1.'''

import numpy as np
from scipy.stats import dirichlet 
import sys,os
import matplotlib.pyplot as plt

gen_fn_dir = os.path.abspath('..') + '/2021_01_shared_scripts/'
sys.path.append(gen_fn_dir)

import pca_lvm_fns as plf

alpha=0.8
#choosing the parameter alpha of size 10
alpha_vec=alpha*np.ones(10)

#creating 10 dirichlet random variables with the above parameter alpha
#note that each random variable is a 10-dim vector (this corresponds to the
#eigenvalues of a covariance matrix along a direction). Therefore the following
#will be a 8 \times 10 matrix where each row can be thought of as an eigenvalue
#of a 1d covariance matrix

# X=np.array([dirichlet.rvs(alpha) for alpha in alpha_list])

# X=np.reshape(X,(8,10))

X=dirichlet.rvs(alpha_vec,size=8)

#linear dim calculated from the tensorproduct of eigenvalues
linear_dims_prod=np.zeros(len(X))

#prod of linear dims of eigenvalues along each dimension
prod_linear_dim=np.zeros(len(X))

list_of_tens_prod_vecs=[X[0]]

#cutoff for calculating linear dims
cutoff=0.9

linear_dims_prod[0]=plf.get_cutoff(X[0],cutoff)+1
prod_linear_dim[0]=plf.get_cutoff(X[0],cutoff)+1


for i in range(1,len(X)):
    prod_linear_dim[i]=prod_linear_dim[i-1]*(plf.get_cutoff(X[i],cutoff)+1)
    list_of_tens_prod_vecs.append(np.kron(list_of_tens_prod_vecs[i-1],X[i]))
    linear_dims_prod[i]=plf.get_cutoff(list_of_tens_prod_vecs[i],cutoff)+1


d=np.arange(1,len(X)+1)

#plotting prod of linear dims and linear dims of tensor product of eigenvals
plt.figure()
plt.plot(d,prod_linear_dim,marker='o',label='prod linear dim')
plt.plot(d,linear_dims_prod+1, marker='s',label='linear dim of prod')
plt.yscale('log')
plt.xlabel('d')
plt.ylabel('linear dim')
plt.legend()
plt.title('linear_dim_vs_d,alpha={}'.format(alpha))
plt.savefig('plots/2023_06_15/linear_dim_vs_d_multiplicative_alpha=08')
plt.show()
plt.close()

#plotting the eigenvalue profiles
plt.figure()
for i in range(len(X)):
    plt.plot(np.flip(np.sort(X[i])),marker='.')

plt.title('eigenvalue profiles,alpha={}'.format(alpha))
plt.savefig('plots/2023_06_15/eigenvalue_profiles_along_each_dim_alpha=08')
plt.show()
plt.close()

linear_dim_each_dim=np.zeros(len(X))

for i in range(len(X)):
    linear_dim_each_dim[i]=plf.get_cutoff(X[i],cutoff)+1

plt.figure()
plt.plot(linear_dim_each_dim,'o')
plt.title('Linear dim for each dimension')

plt.savefig('plots/2023_06_15/linear_dim_along_each_dim_alpha=08')
plt.show()
plt.close()

    