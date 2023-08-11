# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:29:32 2021

@author: Anandita
"""


import numpy as np


''' This file contains:
    Different tuning curves for neurons whose response can be described by a 1d '
    latent variable.
    Neural responses which can be described by a product of 1d responses
     '''

def linear_response(x,alpha):
    ''' x is the stimulus value, mu is the tuning curve center for the neuron, theta is the threshold, 
    this function returns a(x-mu) if x-mu >0 else it returns 0,ie the response function is linear with different slopes'''
    mu=alpha[0]
    slope=alpha[1]
    if np.ndim(x)==0 and np.ndim(mu)==0 and np.ndim(slope)==0:
        if x-mu>=0:
            return slope*(x-mu)
        else:
            return 0
    elif np.ndim(x)==1 and np.ndim(mu)==0 and np.ndim(slope)==0:
        return slope*(np.multiply((x>=mu).astype(int)),x-mu)

    elif np.ndim(x)==0 and np.ndim(mu)==1 and np.ndim(slope)==1 and len(mu)==len(slope):
        return np.multiply(slope,(np.multiply((mu>=x).astype(int),mu-x)))


def sigmoidal_response(x,alpha):
    ''' response function 1/1+exp(-slope*(x-mu))'''
    mu=alpha[0]
    slope=alpha[1]
    return 1/(1+np.exp(np.multiply(-slope,x-mu)))


def quadratic_response(x,alpha):
    mu=alpha[0]
    a1=alpha[1]
    a2=alpha[2]
    return a1*(x-mu) +a2*(x-mu)**2

def gaussian_response(x,alpha):
    mu=alpha[0]
    sigma=alpha[1]
    return np.exp(-(x-mu)**2/(2*sigma**2))

def product_tuning_curves(list_of_fns, list_of_alphas_single_neuron, x):
    '''list_of_fns: len d list of function if function along each dimension is different
                    or 1 function if function along each dimension is same
        list_of_alphas_single_neuron: list of parameters for a single neuron, each element is
                                     a length d list of parameters for functions along
                                     the d dimensions,or a list containing parameter list of one
                                     function if the parameters along each dimension are same.Make
                                     sure that each alpha has the no.of.parameters required by the
                                     respective function.
                            x:d dimensional latent variable
        Returns the product of tuning curves along each dimension 
        f_1(x_1,alpha_1)..f_d(x_d,alpha_d)'''
    if np.ndim(x)==0:
        assert(len(list_of_fns)==1)
        assert(len(list_of_alphas_single_neuron)==1)
    elif np.ndim(x)==1:
        assert(len(list_of_fns)==len(x) or len(list_of_fns)==1)
        assert(len(list_of_alphas_single_neuron)==len(x) or len(list_of_alphas_single_neuron)==1)
    
    if np.ndim(x)==0:
        f=list_of_fns[0]
        alpha=list_of_alphas_single_neuron[0]
        fr_single_neuron=f(x,alpha)
    elif np.ndim(x)==1:
        if len(list_of_fns) ==len(list_of_alphas_single_neuron):
            fr_single_neuron =np.prod(np.array([f_i( x_i,alpha) for (f_i, alpha, x_i) in zip(list_of_fns, list_of_alphas_single_neuron, x)]))
                             
        elif len(list_of_fns)==1:
             f=list_of_fns[0]
             fr_single_neuron = np.prod(np.array([f( x_i,alpha) for (alpha, x_i) in zip(list_of_alphas_single_neuron, x)]))
        elif len(list_of_alphas_single_neuron==1) and len(list_of_fns==1):
            f=list_of_fns[0]
            alpha=list_of_alphas_single_neuron[0]
            fr_single_neuron=np.prod(np.array([f(x_i,alpha) for x_i in x]))
    return fr_single_neuron

def product_tuning_curves_n_neurons(list_of_fns, list_of_alphas_by_neurons,x):
    '''list_of_alphas_by_neurons: list of lists of length of no.of.neurons
      For each neuron, either a list of d parameters for each dimension or a single
      parameter if the paramters for functions along each dimension is same.
      Returns an array of firing rates of each neuron given by the product
      of tuning along each dimension'''
    assert(type(list_of_alphas_by_neurons)==list and type(list_of_alphas_by_neurons[0])==list)
    #d>1
    #number of neurons \times d matrix
    if np.ndim(x)==1:
        if len(list_of_fns)>1:
            frs_matrix_by_d=np.array([[f_i(x_i,alpha_i) for (f_i,x_i,alpha_i) in zip(list_of_fns,x,list_of_alphas_by_neurons[n])]
                                  for n in range(len(list_of_alphas_by_neurons))])
        elif len(list_of_fns)==1 and len(list_of_alphas_by_neurons[0])>1:
            f=list_of_fns[0] 
            frs_matrix_by_d=np.array([[f(x_i,alpha_i) for (x_i,alpha_i) in zip(x,list_of_alphas_by_neurons[n])]
                                 for n in range(len(list_of_alphas_by_neurons))])
        elif len(list_of_fns)==1 and len(list_of_alphas_by_neurons[0])==1:
            f=list_of_fns[0] 
            frs_matrix_by_d=np.array([[f(x_i,alpha[0]) for x_i in x] for alpha in list_of_alphas_by_neurons])
    #d=1
    if np.ndim(x)==0:
        f=list_of_fns[0]
        frs_neurons=np.array([f(x,alpha_i[0]) for alpha_i in list_of_alphas_by_neurons])
        return frs_neurons
        
        
    frs_neurons=np.prod(frs_matrix_by_d,axis=1)
    
    return frs_neurons

def product_tuning_curvesn_data_pts(list_of_fns,list_of_alphas_single_neuron,list_x):
    frs_single_neuron=np.array([product_tuning_curves(list_of_fns,list_of_alphas_single_neuron,x_i)
                                for x_i in list_x])
    return frs_single_neuron

def product_tuning_curves_matrix(list_of_fns, list_of_alphas_by_neurons, x_d):
    ''' Returns the data matrix with multiplicative tuning curves'''
    frs_by_neurons_matrix=np.array([product_tuning_curves_n_neurons(list_of_fns, list_of_alphas_by_neurons, x) for x in x_d])
    return frs_by_neurons_matrix







