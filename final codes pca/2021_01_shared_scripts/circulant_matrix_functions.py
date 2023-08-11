# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:33:52 2020

@author: Anandita
"""


import numpy as np
import cmath
import math

def circulant_matrix(c):
    n=len(c)
    C=np.zeros((n,n))
    for i in range(n):
        C[i]=c[np.arange(-i,n-i)]
    return C


def DFT_matrix(n):
    x1=np.arange(n)
    x=np.outer(x1,x1)
    dft_matrix=np.exp((2*np.pi*1j/n)*x)
    return dft_matrix

def wrapped_normal(mu,sigma,no_of_periods,points_per_period):
    #no of periods are even so that there are equal no of periods on either side of 0
    n=int(no_of_periods/2)
    normalisation=(1/((math.sqrt(2*np.pi))*sigma))
    x=np.linspace(-n*2*np.pi,n*2*np.pi,2*n*points_per_period)
    k=np.arange(-n,n+1)
    f_wn=np.zeros(len(x))
    for k_val in k:
        f_wn=f_wn+normalisation*np.exp((-1/(2*sigma**2))*np.square(x-mu+2*np.pi*k_val))
    return x,f_wn

def row_shifted_matrix(c,shift):
    n=len(c)
    r=int(np.ceil(n/shift))
    C=np.zeros((r,n))
    for i in range(r):
        C[i]=c[np.arange(-shift*i,n-shift*i)]
    return C

def symmetric_toeplitz(period, points_per_period,sigma):
    normalisation=1/(np.sqrt(2*np.pi)*sigma)
    x=np.linspace(0,period,points_per_period)
    xx,yy=np.meshgrid(x,x)
    toep_mat=normalisation*np.exp(-(1/(2*sigma**2)*np.square(xx-yy)))
    return(toep_mat)



