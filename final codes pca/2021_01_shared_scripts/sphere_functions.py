# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:50:02 2021

@author: Anandita
"""

'''This file contains radius and volume functions.'''

import numpy as np
from scipy.special import gamma

def volume_n_sphere(r,d):
    '''Function returns volume of a d dimensional sphere with radius r'''
    V=((np.pi**(d/2))/gamma(0.5*d+1))*(r**d)
    return V


def radius_n_sphere(V,d):
    '''Function returns radius of a d dimensional sphere of V'''
    r=((V*gamma(0.5*d+1))/(np.pi**(d/2)))**(1/d)
    return r


def generate_random_points_in_sphere(r,d,n):
    '''Function to generate n random points in a s d sphere of radius r'''
    x=np.random.normal(0,1,size=(n,d))
    norms=np.reshape(np.linalg.norm(x,axis=1),(len(x),1))
    x=x/norms
    u=np.random.uniform(0,1,size=n)**(1/d)
    u=np.reshape(u,(len(u),1))
    return r*u*x
