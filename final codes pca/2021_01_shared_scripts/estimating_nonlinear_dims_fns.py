# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:33:34 2021

@author: Anandita
"""


import numpy as np
from sklearn import neighbors

def get_nbrs(count_matrix, rad_start, rad_end, step):
    neigh = neighbors.NearestNeighbors()
    neigh.fit(count_matrix)
    rad_list = np.arange(rad_start, rad_end, step)
    all_nbrs = []
    print('running RS analysis. ')
    for point in count_matrix:
        # print (str(rad)),
        nbrs_all_radii = [neigh.radius_neighbors(X=[point], radius=rad, return_distance=False) for
            rad in rad_list]
        all_nbrs.append([(len(nbrs_at_radii[0])-1) for nbrs_at_radii in nbrs_all_radii])
    return np.asarray(rad_list), np.asarray(all_nbrs)