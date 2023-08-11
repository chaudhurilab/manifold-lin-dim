# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:00:24 2023

@author: Anandita
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

d_list=np.array([1,2,3,4])

sigma_list=[0.1,0.2]

circular_multid_data_with_noise=defaultdict(list)

for d in d_list:
    fw=open('circular_multid_data/2023_06_19/pca_cirular_bc_plus_noise_d={}'.format(d),'rb')
    circular_multid_data_with_noise[d]=pickle.load(fw)
    fw.close()


linear_dim_median_by_sigma=defaultdict(list)
linear_dim_lower_by_sigma=defaultdict(list)
linear_dim_upper_by_sigma=defaultdict(list)

linear_dim_median_with_noise_by_sigma=defaultdict(list)
linear_dim_lower_with_noise_by_sigma=defaultdict(list)
linear_dim_upper_with_noise_by_sigma=defaultdict(list)

for sigma in sigma_list:
    for d in d_list:
        linear_dim_median_by_sigma[sigma].append(circular_multid_data_with_noise[d]['linear_dims_d_sim_median'][sigma][0])
        linear_dim_lower_by_sigma[sigma].append(circular_multid_data_with_noise[d]['linear_dims_d_sim_lower'][sigma][0])
        linear_dim_upper_by_sigma[sigma].append(circular_multid_data_with_noise[d]['linear_dims_d_sim_upper'][sigma][0])
        
        linear_dim_median_with_noise_by_sigma[sigma].append(circular_multid_data_with_noise[d]['linear_dims_d_sim_plus_noise_median'][sigma][0])
        linear_dim_lower_with_noise_by_sigma[sigma].append(circular_multid_data_with_noise[d]['linear_dims_d_sim_plus_noise_lower'][sigma][0])
        linear_dim_upper_with_noise_by_sigma[sigma].append(np.percentile(circular_multid_data_with_noise[d]['linear_dims_d_sim_plus_noise'][(d,sigma)],90))
        

        
sigma=sigma_list[0]

plt.figure()
plt.plot(d_list,linear_dim_median_by_sigma[sigma],color='blue',label='without noise')
plt.fill_between(d_list,linear_dim_lower_by_sigma[sigma],linear_dim_upper_by_sigma[sigma],color='blue',alpha=0.4)

plt.plot(d_list,linear_dim_median_with_noise_by_sigma[sigma],'o',color='magenta',label='with noise')
plt.fill_between(d_list,linear_dim_lower_with_noise_by_sigma[sigma],
                 linear_dim_upper_with_noise_by_sigma[sigma],color='magenta',alpha=0.4)
plt.legend()
plt.yscale('log')
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title("Linear dim vs d, sigma={}".format(sigma))
plt.savefig('plots/2023_06_20/Linear_dim_vs_d_with_and_without_noise,sigma=01')
plt.show()
plt.close()

plt.figure()
plt.bar(d_list-0.2,linear_dim_median_by_sigma[sigma],width=0.2,color='blue',label='without noise')


plt.bar(d_list,linear_dim_median_with_noise_by_sigma[sigma],width=0.2,color='magenta',label='with noise')
plt.legend()
plt.yscale('log')
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title("Linear dim vs d, sigma={}".format(sigma))
plt.savefig('plots/2023_06_20/Linear_dim_vs_d_with_and_without_noise_bar,sigma=01')
plt.show()
plt.close()

sigma=sigma_list[1]

plt.figure()
plt.plot(d_list,linear_dim_median_by_sigma[sigma],color='blue',label='without noise')
plt.fill_between(d_list,linear_dim_lower_by_sigma[sigma],linear_dim_upper_by_sigma[sigma],color='blue',alpha=0.4)

plt.plot(d_list,linear_dim_median_with_noise_by_sigma[sigma],'o',color='magenta',label='with noise')
plt.fill_between(d_list,linear_dim_lower_with_noise_by_sigma[sigma],
                 linear_dim_upper_with_noise_by_sigma[sigma],color='magenta',alpha=0.4)
plt.legend()
plt.yscale('log')
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title("Linear dim vs d, sigma={}".format(sigma))
plt.savefig('plots/2023_06_20/Linear_dim_vs_d_with_and_without_noise,sigma=02')
plt.show()
plt.close()

plt.figure()
plt.bar(d_list-0.2,linear_dim_median_by_sigma[sigma],width=0.2,color='blue',label='without noise')


plt.bar(d_list,linear_dim_median_with_noise_by_sigma[sigma],width=0.2,color='magenta',label='with noise')
plt.legend()
plt.yscale('log')
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title("Linear dim vs d, sigma={}".format(sigma))
plt.savefig('plots/2023_06_20/Linear_dim_vs_d_with_and_without_noise_bar,sigma=02')
plt.show()
plt.close()





        
        