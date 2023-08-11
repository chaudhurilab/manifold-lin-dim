# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:31:09 2023

@author: Anandita
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

d_list=np.array([1,2,3])

sigma_list=[0.1,0.2]

circular_multid_data_nonunform_tc=defaultdict(list)

for d in d_list:
    fw=open('circular_multid_data/2023_06_19/linear_dim_data_nonuniform_narrow1_tc_d={}'.format(d),'rb')
    circular_multid_data_nonunform_tc[d]=pickle.load(fw)
    fw.close()


linear_dim_median_by_sigma=defaultdict(list)
linear_dim_lower_by_sigma=defaultdict(list)
linear_dim_upper_by_sigma=defaultdict(list)

linear_dim_theory_by_sigma=defaultdict(list)

for sigma in sigma_list:
    for d in d_list:
        linear_dim_median_by_sigma[sigma].append(circular_multid_data_nonunform_tc[d]['linear_dims_d_sim_median'][sigma][0])
        linear_dim_lower_by_sigma[sigma].append(circular_multid_data_nonunform_tc[d]['linear_dims_d_sim_lower'][sigma][0])
        linear_dim_upper_by_sigma[sigma].append(circular_multid_data_nonunform_tc[d]['linear_dims_d_sim_upper'][sigma][0])
        
        linear_dim_theory_by_sigma[sigma].append(circular_multid_data_nonunform_tc[d]['linear_dims_tensor_theory_by_sigma'][sigma][0][0])


sigma=sigma_list[0]

plt.figure()
plt.plot(d_list,linear_dim_median_by_sigma[sigma],'o',color='blue',label='nonuniform tc')
plt.fill_between(d_list,linear_dim_lower_by_sigma[sigma],linear_dim_upper_by_sigma[sigma],color='blue',alpha=0.4)

plt.plot(d_list,linear_dim_theory_by_sigma[sigma],color='magenta',label='uniform tc')
plt.legend()
plt.xticks([1,2,3])
plt.yscale('log')
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title('Nonuniform narrow tc, background sigma={}'.format(sigma))
plt.savefig('plots/2023_06_20/linear_dim_vs_d_nonuniform_narrow1_tc_background_sigma=01')
plt.show()
plt.close()

# plt.figure()
# plt.bar(d_list-0.2,linear_dim_median_by_sigma[sigma],width=0.2,color='blue',label='nonuniform tc')

# plt.bar(d_list,linear_dim_theory_by_sigma[sigma],width=0.2,color='magenta',label='uniform tc')
# plt.legend()
# plt.yscale('log')
# plt.ylabel("Linear dim")
# plt.title('Linear dim vs d, nonuniform tc, background sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/linear_dim_vs_d_nonuniform_tc_background_sigma=01_bar')
# plt.show()
# plt.show()
# plt.close()

sigma=sigma_list[1]

plt.figure()
plt.plot(d_list,linear_dim_median_by_sigma[sigma],'o',color='blue',label='nonuniform tc')
plt.fill_between(d_list,linear_dim_lower_by_sigma[sigma],linear_dim_upper_by_sigma[sigma],color='blue',alpha=0.4)

plt.plot(d_list,linear_dim_theory_by_sigma[sigma],color='magenta',label='uniform tc')
plt.legend()
plt.yscale('log')
plt.xticks([1,2,3])
plt.xlabel('d')
plt.ylabel("Linear dim")
plt.title('Nonuniform narrow tc, background sigma={}'.format(sigma))
plt.savefig('plots/2023_06_20/linear_dim_vs_d_nonuniform_narrow1_tc_backgorund_sigma=02')
plt.show()
plt.close()

# plt.figure()
# plt.bar(d_list-0.2,linear_dim_median_by_sigma[sigma],width=0.2,color='blue',label='nonuniform tc')

# plt.bar(d_list,linear_dim_theory_by_sigma[sigma],width=0.2,color='magenta',label='uniform tc')
# plt.legend()
# plt.yscale('log')
# plt.ylabel("Linear dim")
# plt.title('Linear dim vs d, nonuniform tc, background sigma={}'.format(sigma))
# plt.savefig('plots/2023_06_20/linear_dim_vs_d_nonuniform_tc_backgorund_sigma=02_bar')
# plt.show()
# plt.show()
# plt.close()