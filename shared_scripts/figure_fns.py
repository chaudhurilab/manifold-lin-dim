# October 11th 2017
# Functions to work with figures

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import numpy.linalg as la

def ref_axes(ax,labels=True):
	ax.tick_params(axis='x', which='both', direction='out', length=10, labelsize=20
	, pad=-7)
	ax.xaxis.set_ticks_position('bottom')
	ax.tick_params(axis='y', which='both', direction='out', length=10, labelsize=20
		, pad=-7)
	ax.yaxis.set_ticks_position('left')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if labels==False:
		ax.set_xticklabels([]); ax.set_yticklabels([])

def get_cols_for_list(inp_list, cmap, x0=0.5, x1=1.):
    '''Get a set of colors from cmap, one each for each entry in
    inp_list. Returns a dictionary that pairs each entry with a color.
    x0 and x1 are the limits of where to index into in the colormap (
    which usually goes from 0 to 1).
    Good colormaps are Reds and Blues.'''

    curr_cmap = plt.get_cmap(cmap)
    col_idx = np.linspace(x0, x1, len(inp_list))
    col_dict = {x : curr_cmap(col) for x, col in zip(inp_list, col_idx)}
    return col_dict

# blues = plt.get_cmap("Blues")
# col_idx = np.linspace(0.5, 1, len(N_list))
# col_dict_n = {n : blues(col) for n, col in zip(N_list, col_idx)}

def get_proj(inp_data, inp_view, rescale=False, show_plots=False):
    '''Turn a 3D plot with a particular view into a set of 2D coordinates. The proj3d
    functions seems to rescale the axes, so include an option to approximately
    correct for that.'''
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    ax.scatter(inp_data[:,0], inp_data[:,1], inp_data[:,2])
    ax.view_init(*inp_view)
    out_data = np.zeros((len(inp_data),2))
    out_data[:,0], out_data[:,1], _ = proj3d.proj_transform(inp_data[:,0], inp_data[:,1],
        inp_data[:,2], ax.get_proj())
    if show_plots:
        fig2 = plt.figure(figsize=(6,6))
        ax = fig2.add_subplot(111)
        ax.scatter(out_data[:,0], out_data[:,1])
    else:
        plt.close(fig)
    if rescale:
        first2_norm = np.mean(la.norm(inp_data[:,:2], axis=1))
        curr_norm = np.mean(la.norm(out_data, axis=1))
        out_data = out_data * first2_norm/curr_norm
    return out_data