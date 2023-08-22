# manifold-lin-dim
Code to calculate the linear dimension for manifolds from De &amp; Chaudhuri (2023).
The directory structure of the code is the following:  //
Code for generating data from translation symmetric tuning curves are in the folder translation_symmetric
circular_1d_uniform_tc_pca creates data from 1d translation symmetric Gaussian tuning curves, performs pca on this data and saves the linear dimension and the singular values of the data in a filepath that the user can update in the code
circular_multid_lattice_tc_pca does the same for data from D dimensional translation symmetric tuning curves.
The load and plot codes in this folder can then be used to load and plot the data to make plots in Figures 2,3 of the paper.
Code for generating data from multiplicative tuning curves are in the folder multiplicative.
The files starting with product_ creates data from product of tuning curves specified in the file, performs pca on this data and saves the linear dimension and the singular values of the data in a file with a filepath the user can provis
The load and plot codes in this folder can then be used to load and plot the data to make plots in Figure 4 of the paper
