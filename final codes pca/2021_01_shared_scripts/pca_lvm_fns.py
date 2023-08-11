'''  '''

import numpy as np
from scipy import linalg as sla
from scipy.stats import multivariate_normal


def outer_fn(f, x, y, vectorized='none'):
    '''Generate data from a latent variable model. The (i,j)th element of data 
    is f(x_i, y_j). Setting the convention that the rows (x) correspond to 
    stimulus / latent variable values and the columns (y) correspond to neuron
    parameter (such as the center of the receptive field).

    Basic implementation just loops over all pairs.
    Vectorized == 'rows' assumes that f is vectorized over the second variable and 
    computes an entire row at a time. Vectorized == 'cols' assumes that f is vectorized
    over the first variable and computes a column at a time.
    Note that we typically won't use 'rows' because this corresponds to calculating the
    population response for a single alpha. For the tuning curves we're using the convention
    that we've vectorized over simulus values for a single neuron, which corresponds to 'cols'

    Quickly tested the vectorized setting but should do a bit more. Note that
    a simple way to test whether vectorization is working is to call with and 
    without vectorized for a few configurations and make sure they return the same 
    thing.'''

    if vectorized == 'none':
        return np.array([[f(ix, iy) for iy in y] for ix in x])
    elif vectorized == 'rows':

        return np.array([f(ix, y) for ix in x])
    elif vectorized == 'cols':
        return np.array([f(x, iy) for iy in y]).T
    else:
        print('Unknown setting for vectorized')
        return np.nan



def gen_data_1d(fn, n_data, n_neurons,period,tuning_centers='random'):
    ''' Function to generate 1d data from tuning curve function, tuning parameters,
    no_of_neurons and no_of data pts. Returns a n_neurons \times n_data_pts matrix'''
    # Get TC centers
    #for random tuning centers,tuning centers are sampled from a uniform distribution
    #in the range, for uniform tuning centers, tuning centers are uniformly spaced
    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=n_neurons)
    elif tuning_centers=='uniform':
        mu_list=np.linspace(0,period,n_neurons,endpoint=False)
     
    #data points are sampled from an uniform distribution in [0,period]
    alpha_vals = np.random.uniform(0,period,size=n_data)
    

    data = outer_fn(fn, alpha_vals, mu_list, vectorized='cols')
    return data


def gen_data_multid(fn,n_data, n_neurons,n_dims,period,neurons_per_dimension,tuning_centers='random'):
    ''' Generates data from d-dimensional tuning curves.Returns n_neurons times
    n_data data matrix.
    '''
    if tuning_centers=='random':
        mu_list = np.random.uniform(0,period,size=(n_neurons, n_dims))
    elif tuning_centers=='uniform':
        mu_list=create_d_dim_grid(neurons_per_dimension,period, n_dims)

    alpha_vals = np.random.uniform(0,period,size=(n_data, n_dims))

    data = outer_fn(fn, alpha_vals, mu_list, vectorized='cols')
    return data

def gaussian(x, mu, sigma, norm='max'):
    '''Univariate or multivariate Gaussian response. If norm is 'max'
    then the maximum value is 1. If norm is 'distr', then it's
    normalized as a distribution (i.e., integrated value is 1). 
    Note that for univariate, sigma is the standard deviation. For
    multivariate it's the covariance matrix (so std^2)'''

    # Check if scalar and return scalar Gaussian
    if np.ndim(mu) == 0 or np.ndim(x) == 0:
        delta = (x - mu)**2
        tmp = 2 * sigma**2
        y = np.exp(-delta/tmp)
        if norm == 'distr':
            y = y/(np.sqrt(2*np.pi)*sigma)
    elif np.ndim(mu) > 0 or np.ndim(x) > 0:
        y = multivariate_normal.pdf(x, mean=mu, cov=sigma)
        if norm == 'max':
            # Divide by the max value of the distribution so that it's one
            y = y/multivariate_normal.pdf(mu, mean=mu, cov=sigma)
    return y


def sum_gaussian(x, mu, sigma, period):
    ''' Wrapped normal function is a sum of infinite number of Gaussians.
    We are only interested in the region [0,1] so the sum is taken over 
    -int( 4 \times sigma) to int(4 \times sigma) assuming that only the gaussians 
    in these periods will contribute ot the function in [0,1]'''
    n = int(np.ceil(4*sigma))
    tmp = 2 * sigma**2
    y = np.zeros(len(x))
    for i in range(-n, n+1):
        delta = (x - mu-i*period)**2
        y = y + np.exp(-delta/tmp)
    y = y/(np.sqrt(2*np.pi)*sigma)
    return y


def sum_gaussian_d(x, mu, sigma, period, d):
    '''  x :N_alpha times d matrix
         mu: 1 times d matrix. 
         This function returns a vector of length
        N_alpha where each element of this vector is a multivariate normal v
        with mean mu and period d. '''
    if np.ndim(mu) > 1:
        print('Dimensions of mean is not correct')
        return np.nan
    n = int(np.ceil(4*sigma))
    tmp = 2 * sigma**2
    integer_lattice = lattice_point_grid(n, d)
    if np.ndim(x) == 0:
        y = 0
    else:
        y = np.zeros(len(x))
    # new_centers=mu-period*integer_lattice
    # cov=np.eye(d)*(sigma**2)
    # x1=np.array([multivariate_normal.pdf(x,new_centers_i,cov) for new_centers_i in new_centers])
    # return np.sum(x1,axis=0)
    for L in integer_lattice:
        delta = np.linalg.norm((x - mu-period*L), ord=2, axis=1)
        y = y + np.exp(-delta**2/tmp)
    y = y/((np.sqrt(2*np.pi)*sigma)**d)
    return y


def circular_gaussian(x, mu, sigma, period):
    '''
    Parameters
    ----------
    x : Latent variable. Float or one-d array
    mu : Neuron label for a neuron with a circular gaussian tuning curve. 
         Float or one-d array with all neuron labels.
    sigma :Width of Gaussian tuning curve. Real number
    period : Tuning curve centers and latent variables are assumed to lie within
            [0,period], though the results do not change for translation invariant
            tuning curves as this period can be translated.

    Returns
    -------
    y : Firing rate

    '''
    tmp = 2 * sigma**2
    normalisation = 1/(np.sqrt(2*np.pi)*sigma)
    if type(mu) == float and type(x) == float:
        if x-mu <= period/2:
            y = normalisation*np.exp(-(x-mu)**2/tmp)
        elif x-mu > period/2:
            y = normalisation*np.exp(-(x-mu-period)**2/tmp)
    elif type(mu) == float or type(mu) == np.float64 or type(mu) == int and type(x) == np.ndarray:
        y = np.zeros(len(x))
        if mu <= period/2:
            y[(x-mu) <= period/2] = normalisation * \
                np.exp((-(x[(x-mu) <= period/2]-mu)**2)/tmp)
            y[(x-mu) > period/2] = normalisation * \
                np.exp((-(x[(x-mu) > period/2]-period-mu)**2)/tmp)
        elif mu > period/2:
            y[(mu-x) <= period/2] = normalisation * \
                np.exp((-(x[(mu-x) <= period/2]-mu)**2)/tmp)
            y[(mu-x) > period/2] = normalisation * \
                np.exp((-(x[(mu-x) > period/2]+period-mu)**2)/tmp)

    elif type(mu) == np.ndarray and type(x) == float or type(x) == np.float64:
        y = np.zeros(len(mu))
        if x <= period/2:
            y[(mu-x) <= period/2] = normalisation * \
                np.exp((-(mu[(mu-x) <= period/2]-x)**2)/tmp)
            y[(mu-x) > period/2] = normalisation * \
                np.exp((-(mu[(mu-x) > period/2]-period-x)**2)/tmp)
        elif x > period/2:
            y[(x-mu) <= period/2] = normalisation * \
                np.exp((-(mu[(x-mu) <= period/2]-x)**2)/tmp)
            y[(x-mu) > period/2] = normalisation * \
                np.exp((-(mu[(x-mu) > period/2]+period-x)**2)/tmp)
    return y


def circular_gaussian_multid(x, mu, sigma, period):
    '''d dimensional generalization of the above circular Gaussian tuning curve '''
    d = len(mu)
    data_columns = np.array([circular_gaussian(x[:, i], mu[i], sigma, period) for i in
                            range(d)])
    return(np.prod(data_columns, axis=0))


def tuning_curve_plus_noise(tuning_fn, epsilon, *args):
    '''adds noise to a tuning curve given by tuning_fn(args). Noise is 
    multiplicative. Can take only one dimensional'''
    y = tuning_fn(*args)
    noise = np.random.uniform(1-epsilon, 1+epsilon, size=y.shape)
    return np.multiply(y, noise)


#Function to find the linear dimension given the eigenvalues of the covariance
#matrix and the cutoff fraction
def get_cutoff(x, p, normalize=True):
    '''Return number of entries that need
    to be summed to be greater than p'''

    x_summed = np.cumsum(np.sort(x)[::-1])
    if normalize:
        x_summed = x_summed/x_summed[-1]
    return np.searchsorted(x_summed, p, side='right')




def create_d_dim_grid(n, L, d):
    '''Function to create a d dimensional grid of n^d points in a volume of L^d
    ie have n points between 0 and L in along each dimension'''
    x = np.linspace(0, L, n, endpoint=False)
    grid = np.reshape(np.linspace(0, L, n, endpoint=False), (len(x), 1))
    for i in range(1, d):
        x1 = np.tile(grid, (len(x), 1))
        y1 = np.reshape(np.repeat(x, len(grid)), (len(x1), 1))
        grid = np.column_stack((x1, y1))
    return grid


def create_d_dim_points(x, d):
    '''
    x: One-d array of points
    d: No of dimensions
    
    Returns:
        A x^d times d matrix which contains points of a grid obtained from 
        taking the cartesian product of x d times
    '''
    grid = np.reshape(x, (len(x), 1))
    for i in range(1, d):
        x1 = np.tile(grid, (len(x), 1))
        y1 = np.reshape(np.repeat(x, len(grid)), (len(x1), 1))
        grid = np.column_stack((x1, y1))
    return grid


def lattice_point_grid(n, d):
    '''Function to create a d dimensional grid of n^d  integer lattice points,
        between -n to n'''

    x = np.arange(-n, n+1)
    grid = np.reshape(np.arange(-n, n+1), (len(x), 1))
    for i in range(1, d):
        x1 = np.tile(grid, (len(x), 1))
        y1 = np.reshape(np.repeat(x, len(grid)), (len(x1), 1))
        grid = np.column_stack((x1, y1))
    return grid


def vector_tens_product(v, d):
    ''' Function to which computes kronecker product of vector v, d times'''
    v1 = v
    for i in range(d-1):
        v1 = np.kron(v1, v)
    return v1



    
