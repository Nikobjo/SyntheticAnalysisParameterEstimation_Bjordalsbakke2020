"""
Module with Monte Carlo methods for uncertainty and  sensitivity analysis using the chaospy package for sampling
"""
import numpy as np

# sample matrices
def generate_sample_matrices(Ns, jpdf, sample_method='R'):
    """
    Generate the sample matrices A, B and C_i for Saltelli's algorithm
    Inputs:
    Ns (int): Number of independent samples for each matrices A and B
    jpdf (dist): distribution object with a sample method
    sample_method (str): specify sampling method to use
    Returns A, B, C
    A, B (arrays): samples are rows and parameters are columns
    C (arrays): samples are first dimension, second index indicates which parameter is fixed,
    and parameters values third are indexed by third dimension
    """
    number_of_parameters = len(jpdf)
    Xtot = jpdf.sample(2*Ns, sample_method).transpose()
    np.random.seed(0)
    np.random.shuffle(Xtot)  # TODO: why is this necessary with Sobol?
    A = Xtot[:Ns]
    B = Xtot[Ns:]
    C = np.empty((number_of_parameters, Ns, number_of_parameters))
    for i in range(number_of_parameters):
        C[i, :, :] = B.copy()
        C[i, :, i] = A[:, i].copy()
    return A, B, C


def calculate_sensitivity_indices(y_a, y_b, y_c):
    """
    Saltelli's algorithm for estimating Si and 
    Sobol 2007 algorithm for S_t using Monte Carlo integration
    
    Inputs:
    y_a, y_b (array): first index corresponds to sample second to variables of interest
    y_c (array): first index corresponds to sample second to conditional index and 
        following dimensions to variables of interest
        
    Returns: s, st
        s (array): first order sensitivities first index corrresponds to input second 
            to variable of interest
        st (array): total sensitivities first index corrresponds to input second 
            to variable of interest
    """
    s_shape = y_c.shape[0:1] + y_c.shape[2:]
    s = np.zeros(s_shape)
    st = np.zeros(s_shape)

    mean = 0.5*(np.mean(y_a,axis=0) + np.mean(y_b,axis=0))
    y_a_center = y_a - mean
    y_b_center = y_b - mean
    f0sq = np.mean(y_a_center,axis=0) * np.mean(y_b_center,axis=0) # 0 when data is centered
    var_est = np.var(y_b, axis=0)
    for i, y_c_i in enumerate(y_c):
        y_c_i_center = y_c_i - mean
        #
        #s[i] = (np.mean(y_a_center*y_c_i_center, axis=0)-f0sq)/var_est #Sobol 1993 
        s[i] = np.mean(y_a_center*(y_c_i_center - y_b_center), axis=0)/var_est #Saltelli 2010
        #st[i] = 1 - (np.mean(y_c_i_center*y_b_center, axis=0) - f0sq)/var_est  #Homma  1996
        st[i] = np.mean(y_b_center*(y_b_center-y_c_i_center), axis=0)/var_est #Sobol 2007
    return s, st


def evaluate_samples_args(func, samplesA, samplesB, args=(), eval_mode=None):
    numberOfSamples, dim = samplesA.shape[0:2]
    if eval_mode == "parallel":
        import multiprocessing
        dataA = func((samplesA, *args))
        dataB = func((samplesB, *args))
        dataC = np.empty((dim,numberOfSamples))
        samplesC = [None,]*dim
        for i in range(dim):
            samplesCi = samplesB.copy()
            samplesCi[:,i] = samplesA[:,i].copy()
            samplesC[i] = (samplesCi, *args)
        #samplesC = np.reshape(samplesC
        pool = multiprocessing.Pool()
        dataC = np.array(pool.map(func, samplesC))
        pool.close()
        pool.join()
    elif eval_mode == "vectorized":
        raise NotImplementedError()
        dataA = func(samplesA)
        dataB = func(samplesB)
        dataC = np.empty((dim, *dataA.shape))
        for i in range(dim):
            samplesCi = samplesB.copy()
            samplesCi[:,i] = samplesA[:,i].copy()
            dataC[i] = func(samplesCi)
    else:
        raise NotImplementedError()
        dataA = np.array([func(z) for z in samplesA]) # TODO Always at least 2 dimensional
        dataB = np.array([func(z) for z in samplesB])
        dataC = np.empty((dim, *dataA.shape))
        for i in range(dim):
            samplesCi = samplesB.copy()
            samplesCi[:,i] = samplesA[:,i].copy()
            dataC[i] = np.array([func(z) for z in samplesCi])
    return dataA, dataB, dataC

def evaluate_samples(func, samplesA, samplesB, eval_mode=None, args=None):
    numberOfSamples, dim = samplesA.shape[0:2]
    if eval_mode == "parallel":
        import multiprocessing
        dataA = func(samplesA)
        dataB = func(samplesB)
        dataC = np.empty((dim,numberOfSamples))
        samplesC = np.empty((dim,numberOfSamples,dim))
        for i in range(dim):
            samplesC[i] = samplesB.copy()
            samplesC[i,:,i] = samplesA[:,i].copy()
        #samplesC = np.reshape(samplesC
        pool = multiprocessing.Pool()
        dataC = np.array(pool.map(func, samplesC))
        pool.close()
        pool.join()
    elif eval_mode == "vectorized":
        dataA = func(samplesA)
        dataB = func(samplesB)
        dataC = np.empty((dim, *dataA.shape))
        for i in range(dim):
            samplesCi = samplesB.copy()
            samplesCi[:,i] = samplesA[:,i].copy()
            dataC[i] = func(samplesCi)
    else:
        dataA = np.array([func(z) for z in samplesA]) # TODO Always at least 2 dimensional
        dataB = np.array([func(z) for z in samplesB])
        dataC = np.empty((dim, *dataA.shape))
        for i in range(dim):
            samplesCi = samplesB.copy()
            samplesCi[:,i] = samplesA[:,i].copy()
            dataC[i] = np.array([func(z) for z in samplesCi])
    return dataA, dataB, dataC


def evaluate_sensitivity_indices(func, Ns, jpdf, sample_method, eval_mode=None):
    samplesA, samplesB, samplesC = generate_sample_matrices(Ns, jpdf, sample_method=sample_method)
    y_a, y_b, y_c = evaluate_samples(func, samplesA, samplesB, eval_mode=eval_mode)
    s_i, s_t = calculate_sensitivity_indices(y_a, y_b, y_c)
    return s_i, s_t


def bootstrap_sensitivity_indices(y_a, y_b, y_c, n_bootstraps):
    n_bootstraps = 10
    S_bs = []
    S_T_bs = []
    sample_size = y_a.shape[0]
    for i in range(n_bootstraps):
        slice_bootstrap = np.random.choice(sample_size, sample_size)
        s_m, s_t = calculate_sensitivity_indices(y_a[slice_bootstrap],
                                                 y_b[slice_bootstrap],
                                                 y_c[:,slice_bootstrap,:])
        
        S_bs.append(s_m)
        S_T_bs.append(s_t)
    return np.mean(S_bs, axis=0), np.std(S_bs, axis=0), np.mean(S_T_bs, axis=0), np.std(S_T_bs, axis=0)


class AdaptiveMonteCarlo:
    def __init__(self):
        self.max_samples = None
        self.sample_size_step = 100


    def increase_sample_size():
        pass
