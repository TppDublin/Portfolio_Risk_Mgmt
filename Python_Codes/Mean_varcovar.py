
import pandas as pd
import numpy as np


def mean_std_covar(data):
    #f_sample,s_sample = self.split()
    sample_mean = data.mean()
    sample_std = data.std()        
    sample_cov = data.cov()

    # Converting it to array
    #f_mean = np.array(sample_mean).reshape(len(sample_mean),1)
    #f_cov = np.array(sample_cov)
    
    return sample_mean,sample_std, sample_cov



def portfolio_performance(weights, mean, cov, time = 1):
    weights = np.array(weights)
    returns = np.sum(mean*weights)*time
    std = np.sqrt(np.dot(weights.T,np.dot(cov,weights)))*np.sqrt(time)
    return returns, std


