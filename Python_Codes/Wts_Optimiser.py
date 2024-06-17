
import numpy as np
import pandas as pd
from scipy.optimize import minimize 


def port_return(mean,cov_arr,stepsize,min,max):

    #ef portfolio_variance(weights):
     #   return weights.T @ cov_arr @ weights
        

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    initial_guess = np.array(len(mean) * [1. / len(mean)])

    target_returns = np.linspace(min,max , stepsize) # since it is taking time, I have kept the observation less
    frontier_volatilities = []

    for return_target in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'eq', 'fun': lambda x: x.T @ mean - return_target})
               #{'type': 'ineq', 'fun': lambda x: x})

        result = minimize(lambda x: x.T @ cov_arr @ x, initial_guess, method='SLSQP', constraints=constraints)
        #print(result)
        frontier_volatilities.append(np.sqrt(result.fun))
    
    return target_returns,frontier_volatilities


