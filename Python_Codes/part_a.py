import Portfolio_Calculation as pc
import pandas as pd 
import yfinance as yf
import numpy as np
import Mean_varcovar as mv
from scipy.stats import norm, t
from scipy.interpolate import interp1d
from arch import arch_model
import matplotlib.pyplot as plt 

#ticker = ['AMZN','CAT','SHW','JPM','LLY','META','MSFT','NEE','WMT','PLD','COP']

ticker=['IEF','TLT','SCHO','SCHR','BIV','BND','CAT','LLY','WMT']
#ticker=['IEF','TLT','SCHO','SCHR','BIV','BND']


# Parametric Risk measures for Var 
def var_parametric(port_ret, port_std, distribution = 'normal', alpha = 1,dof = 2516):
    '''
    Calculate the portfolio Var given a distribution, with known parameters
    '''

    if distribution == 'normal':
        Var = norm.ppf(1-alpha/100)*port_std - port_ret
        #Var = norm.ppf(0.01, port_ret,port_std)
    
    elif distribution =='t-distribution':
        Var = np.sqrt((dof-2)/dof)*t.ppf(1-alpha/100, dof)*port_std - port_ret

    return Var

def cvar_parametric(port_ret, port_std, distribution = 'normal', alpha = 1,dof = 2516):
    '''
    Calculate the portfolio ES also known as Conditional Var given a distribution, with known parameters
    '''

    if distribution == 'normal':
        CVar = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*port_std - port_ret
        #Var = norm.ppf(0.01, port_ret,port_std)
    
    elif distribution =='t-distribution':
        x_anu = t.ppf(alpha/100, dof) 
        CVar = -1/(alpha/100)*(1-dof)**-1* (dof - 2 + x_anu**2) * t.pdf(x_anu, dof)*port_std - port_ret
    return CVar




data = yf.download(tickers = ticker, start='2014-03-26', end ='2024-03-27')['Adj Close']

log_ret = np.log(data/data.shift(1)).dropna()

mean, std, cov = mv.mean_std_covar(log_ret)
maxSR_ret,maxSR_std,minvar_ret,minvar_std, effireturn, efficintvol,maxSR_weight = pc.CalculatedResult(mean, cov , risk_free=0)
#pc.graph(maxSR_ret,maxSR_std,minvar_ret,minvar_std, effireturn, efficintvol)
'''
#------------------------------Parametric--------------------------------------------------------------------------------
var = var_parametric(maxSR_ret,maxSR_std, distribution= 'normal')
t_var = var_parametric(maxSR_ret,maxSR_std, distribution= 't-distribution')

print('Var for normal distribution :', var)
print('Var for T distribution :', t_var)
print('Port - retrun :', maxSR_ret)

cvar = cvar_parametric(maxSR_ret,maxSR_std, distribution= 'normal')
t_cvar = cvar_parametric(maxSR_ret,maxSR_std, distribution= 't-distribution')

print('ES for normal distribution :', cvar)
print('ES for T distribution :', t_cvar)
print('Port - retrun :', maxSR_ret)


portfolio_value = 1000000
confidence_level = 0.99
alpha = 1 - confidence_level

# Monte Carlo Simulation VaR and ES
num_simulations = 10000
simulated_returns = np.random.normal(maxSR_ret, maxSR_std, num_simulations)
# For a 99% VaR, use the 1st percentile of losses (since the lower tail represents losses)
VaR_MC = -np.percentile(simulated_returns, alpha * 100) 
ES_MC = -simulated_returns[simulated_returns <= np.percentile(simulated_returns, alpha * 100)].mean() 

print("Monte Carlo Simulation Results:")
print("VaR_MC:", VaR_MC)
print("ES_MC:", ES_MC)
'''
# ------------------------------------Non - Parametric -------------------------------------------------------------------

def historicalVar(returns , alpha = 1):
    # Output the percentile of the distribution at the given confidence level
    
    return np.percentile(returns , alpha)

def historical_CVar(returns , alpha = 1):
    #Output the percentile of the distribution at the given confidence level
    
    belowvar = returns <= historicalVar(returns, alpha=alpha)
    return returns[belowvar].mean()

def age_weighted(returns , alpha = 0.01):
    lam = 0.95
    n = len(returns)
    wts = [(lam**(i-1) * (1-lam))/(1-lam**n) for i in range(1, n+1)] 
    ret = returns[::-1]
    weights_dict = {'Returns': ret, 'Weights':wts}
    wts_returns = pd.DataFrame(weights_dict)
    sort_wts = wts_returns.sort_values(by='Returns')
    sort_wts['Cumulative'] = sort_wts.Weights.cumsum()
    sort_wts = sort_wts.reset_index().drop(columns=['Date'])
    id = sort_wts[sort_wts.Cumulative <= alpha].Returns.idxmax()
    final_val = sort_wts.loc[id:id+1]

    xp = sort_wts.loc[id:id+1, 'Cumulative'].values
    fp = sort_wts.loc[id:id+1, 'Returns'].values
    VaR_weighted = np.interp(alpha, xp, fp)

    loss = returns[returns <= VaR_weighted]
    print('Alpha :', alpha)
    print("ES Age_weighted",loss.mean())
    print("Var age weighted",VaR_weighted)

    return VaR_weighted,loss.mean()
  


def volatility_weighted(returns, alpha = 1):
    vols = pd.DataFrame(index = returns.index)
    am = arch_model(returns, rescale=False)
    res = am.fit(disp='off')
    vols['vol_garch'] = res.conditional_volatility

    new = vols['vol_garch'].apply(lambda x: 0.95*vols['vol_garch'].iloc[-1] / x)

    vol_ad_port = returns*new

    Var_vol_weighted = np.percentile(vol_ad_port,alpha)

    loss = returns[returns < Var_vol_weighted]

    print('Alpha :', alpha/100)
    print("Volatility weighted Var :",Var_vol_weighted)
    print("Volatility weighted ES",loss.mean())
    return Var_vol_weighted, loss.mean()


def bootstrap_var(returns, alpha =1):
    num_bootstraps = 100
    bootstrapped_VaR = []
    bootstrapped_Cvar = []

    for _ in range(num_bootstraps):
        bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrapped_VaR.append(np.percentile(bootstrap_sample, alpha))

    for var in bootstrapped_VaR:
        loss = returns[returns < var]
        bootstrapped_Cvar.append(np.mean(loss))

    return np.array(bootstrapped_VaR).mean(), np.mean(bootstrapped_Cvar)


#-------------------------------------------------------------------------------------------------------------------------

log_ret['portfolio'] = log_ret.dot(np.array(maxSR_weight['Weights']))
#bs_var, bs_cvar = bootstrap_var(log_ret['portfolio']*1000000)

#print(" ",bs_var)

#-------------------------------------------------------------------------------------------------------------------------

# In- Sample sclicing
# ----------------------------Non Parametric -----------------------------------------------------------------------------
#insample_log_ret = log_ret['portfolio'].loc['2014-03-27':'2021-03-26']
#bs_var, bs_cvar = bootstrap_var(log_ret['portfolio']*1000000)
#print("Bootstrap VAR :", bs_var)
#print("Bootstrap CVAR :", bs_cvar)

#print("Historical Var :", historicalVar(log_ret['portfolio']*1000000))
#print("Historical ES :", historical_CVar(log_ret['portfolio']*1000000))

age_weighted(log_ret['portfolio']*1000000)
volatility_weighted(log_ret['portfolio']*1000000)
# --------------------------------------- Parametric ---------------------------------------------------------------------
