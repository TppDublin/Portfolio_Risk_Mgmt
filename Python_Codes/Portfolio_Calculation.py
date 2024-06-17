import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import numpy as np
import Mean_varcovar as mv
import pandas as pd
import plotly.graph_objects as go
#ticker = ['AMZN','CAT','SHW','JPM','LLY','META','MSFT','NEE','WMT','PLD','COP']

ticker=['IEF','TLT','SCHO','SCHR','BIV','BND','CAT','LLY','WMT']
#ticker=['IEF','TLT','SCHO','SCHR','BIV','BND']

# Optimize for maximum sharpe ratio portfolio 

def portfolioVariance(weights, mean,cov):
    return mv.portfolio_performance(weights,mean,cov)[1]



def  negative_sharpe_ratio(weights, mean, cov , risk_free = 0.05):
    ret,std = mv.portfolio_performance(weights=weights, mean = mean, cov =cov)
    return -(ret - risk_free)/std


# Maximize sharpe ratio
def maxSR(mean, cov ,risk_free, constraintset = (0.05,1)):
    # minimize the negative sharpe ratio by changing weights
    num_assets = len(mean)
    args = (mean, cov , risk_free)
    constraints = ({'type':'eq','fun': lambda x : np.sum(x)-1})
    
    bound = constraintset
    bounds  = tuple(bound for asset in range(num_assets))
    results = minimize(negative_sharpe_ratio,num_assets*[1./num_assets], args = args, method='SLSQP',constraints= constraints, bounds=bounds)

    return results

# Minimize portfolio variace
def minVar(mean, cov , constraintset = (0.05,1)):
    num_assets = len(mean)
    args = (mean, cov)
    constraints = ({'type':'eq','fun': lambda x : np.sum(x)-1})
    bound = constraintset
    bounds  = tuple(bound for asset in range(num_assets))
    results = minimize(portfolioVariance,num_assets*[1./num_assets], args = args, method='SLSQP',constraints= constraints, bounds=bounds)

    return results

# Frontier
def efficinet_frontier(mean, cov , returnIarget, constraintset = (0.05,1)):
    ''' For each return target we want to optimize for min var
    '''
    def portfolioReturn(weights):
        return mv.portfolio_performance(weights, mean,cov)[0]
    
    num_assets = len(mean)
    args = (mean, cov)

    constraints = ({'type':'eq','fun': lambda x : np.sum(x)-1},
                   {'type':'eq','fun': lambda x : portfolioReturn(x)- returnIarget}
                   
                   )
    bound = constraintset
    bounds  = tuple(bound for asset in range(num_assets))
    effi =  minimize(portfolioVariance,num_assets*[1./num_assets], args = args, method='SLSQP',constraints= constraints, bounds=bounds)

    return effi


# Calculation
def CalculatedResult(mean,cov,risk_free = 0.05, constraintset = (0,1)):
    ''' Read mean, cov matrix and other financial information 
        Output , Max Sr, Minvol, efficient frontier                
    '''
    # Tangency Portfolio
    maxSR_output = maxSR(mean, cov, risk_free=risk_free)
    maxSR_ret, maxSR_std = mv.portfolio_performance(maxSR_output['x'], mean, cov)

    maxSR_weight = { 'Ticker' : ticker, 'Weights' : maxSR_output['x']
    }
    maxSR_weight = pd.DataFrame(maxSR_weight)
    maxSR_weight.Weights = [round(i,2) for i in maxSR_weight.Weights]
    print('Portfolio Return (Tangency):', maxSR_ret)
    print('Portfolio Std (Tangency) :', maxSR_std)
    print('Portfolio Sharpe Ratio (Tangency) :', (maxSR_ret - risk_free)/maxSR_std)
    print('------------------------------------------------------')
    print('Portfolio Weight : ', maxSR_weight)

    # Minimum Variance portfolio
    minvar_output = minVar(mean,cov)
    minvar_ret, minvar_std = mv.portfolio_performance(minvar_output['x'], mean, cov)

    # Efficient Frontier
    efficintvol = []
    effireturn = []
    target_return = np.linspace(minvar_ret, maxSR_ret+0.03, 20)
    for target in target_return:
        effireturn.append(target)
        efficintvol.append(efficinet_frontier(mean, cov, target)['fun'])


    return maxSR_ret,maxSR_std,minvar_ret,minvar_std, effireturn, efficintvol, maxSR_weight



def graph(maxSR_ret,maxSR_std,minvar_ret,minvar_std, effireturn, efficintvol):
    # Max SR
    MaxSharpeRatio = go.Scatter(
    name = 'Maximum Sharpe Ratio',
    mode = 'markers',
    x = [maxSR_std],
    y = [maxSR_ret],
    marker = dict(color = 'red' , size = 14 ,line = dict(width = 3, color = 'black'))
    )

    Mixvol = go.Scatter(
    name = 'Minimum volatility',
    mode = 'markers',
    y = [minvar_ret],
    x = [minvar_std],
    marker = dict(color = 'green' , size = 14 ,line = dict(width = 3, color = 'black'))

    )

    ef = go.Scatter(
    name = 'Efficient Frontier',
    mode = 'lines',
    y = [round(ret,5) for ret in effireturn],
    x = [round(vol ,5) for vol in efficintvol],
    line = dict(color = 'black' , width = 4, dash = 'dashdot')

    )

    data = [MaxSharpeRatio, Mixvol, ef]
    layout = go.Layout(
        title = 'Efficient frontier',
        yaxis = dict(title = ' Expected Return annulised'),
        xaxis = dict(title =' Volatility annulised'),
        showlegend=True,
        legend= dict(
            x = 0.75 , y =0, traceorder = 'normal',
            bgcolor= '#E2E2E2',
            bordercolor = 'black'
        ),
        width= 800,
        height= 600   

    )
    fig = go.Figure(data= data, layout=layout)
    fig.show()


