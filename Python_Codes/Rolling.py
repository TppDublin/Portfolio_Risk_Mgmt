import Portfolio_Calculation as pc 
import pandas as pd
import numpy as np 
import yfinance as yf
import Mean_varcovar as mv
import part_a as a
import matplotlib.pyplot as plt

#ticker = ['AMZN','CAT','SHW','JPM','LLY','META','MSFT','NEE','WMT','PLD','COP']

ticker=['IEF','TLT','SCHO','SCHR','BIV','BND','CAT','LLY','WMT']
#ticker=['IEF','TLT','SCHO','SCHR','BIV','BND']

data = yf.download(tickers = ticker, start='2014-03-26', end ='2024-03-27')['Adj Close']
log_ret = np.log(data/data.shift(1)).dropna()


mean, std, cov = mv.mean_std_covar(log_ret)
maxSR_ret,maxSR_std,minvar_ret,minvar_std, effireturn, efficintvol,maxSR_weight = pc.CalculatedResult(mean, cov , risk_free=0)
log_ret['portfolio'] = log_ret.dot(np.array(maxSR_weight['Weights']))


window_size = 1260

def rolling_slicing(data, window_size):
    result = []
    for i in range(len(data) - window_size + 1):
        result.append(data.iloc[i:i+window_size])

    return result

def non_parametric(roll_window,data):

    h_var = []
    h_var_es = []
    age_var = []
    age_es = []
    vol_var = []
    vol_var_es = []
    bs_var = []
    bs_var_es = []

    for window_data in roll_window:
        age , age_e = a.age_weighted(log_ret['portfolio']*1000000)
        age_var.append(age)
        age_es.append(age_e)

        bs, bs_cvar = a.bootstrap_var(log_ret['portfolio']*1000000)
        bs_var.append(bs)
        bs_var_es.append(bs_cvar)

        h_v = a.historicalVar(window_data['portfolio']*1000000)
        h_var.append(h_v)

        h_es = a.historical_CVar(window_data['portfolio']*1000000)
        h_var_es.append(h_es)

        v_var, v_es = a.volatility_weighted(window_data['portfolio']*1000000)
        vol_var.append(v_var)
        vol_var_es.append(v_es)
    
    data = data.iloc[window_size-1:]


    dic = {'Historical_Var':h_var, 'Historical_ES':h_var_es,
           'Age_weighted_Var':age_var, 'Age_weighted_ES' : age_es,
            'Volatility_Var' : vol_var, 'Volatility_ES' : vol_var_es,
            'Bootstrap_Var': bs_var, 'Bootstrap_ES': bs_var_es,
              'portfolio':data['portfolio']*1000000 }
    
    df = pd.DataFrame(dic, index=data.index)
    ax = df[['portfolio','Historical_Var','Age_weighted_Var','Volatility_Var','Bootstrap_Var']].plot(figsize=(12, 8), linewidth=2)
    #ax = df[['portfolio','Historical_ES','Age_weighted_ES','Volatility_ES','Bootstrap_ES']].plot(figsize=(12, 8), linewidth=2)


# Customizing the plot
    ax.set_title('Risk Metrics Comparison', fontsize=16)  # Adding a title
    ax.set_xlabel('Date', fontsize=14)  # Adding label for x-axis
    ax.set_ylabel('Value', fontsize=14)  # Adding label for y-axis
    ax.grid(True, linestyle='--', alpha=0.7)  # Adding gridlines with transparency

    # Customizing legend
    ax.legend(fontsize=12)

    # Adjusting ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=12)  # Increasing tick label size

    # Adding a background color
    ax.set_facecolor('whitesmoke')

    # Save the plot as an image (optional)
    plt.savefig('risk_metrics_comparison.png', dpi=300, bbox_inches='tight')
   
    # Show the plot
    plt.show()


    
    df.to_csv("Non_parametric_bond_equity.csv")

    return h_var,h_var_es,vol_var,vol_var_es,bs_var,bs_var_es


def parametric(roll_window, data):
    n_var = []
    n_es = []
    t_var = []
    t_es = []


    for window_data in roll_window:
        #print(window_data.drop(columns = ['portfolio']))
        mean,std,cov = mv.mean_std_covar(window_data.drop(columns = ['portfolio']))
        weight = a.maxSR_weight['Weights']

        ret, std = mv.portfolio_performance(weight,mean,cov)

        n_v = a.var_parametric(ret,std, dof=window_size)
        n_t = a.var_parametric(ret,std,distribution='t-distribution',dof=window_size)
        n_var.append(-n_v*1000000)
        t_var.append(-n_t*1000000)

        n_s = a.cvar_parametric(ret,std,dof=window_size)
        t_s = a.cvar_parametric(ret,std, distribution='t-distribution', dof=window_size)

        n_es.append(-n_s*1000000)
        t_es.append(-t_s*1000000)


    data = data.iloc[window_size-1:]
    dic = {'Normal_Var':n_var, 'Normal_ES':n_es, 'T_Var' : t_var, 'T_ES' : t_es, 'portfolio':data['portfolio']*1000000 }
    df = pd.DataFrame(dic, index=data.index)
    ax = df.plot(figsize=(12, 8), linewidth=2)

# Customizing the plot
    ax.set_title('Risk Metrics Comparison', fontsize=16)  # Adding a title
    ax.set_xlabel('Date', fontsize=14)  # Adding label for x-axis
    ax.set_ylabel('Value', fontsize=14)  # Adding label for y-axis
    ax.grid(True, linestyle='--', alpha=0.7)  # Adding gridlines with transparency

    # Customizing legend
    ax.legend(fontsize=12)

    # Adjusting ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=12)  # Increasing tick label size

    # Adding a background color
    ax.set_facecolor('whitesmoke')

    # Save the plot as an image (optional)
    plt.savefig('risk_metrics_comparison.png', dpi=300, bbox_inches='tight')
   
    # Show the plot
    plt.show()
    df.plot()
    df.to_csv("Parametric_Bond_equity.csv")

    return n_var,n_es,t_var,t_es


def sensitivity(ticker,data):
    num_tickers = len(ticker)
    equal_weight = 1 / num_tickers
    weights = [equal_weight] * num_tickers
    data['portfolio'] = data.dot(np.array(weights))
    print(data['portfolio'])

    age_var, age_es = a.age_weighted(data['portfolio']*1000000)
    vol_var, vol_es = a.volatility_weighted(data['portfolio']*1000000)

    print('Sensitivity Calculation Age Weighted Var', age_var )
    print('Sensitivity Calculation Age Weighted ES', age_es )
    print('Sensitivity Calculation Volatility Weighted Var', vol_var)
    print('Sensitivity Calculation Volatility Weighted Var', vol_es)




roll_window = rolling_slicing(log_ret,window_size=window_size)
h_var,h_var_es,vol_var,vol_var_es,bs_var,bs_var_es = non_parametric(roll_window,log_ret)

#print(len(h_var))
#print(len(vol_var))
#print(len(bs_var))

#sensitivity(ticker,log_ret.drop(columns  = ['portfolio']))
n_var,n_es,t_var,t_es = parametric(roll_window, log_ret)



