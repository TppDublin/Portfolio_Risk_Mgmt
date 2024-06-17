import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt
import os
import part_a as pa

'''
data_path = "E:\\2nd Trimester\\Portfolio Management\\Risk Management\\Project files\\stock_data.csv"
output_dir = "E:\\2nd Trimester\\Portfolio Management\\Risk Management\\Project files\\Parametric VaR and ES\\"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Calculate the log returns
log_returns = np.log(df / df.shift(1)).dropna()
'''
# Portfolio value
portfolio_value = 1000000  # $1 million investment

# Confidence level for VaR and ES
confidence_level = 0.99
alpha = 1 - confidence_level

'''
# Mean and standard deviation of the log returns
mu = log_returns.mean(axis=1)
sigma = log_returns.std(axis=1)
print(mu)
print(sigma)
'''
# Variance-Covariance Method VaR and ES
z_score = norm.ppf(confidence_level)
VaR_VC = -z_score * pa.maxSR_std * portfolio_value
print(VaR_VC)
ES_VC = -(mu.mean() + sigma.mean() * norm.pdf(z_score) / alpha) * portfolio_value
'''
# Monte Carlo Simulation VaR and ES
num_simulations = 10000
simulated_returns = np.random.normal(mu.mean(), sigma.mean(), num_simulations)
# For a 99% VaR, use the 1st percentile of losses (since the lower tail represents losses)
VaR_MC = -np.percentile(simulated_returns, 100 - (alpha * 100)) * portfolio_value
ES_MC = -np.mean(simulated_returns[simulated_returns <= -VaR_MC / portfolio_value]) * portfolio_value


# Cornish-Fisher VaR
z_cf = z_score + (1/6) * skew(log_returns.sum(axis=1)) * (z_score**2 - 1) + \
       (1/24) * (kurtosis(log_returns.sum(axis=1)) - 3) * (z_score**3 - 3*z_score) - \
       (1/36) * skew(log_returns.sum(axis=1))**2 * (2*z_score**3 - 5*z_score)
VaR_CF = -(mu.mean() + sigma.mean() * z_cf) * portfolio_value

# Function to plot and save histogram for VaR
def plot_var_histogram(data, method, VaR, filename, bins=100):
    plt.figure(figsize=(10, 6))
    data_range = np.percentile(data, [1, 99])  # Focusing on 1st to 99th percentile data
    hist_range = (data_range[0] * 1.5, data_range[1] * 1.5)  # Extend the range slightly
    plt.hist(data, bins=bins, range=hist_range, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(VaR, color='red', linestyle='dashed', linewidth=2, label=f'VaR: {VaR:.2f}')
    plt.title(f'{method} Method VaR at {confidence_level*100}% Confidence Level')
    plt.xlabel('Log Returns' if method != 'Monte Carlo' else 'Simulated Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot histograms for VaR for each method
plot_var_histogram(log_returns.sum(axis=1), 'Variance-Covariance', VaR_VC/portfolio_value, 'var_vc_histogram.png')
plot_var_histogram(simulated_returns, 'Monte Carlo', VaR_MC/portfolio_value, 'var_mc_histogram.png')
plot_var_histogram(log_returns.sum(axis=1), 'Cornish-Fisher', VaR_CF/portfolio_value, 'var_cf_histogram.png')


# Save the VaR and ES results to CSV
results = pd.DataFrame({
    'Method': ['Variance-Covariance', 'Monte Carlo', 'Cornish-Fisher'],
    'VaR': [VaR_VC, VaR_MC, VaR_CF],
    'ES': [ES_VC, ES_MC, 'N/A']  # ES for Cornish-Fisher not calculated
})
results.to_csv(os.path.join(output_dir, 'pvar_es_results.csv'), index=False)

# Print the VaR and ES results
print(results)
'''