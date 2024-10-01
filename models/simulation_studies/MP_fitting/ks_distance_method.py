""" 
Find the scale parameter of the Marcenko-Pastur distribution that best fits the data 
by computing the K-S distance between the empirical cdf and the Marcenko-Pastur cdf
"""

#!/usr/bin/env python3
# USAGE: ./ks_distance_method.py <DATA.csv> 

import numpy as np
from numpy.random import default_rng
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from scipy.integrate import quad
from scipy.optimize import minimize
from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
from sklearn.preprocessing import MinMaxScaler
import time


SEED = 342
random_state = default_rng(seed=SEED)
num_permutations = 100
Q=1

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols=range(1,123))
T = Xs.shape[0]
N = Xs.shape[1] 
print(f"Xs shape : {Xs.shape}")

scaler = MinMaxScaler(feature_range=(-1,1)) 
scaler.fit(Xs) 
Xs = scaler.transform(Xs)
X_train_mean = np.mean(Xs,axis=0)
Xs -= X_train_mean

eta = N/T 

###### Compute eigenvalues of randomised covariance matrix ######
random_permutations = random_state.permuted(random_state.permuted(np.array(np.hsplit(np.tile(Xs,num_permutations),num_permutations)),axis=1),axis=2)
eigenvalues = np.zeros((num_permutations,N))
# eigenvalues = np.zeros((num_permutations,T))
for i in range(num_permutations):
    # S_N = np.corrcoef(random_permutations[i].T)   
    S_N = np.cov(random_permutations[i].T)
    eigenvalues[i] = np.real(np.linalg.eigvals(S_N))
flat_eigenvalues = eigenvalues.flatten() 


def ks_statistic(sigma_squared, data, q):
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    mpl = MarchenkoPasturDistribution(beta=1, ratio=q, sigma=np.sqrt(sigma_squared) )
    theoretical_cdf = mpl.cdf(np.sort(data))
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    return ks_stat

def exponential_weight_function(empirical_cdf, alpha=5):
    return np.exp(alpha * empirical_cdf)

# Weighted KS statistic function with exponential weights
def weighted_ks_statistic(sigma_squared, data, q, alpha=10):
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    mpl = MarchenkoPasturDistribution(beta=1, ratio=q, sigma=np.sqrt(sigma_squared) )

    theoretical_cdf = mpl.cdf(sorted_data)
    
    # Calculate weights using the exponential weight function
    weights = exponential_weight_function(empirical_cdf, alpha)
    
    # Calculate weighted discrepancies
    weighted_discrepancies = weights * np.abs(empirical_cdf - theoretical_cdf)
    
    # Calculate weighted KS statistic
    ks_stat = np.sum(weighted_discrepancies)
    return ks_stat

initial_guess = np.var(flat_eigenvalues)

# Perform optimization to minimize the KS statistic
# result = minimize(ks_statistic, initial_guess, args=(flat_eigenvalues, eta), bounds=[(0.01, 0.3)])
result = minimize(weighted_ks_statistic, initial_guess, args=(flat_eigenvalues, eta), bounds=[(0.01, 0.3)])
sigma_squared_ks = result.x[0] 

print("Estimated Scale Parameter (σ^2) using KS Test:", sigma_squared_ks)
estimated_cutoff = sigma_squared_ks*(1 + np.sqrt(eta))**2
print(f"estimated cuttoff : {estimated_cutoff}")
print(f"The upper bound of the support with this estimated sigma is: {estimated_cutoff}") 

###### Compute d using estimated sigma ######
S_N_non_random = np.cov(Xs.T) 
# S_N_non_random = np.corrcoef(Xs.T) 
# S_N_non_random = np.linalg.inv(np.cov(Xs[:,:100].T)) 
eigenvalues_non_random = np.real(np.linalg.eigvals(S_N_non_random))
d_hat = np.count_nonzero(eigenvalues_non_random[eigenvalues_non_random>estimated_cutoff])
print(f"d_hat : {d_hat}")

S_corr = np.corrcoef(Xs.T) 
evals_corr = np.linalg.eigvals(S_corr) 
estimated_cutoff_corr = (1 + np.sqrt(eta))**2
d_corr = np.count_nonzero(evals_corr[evals_corr>estimated_cutoff_corr])
print(f"d_corr : {d_corr}") 

x1 = np.linspace(0, 0.15, num=1000)
mpl = MarchenkoPasturDistribution(beta=1, ratio=N/T, sigma=np.sqrt(sigma_squared_ks))
y1 = mpl.pdf(x1)

colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]

# Create the figure
fig = go.Figure()

# Add the first histogram
fig.add_trace(go.Histogram(
    x=flat_eigenvalues,
    histnorm='probability density',
    nbinsx=200,
    name=r'Randomised Data',
    opacity=0.75
))

# Add the second histogram
# fig.add_trace(go.Histogram(
#     x=eigenvalues_non_random[eigenvalues_non_random<20],
#     histnorm='probability density',
#     nbinsx=200,
#     name='Data',
#     opacity=0.75
# ))

# Add the Marcenko-Pastur distribution line
fig.add_trace(go.Scatter(
    x=x1,
    y=y1,
    mode='lines',
    name='Best fit MP'
))

# Update layout for overlaying histograms
fig.update_layout(
    barmode='overlay',
    xaxis_title=r'$\lambda$',
    yaxis_title='Probability Density',
    legend=dict(
        orientation='h',
        y=1.02,
        x=1,
        xanchor='right',
        yanchor='bottom'
    )
)

layout = go.Layout(
    yaxis=dict(title='Probability', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title='Eigenvalue',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)

fig.update_layout(layout)

fig.write_image("MP-FRED-cov.pdf")
time.sleep(1)
fig.write_image("MP-FRED-cov.pdf") 
