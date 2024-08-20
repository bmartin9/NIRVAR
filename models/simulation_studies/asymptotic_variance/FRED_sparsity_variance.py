""" 
Script to compute the variance of a NIVAR estimator for differing levels of within-block sparsity of the 
FRED-MD "ground truth". Plots the results to line plot pdf.
"""

#!/usr/bin/env python3
# USAGE: ./FRED_sparsity_variance.py ../../../data/processed/FRED-MD/fred-balanced.csv ../../../data/generated/FRED-MD/NIRVAR/labels_corr.csv


import numpy as np
from numpy.random import default_rng
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go
from src.visualization import utility_funcs
from src.models import generativeVAR
import sys

##### PARAMETERS ######
Q=1
SEED = 4460
random_state = default_rng(seed=SEED) 
spectral_radius = 0.9
B=5
d = 5
n_iter = 7 
target_feature = 0 
N_gaussian_mixtures = 5
p_out = 0
var = 1 
backtesting_day_labels = 0
first_backtest_month = 480
lookback_window = 480 

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols=range(1,123))
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N) 

# Fix A_hat using the NIRVAR estimated FRED-MD groups
labels = np.genfromtxt(sys.argv[2],delimiter=",")[backtesting_day_labels]
A_hat = utility_funcs.groupings_to_2D(labels)
R_hat = utility_funcs.get_R(A_hat)

# Specify the values of p_in you wish to look as 
start = 0.3
end = 1.05  # 1 + 0.05 to include 1 in the list
step = 0.05
values = [round(x, 2) for x in range(int(start*100), int(end*100), int(step*100))]
p_in_list = [x/100 for x in values]
num_ps = len(p_in_list)
percentage_sparsity_list = [(100 - i*100) for i in p_in_list]

###### Estimate Gamma using data ######
X_backtesting = Xs[first_backtest_month-lookback_window:first_backtest_month+1] 
Gamma = X_backtesting.T@X_backtesting/lookback_window

store_diagonal_variances = np.zeros((num_ps))
for i, p in enumerate(p_in_list): 
    print(p)
    A = A_hat.copy()
    # Find the indices where the value is 1
    indices = np.argwhere(A == 1)

    # Filter out the diagonal indices (i.e., where row == column)
    off_diagonal_indices = indices[indices[:, 0] != indices[:, 1]]

    # Determine how many off-diagonal indices you need to change (10% of the off-diagonal 1s)
    num_to_change = int((1-p) * len(off_diagonal_indices))

    # Randomly select the off-diagonal indices to change
    indices_to_change = random_state.choice(len(off_diagonal_indices), size=num_to_change, replace=False)

    # Change the selected off-diagonal indices from 1 to 0
    A[off_diagonal_indices[indices_to_change, 0], off_diagonal_indices[indices_to_change, 1]] = 0

    unrestricted_variance = np.kron(Gamma,var*np.identity(N))
    restricted_variance_inv = R_hat.T@unrestricted_variance@R_hat
    restricted_variance = np.linalg.inv(restricted_variance_inv) # shape = (M_hat,M_hat) 
    restricted_variance_N_space_R_hat = R_hat@restricted_variance@R_hat.T 

    R = utility_funcs.get_R(A) 
    restricted_variance_inv_R = R.T@unrestricted_variance@R
    restricted_variance_R = np.linalg.inv(restricted_variance_inv_R) # shape = (M,M) 
    restricted_variance_N_space_R = R@restricted_variance_R@R.T 

    measure = np.trace(restricted_variance_N_space_R_hat)/np.trace(restricted_variance_N_space_R)

    store_diagonal_variances[i] = measure

print(store_diagonal_variances)

###### SAVE VARIANCE RATIOS TO CSV FILE ######
np.savetxt('variance_ratios_FRED.csv', store_diagonal_variances, delimiter=',', fmt='%d')

# store_diagonal_variances = np.array([133.76891731 ,117.40013067  ,14.35871779 , 14.07329704 ,  7.74235662 , 5.19592178 ,  2.73541633 ,  2.69188702  , 1.47810436 ,  3.80806322 ,1.60654495 ,  1.45981058 ,  1.74764567 ,  1.43468436  , 1.        ])  
# store_diagonal_variances = np.array([15.09412555 ,15.54891539, 14.84567127 , 7.88519352,  5.33269451,  5.16560115, 3.93751942 , 5.05169452,  2.28574826 , 1.99866704 , 1.45588599,  1.4622407, 1.44568546,  1.06949187,  1.        ])
    
###### PLOT LINE PLOT ###### 
fig = go.Figure()
fig.add_trace(go.Scatter(
        x=percentage_sparsity_list[2:], 
        y=store_diagonal_variances[2:], 
        mode='lines+markers', 
        marker=dict(symbol="circle", size=8)  # Use a different marker for each cluster
    ))

# Set the title and axis labels
fig.update_layout(
                    xaxis_title='Sparsity level (percentage)',
                    yaxis_title=r'$\alpha_{V}$',
                    xaxis=dict(
                        dtick=10,  # Set x-axis tick interval to 10
                        showline=True, 
                        linewidth=1, 
                        linecolor='black', 
                        ticks='outside', 
                        mirror=True, 
                        automargin=True
                    ),
                    yaxis=dict(
                        showline=True, 
                        linewidth=1, 
                        linecolor='black', 
                        ticks='outside', 
                        mirror=True
                    )
)

# Set the layout properties
layout = go.Layout(
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=18, 
    margin=dict(l=5, r=5, t=5, b=5),
    width=500, 
    height=350
)
fig.update_layout(layout)

# Save the figure
fig.write_image("FRED_ratio_of_variances.pdf", format='pdf')
time.sleep(1)
fig.write_image("FRED_ratio_of_variances.pdf", format='pdf')

