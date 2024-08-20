""" 
Script to compute the variance of a NIVAR estimator for differing levels of within-block sparsity of the 
ground truth. Plots the results to line plot pdf.
"""

#!/usr/bin/env python3
# USAGE: ./multipleN_variance_simulation.py 


import numpy as np
from numpy.random import default_rng
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go
from src.visualization import utility_funcs
from src.models import generativeVAR


##### PARAMETERS ######
T = 5000
Q=1
SEED = 94032
random_state = default_rng(seed=SEED) 
spectral_radius = 0.9
B=5
d = 5
n_iter = 7 
target_feature = 0 
N_gaussian_mixtures = 5
p_out = 0
var = 1 
N_list = [50,100,150] 

# Specify the values of p_in you wish to look as 
start = 0.3
end = 1.05  # 1 + 0.05 to include 1 in the list
step = 0.05
values = [round(x, 2) for x in range(int(start*100), int(end*100), int(step*100))]
p_in_list = [x/100 for x in values]
num_ps = len(p_in_list)
percentage_sparsity_list = [(100-i*100) for i in p_in_list]

num_Ns = len(N_list) 

variances = np.zeros((num_Ns,num_ps))
for j, N in enumerate(N_list): 

    # Fix A_hat 
    # phi_dist = np.ones((N,N))
    phi_dist = random_state.uniform(low=0.0, high=1.0, size=(N, N))
    generator = generativeVAR.generativeVAR(random_state=random_state,
                                            N=N,
                                            T=T,
                                            Q=Q,
                                            multiplier=spectral_radius,
                                            B=B,
                                            p_in=1,
                                            p_out=p_out,
                                            phi_distribution=phi_dist
                                            )
    A_hat = generator.adjacency_matrix.reshape((N,N))
    phi_full = generator.phi_coefficients.reshape((N,N))
    R_hat = utility_funcs.get_R(A_hat)

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

        phi = np.where(A==1,phi_full,0) 

        Gamma = utility_funcs.get_gamma(phi=phi,noise_variance=var) 
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

    variances[j] = store_diagonal_variances

print(variances)

###### SAVE VARIANCE RATIOS TO CSV FILE ######
np.savetxt('variance_ratios_multipleN.csv', variances, delimiter=',')

    
###### PLOT LINE PLOT ###### 
fig = go.Figure()

markers = ["circle", "square", "diamond", "cross", "triangle-up"]  # Define different markers
colors = px.colors.qualitative.Set1  # Use a predefined color set

for j, N in enumerate(N_list):
    fig.add_trace(go.Scatter(
        x=percentage_sparsity_list[2:], 
        y=variances[j][2:], 
        mode='lines+markers',
        name=f'N={N}',  # Add N value to the legend
        marker=dict(symbol=markers[j % len(markers)], size=8),  # Use different markers
        line=dict(color=colors[j % len(colors)])  # Use different colors
    ))

# Set the title and axis labels
fig.update_layout(
    xaxis_title='Sparsity level (percentage)',
    yaxis_title=r'$\alpha_{V}$'
)

layout = go.Layout(
    yaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(showline=True, dtick=10, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif",
    font_size=16,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)
fig.update_layout(layout)

fig.write_image(f"ratio_of_variances_multiN.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"ratio_of_variances_multiN.pdf", format='pdf')