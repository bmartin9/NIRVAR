""" 
Script to compute the asymptotic bias and variance of a NIRVAR estimator for differing 
levels of inter-block LASSO penalty. The ground truth p_in and p_out
are fixed. Plots the results to line plot pdf.
Work at sample level.
"""

#!/usr/bin/env python3
# USAGE: ./simulate.py 


import numpy as np
from numpy.random import default_rng
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go
from src.visualization import utility_funcs
from src.models import generativeVAR
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.linalg 

##### PARAMETERS ######
T = 10000
Q=1
SEED = 94032
random_state = default_rng(seed=SEED) 
spectral_radius = 0.9
K=5
d = K
n_iter = 7 
target_feature = 0 
p_in = 1
p_out_list = [0,0.1,0.2,0.3,0.8]
N  = 50
LASSO_weights = [0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35]  
num_percents = len(LASSO_weights)
var=1
gmm_random = 367
num_replicas = 10

num_pouts = len(p_out_list)

###### UTILITY FUNCTIONS ######
import numpy as np

def fit_masked_lasso_moment(G0, G1, labels, lam, symmetrize=False,
                            ridge=1e-8, max_iter=1000, tol=1e-6):
    """
    Solve: min_Phi tr(Phi G0 Phi^T) - 2 tr(Phi G1) + sum_{ij} w_ij |Phi_ij|,
    with w_ij = 0 if same block, lam otherwise.
    """
    N = G0.shape[0]
    # SPD + symmetry guards
    G0s = 0.5*(G0 + G0.T) + ridge*np.eye(N)
    G1s = 0.5*(G1 + G1.T) if symmetrize else G1

    # penalty weights
    lab = np.asarray(labels)
    inter = (lab[:, None] != lab[None, :])
    W = lam * inter.astype(float)  # 0 inside blocks, lam across blocks

    # Lipschitz constant of grad f(Phi)=2(Phi G0 - G1): L = 2 * ||G0||_op
    L = 2.0 * np.linalg.eigvalsh(G0s).max()
    eta = 1.0 / L

    Phi = np.zeros((N, N))
    for _ in range(max_iter):
        grad = 2.0*(Phi @ G0s - G1s)              # gradient step
        Z = Phi - eta*grad

        # entrywise soft-threshold ONLY on inter-block entries
        Phi_new = np.sign(Z) * np.maximum(np.abs(Z) - eta*W, 0.0)
        if symmetrize:
            # keep symmetry; do not penalize twice
            Phi_new = 0.5*(Phi_new + Phi_new.T)

        # convergence
        if np.linalg.norm(Phi_new - Phi, 'fro') <= tol * max(1.0, np.linalg.norm(Phi, 'fro')):
            Phi = Phi_new
            break
        Phi = Phi_new

    return Phi


variances = np.zeros((num_pouts,num_percents,num_replicas))
biases = np.zeros((num_pouts,num_percents,num_replicas))
percentage_incorrect_edges = np.zeros((num_pouts,num_percents,num_replicas))

for l in range(num_replicas):

    for j, p_out in enumerate(p_out_list):


        # Fix A
        # phi_dist = np.ones((N,N))
        phi_dist = random_state.uniform(low=0.0, high=1.0, size=(N, N))
        generator = generativeVAR.generativeVAR(random_state=random_state,
                                                N=N,
                                                T=T,
                                                Q=Q,
                                                multiplier=spectral_radius,
                                                B=K,
                                                p_in=p_in,
                                                p_out=p_out,
                                                phi_distribution=phi_dist,
                                                )
        X = generator.generate()[:,:,0] 
        Gamma0_hat = X.T@X/T 
        Gamma1_hat = X[1:].T@X[:-1]/(T-1) 

        eigvals, eigvecs = np.linalg.eigh(Gamma0_hat)
        idx = np.argsort(np.abs(eigvals))[-K:]  # Indices of K largest magnitude eigenvalues
        U = eigvecs[:, idx]

        A = generator.adjacency_matrix.reshape((N,N))
        true_labels = list(generator.categories.values())
        A_intra = utility_funcs.groupings_to_2D(true_labels) 
        A_inter = A - A_intra 
        num_intra_edges = np.sum(A_intra)
        num_inter_edges = np.sum(A_inter)
        phi = generator.phi_coefficients.reshape((N,N))
        R = utility_funcs.get_R(A) 

        gmm_labels = GaussianMixture(n_components=K, random_state=gmm_random, init_params='k-means++').fit_predict(U)

        ari = adjusted_rand_score(labels_true = true_labels,labels_pred = gmm_labels)
        print(f"Adjusted Rand Index for N={N}: {ari}")
        
        A_hat = utility_funcs.groupings_to_2D(gmm_labels) 

        # --- build masks using TRUE blocks for evaluation ---
        labels = np.asarray(true_labels)
        same_block = labels[:, None] == labels[None, :]
        inter_mask = ~same_block
        np.fill_diagonal(inter_mask, False)

        store_diagonal_variances = np.zeros((num_percents))
        store_biases = np.zeros((num_percents))
        store_percentage_incorrect = np.zeros((num_percents))
        for i, p in enumerate(LASSO_weights): 
            print(p)

            phi_hat = fit_masked_lasso_moment(G0=Gamma0_hat, G1=Gamma1_hat, labels=gmm_labels, lam=p, symmetrize=False,
                                ridge=1e-8, max_iter=1000, tol=1e-6)
        

            Gamma_inter_block = np.where(A_hat==1,0,phi_hat) 

            mask = np.abs(phi_hat) > 0


            # A_hat_inter_block = np.where(mask,1,0)
            A_hat_inter_block = np.where(phi_hat==0,0,1)
            A_hat_inter_block = np.where(A_hat==1,0,A_hat_inter_block)

            A_hat_p = A_hat_inter_block + A_hat 

            percentage_incorrect = np.sum(np.abs(A_hat_inter_block - A_inter))/(N**2 - num_intra_edges)
            print(f"Percentage of incorrect edges in A_hat after soft-thresholding: {percentage_incorrect*100}%")
            store_percentage_incorrect[i] = percentage_incorrect*100

            R_hat = utility_funcs.get_R(A_hat_p)

            unrestricted_variance = np.kron(Gamma0_hat,var*np.identity(N))
            restricted_variance_inv = R_hat.T@unrestricted_variance@R_hat
            restricted_variance = np.linalg.inv(restricted_variance_inv) # shape = (M_hat,M_hat) 
            restricted_variance_N_space_R_hat = R_hat@restricted_variance@R_hat.T 

            R = utility_funcs.get_R(A) 
            restricted_variance_inv_R = R.T@unrestricted_variance@R
            restricted_variance_R = np.linalg.inv(restricted_variance_inv_R) # shape = (M,M) 
            restricted_variance_N_space_R = R@restricted_variance_R@R.T 

            measure = np.trace(restricted_variance_N_space_R_hat)/np.trace(restricted_variance_N_space_R)

            store_diagonal_variances[i] = measure

            C_inf = restricted_variance@R_hat.T@unrestricted_variance@R 
            store_biases[i] = np.linalg.norm(C_inf,ord=2)

        variances[j,:,l] = store_diagonal_variances
        biases[j,:,l] = store_biases
        percentage_incorrect_edges[j,:,l] = store_percentage_incorrect

    print(variances)
    print(biases)

variances = np.mean(variances, axis=-1)
biases = np.mean(biases,axis=-1)
percentage_incorrect_edges = np.mean(percentage_incorrect_edges,axis=-1) 

###### SAVE VARIANCE RATIOS TO CSV FILE ######
np.savetxt('variance_ratios_multipleN.csv', variances, delimiter=',')
np.savetxt('biases_multipleN.csv', biases, delimiter=',')

    
###### PLOT LINE PLOT ###### 
fig = go.Figure()

markers = ["circle", "square", "diamond", "cross", "triangle-up"]  # Define different markers
colors = px.colors.qualitative.Set1  # Use a predefined color set

for j, pout in enumerate(p_out_list):
    fig.add_trace(go.Scatter(
        x=[sp for sp in LASSO_weights], 
        y=variances[j][:], 
        mode='lines+markers',
        name=fr"$p_{{\text{{out}}}} = {pout}$",  # Add pout value to the legend
        marker=dict(symbol=markers[j % len(markers)], size=8),  # Use different markers
        line=dict(color=colors[j % len(colors)])  # Use different colors
    ))

# Set the title and axis labels
fig.update_layout(
    xaxis_title=r'$\lambda$',
    yaxis_title=r'$\alpha_{V}$'
)

layout = go.Layout(
    yaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(showline=True,  linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif",
    font_size=16,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)
fig.update_layout(layout)

fig.write_image(f"ratio_of_variances_multi_pout.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"ratio_of_variances_multi_pout.pdf", format='pdf')

fig = go.Figure()

for j, pout in enumerate(p_out_list):
    fig.add_trace(go.Scatter(
        x=[sp for sp in LASSO_weights], 
        y=biases[j][:], 
        mode='lines+markers',
        name=fr"$p_{{\text{{out}}}} = {pout}$",  # Add pout value to the legend
        marker=dict(symbol=markers[j % len(markers)], size=8),  # Use different markers
        line=dict(color=colors[j % len(colors)])  # Use different colors
    ))

# Set the title and axis labels
fig.update_layout(
    xaxis_title=r'$\lambda$',
    yaxis_title=r'$\lVert C_{\infty} \rVert_{2}$'
)

fig.update_layout(layout)

fig.write_image(f"biases_multi_pout.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"biases_multi_pout.pdf", format='pdf')

fig = go.Figure()

for j, pout in enumerate(p_out_list):
    fig.add_trace(go.Scatter(
        x=LASSO_weights, 
        y=percentage_incorrect_edges[j][:], 
        mode='lines+markers',
        name=fr"$p_{{\text{{out}}}} = {pout}$",  # Add pout value to the legend
        marker=dict(symbol=markers[j % len(markers)], size=8),  # Use different markers
        line=dict(color=colors[j % len(colors)])  # Use different colors
    ))

# Set the title and axis labels
fig.update_layout(
    xaxis_title=r'$\lambda$',
    yaxis_title=f'Variable Selector Error (%)'
)

fig.update_layout(layout)

fig.write_image(f"percent_incorrect_edges.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"percent_incorrect_edges.pdf", format='pdf')
