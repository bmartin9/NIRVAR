""" 
Simulation study comparing the latent embeddings across different features.
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py hyperparameters.yaml 

import sys 
import yaml 
from src.models import generativeVAR 
from numpy.random import default_rng
import numpy as np
import plotly.graph_objects as go
import time
import itertools


with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
T = config['T']
N = config['N1']
K = config['K'] 
Q = config['Q'] 
block_sizes =config['block_sizes']
VAR_spectral_radius = config['VAR_spectral_radius']
innovations_variance = config['innovations_variance']

random_state = default_rng(seed=SEED)

###### DEFINE BLOCK MATRICES ######
B_1 = np.array([[0.8,0.8,0.05,0.05],
                [0.8,0.8,0.05,0.05],
                [0.05,0.05,0.8,0.8],
                [0.05,0.05,0.8,0.8]])

B_2 = np.array([[0.8,0.8,0.05,0.05],
                [0.8,0.8,0.05,0.05],
                [0.8,0.05,0.05,0.01],
                [0,0,0.01,0.05]])

B_3 = np.array([[0.8,0.8,0.05,0.05],
                [0.8,0.8,0.05,0.05],
                [0.05,0.05,0.8,0.8],
                [0.05,0.05,0.8,0.8]])

B_4 = np.array([[0.8,0.8,0.05,0.05],
                [0.8,0.8,0.05,0.05],
                [0.8,0.05,0.05,0.01],
                [0,0,0.01,0.05]])



###### UTILITY FUNCTIONS ###### 
def block_indicator_matrix(block_sizes):
    """
    Create an N x K matrix Z, where each row i belongs to exactly one block j.
    For row i in block j, Z[i, j] = 1. Otherwise, Z[i, j] = 0.

    Parameters:
    -----------
    block_sizes : 1D array-like of length K
        Sizes of each block.

    Returns:
    --------
    Z : 2D numpy array of shape (N, K)
        The block indicator matrix.
    """
    # Total number of rows N is the sum of block_sizes
    N = np.sum(block_sizes)
    # Number of blocks K is the length of block_sizes
    K = len(block_sizes)

    # Initialize Z with zeros
    Z = np.zeros((N, K), dtype=int)
    
    start = 0
    # Fill each column of Z by marking the rows that belong to each block
    for j, size in enumerate(block_sizes):
        end = start + size
        Z[start:end, j] = 1
        start = end

    return Z

def scale_M(M : np.ndarray, rho : float) -> np.ndarray:
    M_eigs = np.linalg.eig(M)[0]
    M_scaled = (rho/abs(np.max(M_eigs)))*M
    return M_scaled

###### SAMPLE FROM SBM ###### 
Z = block_indicator_matrix(block_sizes=block_sizes) 

P_1 = Z@B_1@Z.T
P_2 = Z@B_2@Z.T 
P_3 = Z@B_3@Z.T 
P_4 = Z@B_4@Z.T 

A_11 = random_state.binomial(1,P_1)
A_12 = random_state.binomial(1,P_3)
A_21 = random_state.binomial(1,P_4) 
A_22 = random_state.binomial(1,P_2)

np.fill_diagonal(A_11, 1)
np.fill_diagonal(A_12, 1)
np.fill_diagonal(A_21, 1)
np.fill_diagonal(A_22, 1)

A = np.block([[A_11,A_12],
                [A_21,A_22]])

# A = np.block([[A_11,np.zeros((N,N))],
#                 [np.zeros((N,N)),A_22]])

Phi = scale_M(A,VAR_spectral_radius) 

print(Phi) 


###### SIMULATE TIME SERIES ###### 
X_stored = np.zeros((T,N,Q))
X = np.zeros((2*N))
for t in range(T): 
    Z_innovations = random_state.normal(0,np.sqrt(innovations_variance),size=(2*N)) 
    X = Phi@X + Z_innovations
    X_stored[t,:,0] = X[:N]
    X_stored[t,:,1] = X[N:]

# Visualise generated time series 
components_to_plot = [7,8]
timelength_to_plot = 100
fig = go.Figure()
for i in components_to_plot:
    fig.add_trace(go.Scatter(x=np.arange(timelength_to_plot), y=X_stored[:timelength_to_plot,i,0], mode='lines', name=f'Component {i+1}'))

fig.update_layout(
                  xaxis_title='T',
                  yaxis_title=r'$X_{T}$')   

# fig.show()

###### SPECTRAL EMBEDDING OF COVARIANCE MATRICES ######
X1 = X_stored[:,:,0]
X2 = X_stored[:,:,1]
S11 = X1.T@X1/T 
S22 = X2.T@X2/T  
S12 = X1.T@X2/T 
S21 = X2.T@X1/T  

S = np.hstack([S11,S22])
# S = np.block([[S11,S12],
#               [S21,S22]])
# S = S11
print(S.shape)

U, D, Vt = np.linalg.svd(S, full_matrices=False) 

# Truncate the SVD components.
U_K = U[:, :K]     
D_K = np.diag(D[:K])        
V_K = Vt.T[:, :K]  
print(V_K.shape)
print(D_K.shape)

embeddings = V_K@np.sqrt(D_K) 

embeddings1 = embeddings[:]
embeddings2 = embeddings[N:] 

# Define a base layout (will be updated for each combination)
base_layout = dict(
    paper_bgcolor='white',
    plot_bgcolor='white',
    font_family="Serif",
    font_size=11,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)

# Define an axis style that will be used in each updated layout
axis_style = dict(
    showline=True,
    linewidth=1,
    linecolor='black',
    ticks='outside',
    mirror=True,
    automargin=True,
)

# -------------------------------------------------------------
# 1) Determine each node's block membership from Z
#    (argmax over the row finds the column index where Z[i,:] = 1)
# -------------------------------------------------------------
block_membership = np.argmax(Z, axis=1)

# -------------------------------------------------------------
# 2) & 3) Plot embeddings1 (the first two dimensions) in one figure,
#         and embeddings2 (the first two dimensions) in another figure.
#         Color by block membership.
# -------------------------------------------------------------

for (i, j) in itertools.combinations(range(K), 2):
    # -------------------------
    # Plot for embeddings1
    # -------------------------
    fig1_data = []
    for block_id in range(K):
        # Get indices for current block
        idx = np.where(block_membership == block_id)[0]
        # for node in range(10):
        #     idx = np.append(idx,[idx[node]+30])
        fig1_data.append(
            go.Scatter(
                x=embeddings1[idx, i],  # embedding dimension i
                y=embeddings1[idx, j],  # embedding dimension j
                mode='markers',
                name=f'Community {block_id + 1}'
            )
        )
    
    # Update the layout for this combination
    layout1 = go.Layout(
        xaxis=dict(title=f'Embedding dimension {i+1}', **axis_style),
        yaxis=dict(title=f'Embedding dimension {j+1}', **axis_style),
        **base_layout
    )
    
    fig1 = go.Figure(data=fig1_data, layout=layout1)
    
    # Save the figure with a file name indicating the dimensions plotted
    file_name1 = f"feature1_embedding_dim{i}_{j}.pdf"
    fig1.write_image(file_name1)
    time.sleep(1)
    fig1.write_image(file_name1)

    # -------------------------
    # Plot for embeddings2
    # -------------------------
    fig2_data = []
    for block_id in range(K):
        idx = np.where(block_membership == block_id)[0]
        fig2_data.append(
            go.Scatter(
                x=embeddings2[idx, i],
                y=embeddings2[idx, j],
                mode='markers',
                name=f'Community {block_id+1}'
            )
        )
    
    layout2 = go.Layout(
        xaxis=dict(title=f'Embedding dimension {i+1}', **axis_style),
        yaxis=dict(title=f'Embedding dimension {j+1}', **axis_style),
        **base_layout
    )
    
    fig2 = go.Figure(data=fig2_data, layout=layout2)
    
    file_name2 = f"feature2_embedding_dim{i}_{j}.pdf"
    fig2.write_image(file_name2)
    time.sleep(1)
    fig2.write_image(file_name2)





