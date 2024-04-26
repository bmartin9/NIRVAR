#!/usr/bin/env python3 
# USAGE: ./utility_funcs.py 

import plotly.graph_objs as go
import plotly.express as px
import numpy as np 
from scipy.stats import norm 
import networkx as nx

# Compute Restriction Matrix, R, from the adjacency matrix 
def get_R(A : np.ndarray):
    """ 
    :param A: The adjacency matrix with 1s on the diagonals. Shape = (N,N)
    :type A: np.ndarray 

    :return R: Shape = (N**2,M)
    :rtype R: np.ndarray 
    """
    N = A.shape[0]
    M = np.sum(A)
    R = np.identity(M)
    alpha = A.flatten('F') 
    zero_rows = []
    counter = 0 
    for i in range(N**2):
        if alpha[i]==0:
            zero_rows.append(counter)
        elif alpha[i]==1:
            counter+=1
    R = np.insert(R,zero_rows,0,axis=0)
    return R

# functions to compute bias factor 
def masked_design(X : np.ndarray, A : np.ndarray) -> np.ndarray:
    """ 
    :param X: Design matrix. Shape = (T-1,N) 
    :type X: np.ndarray 

    :param A: Some adjacency matrix. Shape = (N,N)
    :type A: np.ndarray 

    :return X_mask: Shape = (N,T-1,N)
    :rtype X_mask: np.ndarray
    """
    X_mask = A[:,np.newaxis,:]*X[np.newaxis,:,:]
    return X_mask 

def bias_factor(X : np.ndarray,A : np.ndarray,A_hat : np.ndarray) -> np.ndarray:
    """ 
    :param X: Design matrix. Shape = (T-1,N) 
    :type X: np.ndarray 

    :param A: Some true adjacency matrix. Shape = (N,N)
    :type A: np.ndarray 

    :param A_hat: Some reconstructed adjacency matrix. Shape = (N,N)
    :type A_hat: np.ndarray 

    :return inverse_bias: Shape = (N,N,N)
    :rtype inverse_bias: np.ndarray
    """
    tolerance = 1e-8
    X_A = masked_design(X,A)
    X_A_hat = masked_design(X,A_hat)
    N = A.shape[0] 
    inverse_bias = np.zeros((N,N,N))
    for i in range(N):
        X_A_non_zero_col_indices = np.where(X_A[i].any(axis=0))[0] #only do ols on stocks that are connected to node i
        X_A_hat_non_zero_col_indices = np.where(X_A_hat[i].any(axis=0))[0] #only do ols on stocks that are connected to node i
        X_A_i = X_A[i,:,X_A_non_zero_col_indices].T
        X_A_hat_i = X_A_hat[i,:,X_A_hat_non_zero_col_indices].T
        D = np.linalg.inv(X_A_hat_i.T@X_A_hat_i)@(X_A_hat_i.T@X_A_i)
        close_to_zero = np.isclose(D, 0, atol=tolerance, rtol=tolerance)
        D[close_to_zero] = 0
        # print(D)
        inverse_bias[i][np.ix_(X_A_hat_non_zero_col_indices, X_A_non_zero_col_indices)] = D 
    return inverse_bias

def correct_for_bias(X : np.ndarray,A : np.ndarray,A_hat : np.ndarray, phi_estimate : np.ndarray, phi_true : np.ndarray) -> np.ndarray:
    """ 
    :param X: Design matrix. Shape = (T-1,N) 
    :type X: np.ndarray 

    :param A: Some true adjacency matrix. Shape = (N,N)
    :type A: np.ndarray 

    :param A_hat: Some reconstructed adjacency matrix. Shape = (N,N)
    :type A_hat: np.ndarray 

    :param phi_estimate: Shape = (N,N)
    :type phi_estimate: np.ndarray 

    :param phi_true: Shape = (N,N)
    :type phi_true: np.ndarray 

    :return phi_corrected: Shape = (N,N)
    :rtype phi_corrected: np.ndarray
    """
    D = bias_factor(X,A,A_hat) 
    N = A.shape[0] 
    phi_corrected = np.zeros((N,N)) 
    for i in range(N):
        phi_corrected[i] = phi_estimate[i] + phi_true[i] - D[i]@phi_true[i] 
    return phi_corrected 

# Draw SBM
def drawSBM(sizes : list , p_in : float, p_out :float):
    G = nx.random_partition_graph(sizes=sizes,p_in=p_in,p_out=p_out,directed=False)
    positions = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]] 
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.size = node_adjacencies
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title = "",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text = "",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig 

# Zhu and Ghodsi Function
def zhu(d : np.ndarray):
    """ 
    :param d: An array of eigenvalues (or another measure of importance) sorted in descending order of importance
    :type d: np.ndarray 

    :return profile_likelihood: Array of the profile likelihood values, one for each dimension, q
    :rtype profile_likelihood: np.ndarray 

    :return np.argmax(profile_likelihood): The dimension, q, at which the profile log-likelihood is maximum
    :rtype np.argmax(profile_likelihood): int
    """
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1,p-1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(((q-1) * (np.std(d[:q]) ** 2) + (p-q-1) * (np.std(d[q:]) ** 2)) / (p-2))
        profile_likelihood[q] = norm.logpdf(d[:q],loc=mu1,scale=sd).sum() + norm.logpdf(d[q:],loc=mu2,scale=sd).sum()
    return profile_likelihood[1:p-1], np.argmax(profile_likelihood[1:p-1])+1

def iterate_zhu(d : np.ndarray ,x : int =3): 
    """ 
    Find the dimension, q1, of the 1st largest gap in the scree plot, then the dimension, q2 > q1, of the 
    second largest gap in the scree plot given q1, and so on ... up to dimension x 

    :param d: An array of eigenvalues (or another measure of importance) sorted in descending order of importance
    :type d: np.ndarray 

    :return results: Array of dimensions [q1, q2, ..., qx], where q1 is the dimension of the 1st largest gap in the scree plot, q2 is the dimension of the second largest gap given q1, and so on up to dimension x
    :rtype results: np.ndarray
    
    """
    results = np.zeros(x,dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x-1):
        results[i+1] = results[i] + zhu(d[results[i]:])[1]
    return results


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig