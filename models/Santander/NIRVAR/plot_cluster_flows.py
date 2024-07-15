""" 
Script to plot a heatmap of the flow of bikes between clusters. 
Also plot a normalised flow heatmap.
"""

#!/usr/bin/env python3
# USAGE: ./plot_cluster_flows.py flow_clusters.csv 

import numpy as np
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go
import time

###### READ IN DATA ######
M = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=0)

# Normalise the flow matrix
M_normalised = np.zeros_like(M, dtype=float) 
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        M_normalised[i,j] = (M[i,j] - M[j,i])/(M[i,j] + M[j,i]) 

# Plot the flow matrix
fig = go.Figure(data=go.Heatmap(
    z=M,
    x=list(range(1, M.shape[1] + 1)),
    y=list(range(1, M.shape[1] + 1)),
    colorscale='Viridis'
))

fig.update_layout(
                    xaxis_title='NIRVAR Cluster',
                    yaxis_title='NIRVAR Cluster')

layout = go.Layout(
    yaxis=dict( showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,autorange='reversed'),
    xaxis=dict(showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)
fig.update_layout(layout)


fig.write_image(f"flow_cluster.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"flow_cluster.pdf", format='pdf')


# Plot the normalised flow matrix

# Define the custom colorscale
colorscale = [
    [0.0, 'red'],    # -1 corresponds to red
    [0.5, 'white'],  #  0 corresponds to white
    [1.0, 'blue']    #  1 corresponds to blue
]
fig = go.Figure(data=go.Heatmap(
    z=M_normalised,
    x=list(range(1, M.shape[1] + 1)),
    y=list(range(1, M.shape[1] + 1)),
    colorscale=colorscale,
    zmin=-np.max(np.abs(M_normalised)),
    zmax=np.max(np.abs(M_normalised))
))

fig.update_layout(
                    xaxis_title='NIRVAR Cluster',
                    yaxis_title='NIRVAR Cluster')

layout = go.Layout(
    yaxis=dict( showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,autorange='reversed'),
    xaxis=dict(showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)
fig.update_layout(layout)


fig.write_image(f"flow_cluster_normalised.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"flow_cluster_normalised.pdf", format='pdf')

