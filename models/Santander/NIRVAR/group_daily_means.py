""" 
Script to plot the average number of Santander bike rides on Monday, Tuesday,... for each NIRVAR cluster
BACKTEST_DAY is the day on which you look at the clusters. It is an element of {0,...,29}.
"""

#!/usr/bin/env python3
# USAGE: ./cluster_histogram.py <DESIGN_MATRIX>.csv labels_hat.csv  <BACKTEST_DAY>

import numpy as np 
import pandas as pd 
import csv
import sys 
import plotly.express as px
import plotly.graph_objects as go
import time

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N = Xs.shape[1]

labels_hat = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=0).astype(int)

backtest_day = int(sys.argv[3]) 

labels_day = labels_hat[backtest_day]

num_clusters = np.max(labels_hat) + 1 

mean_array = np.zeros((num_clusters, 7)) 

for cluster_number in range(num_clusters):
    cluster_indices = np.where(labels_day == cluster_number)[0]

    X_subset = Xs[:,cluster_indices] 
    T_lookback = X_subset.shape[0] 
    N_cluster = X_subset.shape[1]

    # Create an array representing days of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    days_of_week = (np.arange(T_lookback)+2) % 7 

    # Calculate the mean value for each day of the week
    for day in range(7):
        mean_array[cluster_number, day] = np.mean(X_subset[days_of_week == day])  


# Plot line plot of mean array
fig = go.Figure()

# Define marker symbols
marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'star', 'triangle-up']

# Iterate over each cluster
for cluster_number in range(num_clusters):
    # Get the mean values for the cluster
    cluster_mean = mean_array[cluster_number]

    # Create an array representing days of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    cluster_num_plus1 = cluster_number + 1

    # Add a line trace for the cluster with marker
    fig.add_trace(go.Scatter(
        x=days_of_week, 
        y=cluster_mean, 
        mode='lines+markers', 
        name=f'Cluster {cluster_num_plus1}',
        marker=dict(symbol=marker_symbols[cluster_number % len(marker_symbols)], size=8)  # Use a different marker for each cluster
    ))

# Set the title and axis labels
fig.update_layout(
                    xaxis_title='Day of the Week',
                    yaxis_title='Average Number of Bike Rides')

layout = go.Layout(
    yaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width=500, 
    height=350
)
fig.update_layout(layout)

fig.write_image(f"group_mean_line.pdf", format='pdf')
time.sleep(1)
fig.write_image(f"group_mean_line.pdf", format='pdf')
