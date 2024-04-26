""" 
Produce plots of how the embedding dimension changes over time as we backtest.
"""

#!/usr/bin/env python3 
# USAGE: ./0.3-embedding-dim.py <DESIGN_MATRIX>.csv backtesting_config.yaml 

import sys
import yaml
import numpy as np
from src.models import train_model
from numpy.random import default_rng
import fcntl
import os
import pandas as pd 
import plotly.graph_objects as go
import plotly.io as pio
import time


with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config['SEED']
Q = config['Q']
# n_backtest_days_tot = config['n_backtest_days']
n_backtest_days_tot = 30
first_prediction_day = config['first_prediction_day']
target_feature = config['target_feature']
SVD_niter = config['SVD_niter']
SVD_random_state = config['SVD_random_state']
quantile = config['quantile'] #The top quantile stocks with the strongest predictions 
target_feature = config['target_feature']
embedding_method = config['embedding_method']

random_state = default_rng(seed=SEED)

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N)

Xs = np.reshape(Xs,(T,N,Q),order='F')



###### BACKTESTING ###### 
d_array1 = np.zeros((n_backtest_days_tot)) # estimated d each day
d_array2 = np.zeros((n_backtest_days_tot)) # estimated d each day for pearson correlation

for index in range(n_backtest_days_tot):
    t = index 
    todays_date = first_prediction_day+t 
    print(todays_date)
    #get current embedding
    furthest_lookback_day = todays_date - 1008
    X_train = Xs[furthest_lookback_day:todays_date,:,:] #Shape = (todays_date,N,Q) 
    current_embedding = train_model.Embedding(X_train,embedding_method=embedding_method) #If d is not specified, then it is computed via marchenko-pastur method
    d_hat = current_embedding.d
    # print(f"d_hat: {d_hat}")
    d_array1[index] = d_hat 

    current_embedding2 = train_model.Embedding(X_train,embedding_method="Pearson Correlation") #If d is not specified, then it is computed via marchenko-pastur method
    d_hat2 = current_embedding2.d 
    d_array2[index] = d_hat2
    print ("\033[A                             \033[A") 



###### PLOTS ######
date_range = pd.date_range(start='2001-01-01', end='2020-12-31', periods=n_backtest_days_tot) 

# Create traces for N lines
traces = []
names = ["Normalised Precision", "Pearson Correlation"] 
trace1 = go.Scatter(x=date_range, y=d_array1, mode='lines', name=names[0])
trace2 = go.Scatter(x=date_range, y=d_array2, mode='lines', name=names[1])
traces.append(trace1)
traces.append(trace2)

# Create a layout
layout = go.Layout(
    yaxis=dict(title='$\hat{d}$') 
)

# Create a figure with the traces and layout
fig = go.Figure(data=traces, layout=layout)

# Save the figure as a PNG file
pio.write_image(fig, 'd_hat.pdf')
time.sleep(2)
pio.write_image(fig, 'd_hat.pdf')
