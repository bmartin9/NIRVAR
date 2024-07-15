""" 
Script to plot the percentage turnover of your portfolio on a given day over time.
"""

#!/usr/bin/env python3 
# USAGE: ./0.3-turnover.py predictions1.csv predictions2.csv ...

import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import plotly.io as pio
import time 

N = 648

parser = argparse.ArgumentParser() 
parser.add_argument("files", nargs="+")

opts = parser.parse_args() 
files_list = opts.files

num_models = len(files_list)
color = cm.rainbow(np.linspace(0, 1, num_models))
models_list = ['model-{0}'.format(i) for i in range(num_models)]
colors_dictionary = dict(zip(models_list, color))

predictions_list  = [] 

for f in range(num_models):
    predictions_list.append(genfromtxt(files_list[f],delimiter=',',skip_header=0))

num_backtest_days = predictions_list[0].shape[0]
num_turnover_days = num_backtest_days - 1 

date_range = pd.date_range(start='2004-01-02', end='2020-12-31', periods=num_turnover_days)

def transaction_indicator(predictions : np.ndarray, yesterdays_predictions : np.ndarray):
    """ 
    Parameters
    ----------
    predictions : np.ndarray
        shape = (N)

    yesterdays_predictions : np.ndarray
        shape = (N)

    Returns
    -------
    transaction_indicator : np.ndarray
        1 if a transaction occured, 0 otherwise. Shape = (N)
    """
    transaction_indicator = np.where(np.sign(predictions)-np.sign(yesterdays_predictions)==0,0,1)
    return transaction_indicator 

def turnover_percentages(prediction_array : np.ndarray):
    """ 
    Parameters
    ----------
    prediction_array : np.ndarray 
        shape = (T,N)

    Returns
    -------
    turnover_percentages : np.ndarray
        shape = (T-1)

    """
    T = prediction_array.shape[0] 
    N = prediction_array.shape[1]
    turnover_percentages = np.zeros((T-1))
    for t in range(T-1):
        yesterdays_pred = prediction_array[t,:]
        todays_pred = prediction_array[t+1,:]
        trans_ind = transaction_indicator(todays_pred,yesterdays_pred) 
        todays_turnover = np.sum(trans_ind)/N
        turnover_percentages[t] = todays_turnover 

    return turnover_percentages 

turnover_list = []
for m in range(num_models):
    turnover_list.append(turnover_percentages(predictions_list[m])) 

# Create traces for N lines
traces = []
names = ["NIRVAR","FARM","GNAR"]
for i in range(num_models):
    trace = go.Scatter(x=date_range, y=turnover_list[i], mode='lines', name=names[i])
    traces.append(trace)

# Create a layout
layout = go.Layout(
    yaxis=dict(title='Percentage Turnover')
)

# Create a figure with the traces and layout
fig = go.Figure(data=traces, layout=layout)

fig.update_yaxes(tickformat=".0%")

# Save the figure as a PNG file
pio.write_image(fig, 'turnover.pdf')
time.sleep(2)
pio.write_image(fig, 'turnover.pdf') 
