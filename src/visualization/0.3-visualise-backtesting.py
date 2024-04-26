""" 
Script that takes as input $x \in \mathbb{N}$ csv files containing daily PnL values and outputs plots of 
    * Cumulative PnL accross time
    * Sharpe Ratio Histograms
    * PnL per trade histograms
    * Annualized average return histograms 
"""

#!/usr/bin/env python3 
# USAGE: ./0.3-visualise-backtesting.py PnL-1.csv PnL-2.csv ... PnL-x.csv 

import argparse
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import plotly.graph_objects as go
import pandas as pd
import datetime
import plotly.io as pio
import time 
import tikzplotly 

NUM_STOCKS = 648

parser = argparse.ArgumentParser() 
parser.add_argument("files", nargs="+")

opts = parser.parse_args() 
files_list = opts.files

num_models = len(files_list)
color = cm.rainbow(np.linspace(0, 1, num_models))
models_list = ['model-{0}'.format(i) for i in range(num_models)]
colors_dictionary = dict(zip(models_list, color))

PnL_list  = [] 

for f in range(num_models):
    PnL_list.append(genfromtxt(files_list[f],delimiter=',',skip_header=0))

num_days = PnL_list[0].shape[0]
horizontal_vals = np.arange(num_days)


date_range = pd.date_range(start='2004-01-01', end='2020-12-31', periods=num_days)
date_range = pd.date_range(start='2004-01-01', end='2020-12-31', periods=num_days)
cum_PnL_bpts = []
for m in range(num_models):
    # cum_PnL_bpts.append([1000*np.cumsum(PnL_list[m])[t]/NUM_STOCKS*t for t in range(num_days)]) 
    cum_PnL_bpts.append(10000*np.cumsum(PnL_list[m])/(num_days)) 


# Create traces for N lines
traces = []
# names = ["NIRVAR C1", "NIRVAR C2","NIRVAR P1","NIRVAR P2","FNETS","GNAR","FARM"]
names = ["NIRVAR","FNETS","GNAR","FARM"]
# names = ["0 bpts","1 bpts","2 bpts","3 bpts","4 bpts",]
# dash_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
dash_styles = ['solid', 'dot', 'dash', 'longdash']
# marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x','star', 'triangle-up']
marker_symbols = ['circle', 'square', 'diamond', 'cross']
# colors = [
#     'rgb(55, 126, 184)',   # Plotly Blue
#     'rgb(228, 26, 28)',    # Plotly Red
#     'rgb(77, 175, 74)',    # Plotly Green
#     'rgb(152, 78, 163)',   # Plotly Purple
#     'rgb(255, 127, 0)',    # Plotly Orange
#     'rgb(0, 139, 139)' ,     # Dark Cyan
#     '#e377c2',  # Plotly Pink
# ]
colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
]

# custom_dashes = [
#     '10px,2px',   # Dash pattern for NIRVAR C1
#     '5px,5px',    # Dash pattern for NIRVAR C2
#     '10px,5px',   # Dash pattern for NIRVAR P1
#     '15px,3px',   # Dash pattern for NIRVAR P2
#     '5px,5px',    # Dash pattern for FNETS
#     '10px,2px',   # Dash pattern for GNAR
#     '8px,8px'     # Dash pattern for FARM
# ]
custom_dashes = [
    '10px,2px',   # Dash pattern for NIRVAR C1
    '5px,5px',    # Dash pattern for NIRVAR C2
    '10px,5px',   # Dash pattern for NIRVAR P1
    '15px,3px',   # Dash pattern for NIRVAR P2
]


# for i in range(num_models):
#     trace = go.Scatter(x=date_range, y=cum_PnL_bpts[i],  mode='lines+markers', name=names[i],line=dict(color=colors[i]),marker=dict(symbol=marker_symbols[i],size=8,opacity=0),text=['']*len(date_range))
#     trace = go.Scatter(x=date_range[::500], y=cum_PnL_bpts[i][::500],  mode='markers', name=names[i],line=dict(color=colors[i]),marker=dict(symbol=marker_symbols[i],size=8), text=['']*len(date_range[::500]))
#     traces.append(trace)

for i in range(num_models):
    # Create a trace with lines and invisible markers
    trace_lines = go.Scatter(
        x=[d.strftime('%Y-%m-%d') for d in pd.to_datetime(date_range).date],
        y=cum_PnL_bpts[i],
        mode='lines',
        name=names[i],
        line=dict(color=colors[i],width = 1),
        text=['']*len(date_range),
        showlegend = False
    )
    traces.append(trace_lines)

    # Create a trace with markers every 500th point
    trace_markers = go.Scatter(
        x=[d.strftime('%Y-%m-%d') for d in pd.to_datetime(date_range[::500]).date],
        y=cum_PnL_bpts[i][::500],
        mode='markers',
        name=names[i],
        line=dict(color=colors[i]),
        marker=dict(symbol=marker_symbols[i], size=6),
        text=['']*len(date_range[::500])
    )
    traces.append(trace_markers)

white_space_duration = datetime.timedelta(days=50)
your_x_min = min(date_range) - white_space_duration
your_x_max = max(date_range) + white_space_duration
# Create a layout
layout = go.Layout(
    yaxis=dict(title='Cumulative PnL (bpts)', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title='Day',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True,range=[your_x_min, your_x_max]),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)


# Create a figure with the traces and layout
fig = go.Figure(data=traces, layout=layout)

# Save the figure as a PNG file
pio.write_image(fig, 'CumPnL-vtss.eps')
time.sleep(2)
pio.write_image(fig, 'CumPnL-vtss.eps')
# tikzplotly.save("CumPnL.tex",fig)


