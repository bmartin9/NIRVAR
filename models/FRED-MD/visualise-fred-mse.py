""" 
Script to visualise the MSE between predicted IP and realised IP.
"""

#!/usr/bin/env python3
# USAGE: ./visualise-fred-mse.py <DESIGN_MATRIX>.csv predictions-1.csv predictions-2.csv ... 

import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from datetime import datetime
import time

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols=range(1,123))
T = Xs.shape[0]
N = Xs.shape[1] 

print(Xs.shape)
IP_targets = Xs[480:,5] 
print(IP_targets[:6])

def read_csv_files(argv):
    arrays_list = []

    # Iterate over the system arguments starting from the second argument
    for file_path in argv[2:]:
        try:
            # Read the CSV file as a DataFrame
            df = pd.read_csv(file_path,header=None)

            # Convert the DataFrame to a NumPy array and add it to the list
            arrays_list.append(df.to_numpy())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return arrays_list

predictions_list = read_csv_files(sys.argv) 

mse_list = [] 
mse_ratio_list = []
for i in range(len(predictions_list)):
    absolute_mse = (IP_targets[:] - predictions_list[i][:,0])**2
    mse_ratio = np.cumsum((IP_targets[:] - predictions_list[0][:,0])**2)/np.cumsum(absolute_mse) #NIRVAR should be the first model in the list
    mse_list.append(absolute_mse)
    print(np.sum(absolute_mse))
    mse_ratio_list.append(mse_ratio) 

compare_mse_metric = [] 
for i in range(1,len(predictions_list)):
    MSE_NIRVAR = mse_list[0]
    MSE_comparison = mse_list[i] 
    delta = (MSE_comparison - MSE_NIRVAR)/MSE_NIRVAR 
    abs_delta = [abs(num) for num in delta]
    metric = np.sign(delta)*np.log(abs_delta)
    compare_mse_metric.append(metric) 

IP_covariate = 5
gt_compare_list = [IP_targets] 
for i in range(len(predictions_list)):
    gt_compare_list.append(predictions_list[i][:,0]) 



dates = pd.date_range(start='2000-01-01', end='2019-12-31', freq='M')

fig = go.Figure()
fig_ratio = go.Figure()
fig_metric = go.Figure() 
fig_gt_pred = go.Figure() 


names = ["NIRVAR","FNETS","GNAR","FARM"]
colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]
dash_styles = ['solid', 'dash', 'dot', 'dashdot','longdash','dashdotdot','longdashdot']
# Add a line for each list in mse_list
for i, mse_values in enumerate(mse_list):
    fig.add_trace(go.Scatter(x=dates, y=mse_values, mode='lines', name=names[i],line=dict(color=colors[i])))
    fig_ratio.add_trace(go.Scatter(x=dates, y=mse_ratio_list[i], mode='lines', name=names[i],line=dict(color=colors[i]))) 
    if i < len(predictions_list)-1 :
        fig_metric.add_trace(go.Scatter(x=dates, y=compare_mse_metric[i], mode='lines', name=names[i+1],line=dict(color=colors[i]))) 

gt_names = names 
gt_names.insert(0,"Ground Truth") 
for i in range(len(predictions_list) + 1):
    fig_gt_pred.add_trace(go.Scatter(x=dates, y=gt_compare_list[i], mode='lines', name=names[i],line=dict(color=colors[i])))

# Update layout
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='MSE',
    xaxis=dict(
        tickformat="%b\n%Y",  # Format x-axis ticks as "Jan\n2000"
    )
)

fig_ratio.update_layout(        
    xaxis_title='Month',
    yaxis_title='Ratio of MSE',
    xaxis=dict(
        tickformat="%b\n%Y",  # Format x-axis ticks as "Jan\n2000"
    )
)   

fig_metric.update_layout(        
    xaxis_title='Month',
    yaxis_title='$sign(\delta)*\log(|\delta|)$',
    xaxis=dict(
        tickformat="%b\n%Y",  # Format x-axis ticks as "Jan\n2000"
    )
)   
fig_gt_pred.update_layout(
    xaxis_title='Month',
    yaxis_title='Industrial Production', 
    xaxis=dict(
        tickformat="%b\n%Y",  # Format x-axis ticks as "Jan\n2000"
    )
)

layout_gt = go.Layout(
    yaxis=dict( showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)
fig_gt_pred.update_layout(layout_gt)

layout_ratio = go.Layout(
    yaxis=dict( showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
)
fig_ratio.update_layout(layout_ratio)



# Save the plot to a file
fig.write_image("plot.eps")
fig_ratio.write_image("plot_ratio.eps")
fig_metric.write_image("plot_metric.eps")
fig_gt_pred.write_image("plot_gt_pred.eps")
time.sleep(1)
fig.write_image("plot.eps")
fig_ratio.write_image("plot_ratio.eps")
fig_metric.write_image("plot_metric.eps")
fig_gt_pred.write_image("plot_gt_pred.eps")
