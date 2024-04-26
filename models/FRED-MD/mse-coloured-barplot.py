""" 
Script to calculate the overall MSE between predicted IP and realised IP for each model 
and plot a bar plot of the MSE coloured by whether the magnitude of the predicted IP 
is greater than the magnitude of the realised IP
"""

#!/usr/bin/env python3
# USAGE: ./mse-coloured-barplot.py <DESIGN_MATRIX>.csv predictions-1.csv predictions-2.csv ... 

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

IP_targets = Xs[480:,5] 

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

overall_mse_array = np.zeros((len(predictions_list),2))
for i in range(len(predictions_list)):
    red_observations = np.where(np.abs(predictions_list[i][:,0]) >= np.abs(IP_targets[:]), 1, 0)
    green_observations = np.where(np.abs(predictions_list[i][:,0]) < np.abs(IP_targets[:]), 1, 0)
    red_mse = np.dot((IP_targets[:] - predictions_list[i][:,0])**2,red_observations)
    green_mse = np.dot((IP_targets[:] - predictions_list[i][:,0])**2,green_observations)
    overall_mse_array[i,0] = green_mse
    overall_mse_array[i,1] = red_mse

data = { 
    "green":overall_mse_array[:,0],
    "red":overall_mse_array[:,1],
    "labels": ["NIRVAR","FNETS","GNAR","FARM"]
}

fig = go.Figure(
    data=[
        go.Bar(
            name="Under-Estimated",
            x=data["labels"],
            y=data["green"],
            offsetgroup=1,
            marker_color='rgb(55, 126, 184)'
        ),
        go.Bar(
            name="Over-Estimated",
            x=data["labels"],
            y=data["red"],
            offsetgroup=1,
            base=data["green"],
            marker_color='rgb(228, 26, 28)'
        )
    ],
    layout=go.Layout(
        yaxis_title="MSE",
        xaxis_title="Model"
    )
)

print(overall_mse_array)

layout = go.Layout(
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
fig.update_layout(layout)

fig.write_image("red_green_barplot.eps")
time.sleep(1)
fig.write_image("red_green_barplot.eps")

###### SPLIT realised IP Production into top 20 percent Extreme Values ###### 
extreme_mse_array = np.zeros((len(predictions_list),2))
quantile_80 = np.percentile(np.abs(IP_targets[:]), 90)
for i in range(len(predictions_list)):
    extreme_points = np.where(np.abs(IP_targets[:]) >= quantile_80,1,0)
    non_extreme_points = np.where(np.abs(IP_targets[:]) < quantile_80,1,0)
    extreme_mse = np.dot((IP_targets[:] - predictions_list[i][:,0])**2,extreme_points)
    non_extreme_mse = np.dot((IP_targets[:] - predictions_list[i][:,0])**2,non_extreme_points)
    extreme_mse_array[i,0] = non_extreme_mse
    extreme_mse_array[i,1] = extreme_mse

print(extreme_mse_array)

data = { 
    "green":extreme_mse_array[:,0],
    "red":extreme_mse_array[:,1],
    "labels": ["NIRVAR","FNETS","GNAR","FARM"]
}

fig = go.Figure(
    data=[
        go.Bar(
            name="Below 90th Quantile",
            x=data["labels"],
            y=data["green"],
            offsetgroup=1,
            marker_color='rgb(55, 126, 184)'
        ),
        go.Bar(
            name="Above 90th Quantile",
            x=data["labels"],
            y=data["red"],
            offsetgroup=1,
            base=data["green"],
            marker_color='rgb(228, 26, 28)'
        )
    ],
    layout=go.Layout(
        yaxis_title="MSE",
        xaxis_title="Model"
    )
)

layout = go.Layout(
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
fig.update_layout(layout)

fig.write_image("extreme_points_barplot.eps")
time.sleep(1)
fig.write_image("extreme_points_barplot.eps")
