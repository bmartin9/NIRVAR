# Plot the predicted number of daily rides against the ground truth 

#!/usr/bin/env python3
# USAGE: ./visualise_gt.py <DESIGN_MATRIX>.csv backtesting_config.yaml predictions-1.csv predictions-2.csv ... 

import sys
import numpy as np 
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time
import yaml 
import argparse


parser = argparse.ArgumentParser(description='Undifference predictions')
parser.add_argument('--differenced_order', help='First or Second Difference', default="SECOND") 
args, unknown = parser.parse_known_args()


with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config['SEED']
Q = config['Q']
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
target_feature = config['target_feature']
SVD_niter = config['SVD_niter']
SVD_random_state = config['SVD_random_state']
quantile = config['quantile'] #The top quantile stocks with the strongest predictions 
embedding_method = config['embedding_method']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=0)
T = Xs.shape[0]
N = Xs.shape[1] 

# targets = Xs[first_prediction_day+1:first_prediction_day+1+n_backtest_days_tot,: ]
targets = Xs[first_prediction_day+1:first_prediction_day+1+n_backtest_days_tot,: ] - Xs[first_prediction_day+1 - 1:first_prediction_day+1+n_backtest_days_tot -1,: ]


def read_csv_files(argv):
    arrays_list = []

    # Iterate over the system arguments starting from the second argument
    for file_path in argv[3:]:
        try:
            # Read the CSV file as a DataFrame
            df = pd.read_csv(file_path,header=None)

            # Convert the DataFrame to a NumPy array and add it to the list
            arrays_list.append(df.to_numpy())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return arrays_list

predictions_list = read_csv_files(sys.argv) 

hospital_i = 33
gt_compare_list = [targets[:,hospital_i]] 
for i in range(len(predictions_list)):
    gt_compare_list.append(predictions_list[i][:,hospital_i]) 


dates = pd.date_range(start='2020-02-09', end='2020-03-10', freq='D')

fig_gt_pred = go.Figure() 
names = ["Ground truth","NIRVAR" ,"GNAR", "FARM","FNETS"] 
colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)',    # Dark Cyan
    '#e377c2',             # Plotly Pink
    'rgb(255, 187, 120)',  # Light Orange
    'rgb(128, 177, 211)',  # Light Blue
    'rgb(255, 152, 150)',  # Light Red
]
for i in range(len(predictions_list) + 1):
    fig_gt_pred.add_trace(go.Scatter(x=dates, y=gt_compare_list[i], mode='lines', name=names[i],line=dict(color=colors[i])))

fig_gt_pred.update_layout(
    xaxis_title='Day',
    yaxis_title='Log Number of Rides', 
    xaxis=dict(
        tickformat="%b\n%Y",  # Format x-axis ticks as "Jan\n2000"
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
fig_gt_pred.update_layout(layout)


fig_gt_pred.write_image("plot_gt_pred.pdf")
time.sleep(1)
fig_gt_pred.write_image("plot_gt_pred.pdf")
