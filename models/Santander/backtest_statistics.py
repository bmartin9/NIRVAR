""" 
Compare the predictions of different models on the Santander dataset 
accross various metrics.
"""

#!/usr/bin/env python3
# USAGE: ./backtest_statistics.py <DESIGN_MATRIX>.csv backtesting_config.yaml predictions-1.csv predictions-2.csv ... 

import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from datetime import datetime
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import yaml 
import argparse


parser = argparse.ArgumentParser(description='Undifference predictions')
parser.add_argument('--differenced_order', help='First or Second Difference', default="SECOND") 
args, unknown = parser.parse_known_args()


with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config['SEED']
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N = Xs.shape[1] 

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

targets = Xs[first_prediction_day+1:first_prediction_day+1+n_backtest_days_tot,: ]
# targets = Xs[first_prediction_day+1:first_prediction_day+1+n_backtest_days_tot,: ] - Xs[first_prediction_day+1 - 1:first_prediction_day+1+n_backtest_days_tot -1,: ]


mspe_list = [] 
mspe_list_normalised = [] 
mspe_sd_list = [] 

for i in range(len(predictions_list)):
    mspe_list.append(np.mean((targets - predictions_list[i])**2)) 
    mspe_sd_list.append(np.std((targets - predictions_list[i])))
    mspe_list_normalised.append(np.mean((targets - predictions_list[i])**2)/(np.mean(targets**2)))

proportion_list = []
station_wide_mse_list = []
# Get proportion of stations where NIRVAR has a lower MSE 
for i in range(len(predictions_list)):
    station_wide_mse = ((targets - predictions_list[i])**2).mean(axis=0)
    station_wide_mse_list.append(station_wide_mse)

for i in range(len(predictions_list)):
    a = np.where(station_wide_mse_list[0]<station_wide_mse_list[i],1,0)
    prop_percentage = 100*np.sum(a)/N
    proportion_list.append(prop_percentage)


summary_statistics = {'MSPE' : mspe_list,
                      'MSPE Normalised' : mspe_list_normalised,
                      'MSPE_SD' : mspe_sd_list,
                      'Proportion of Stations where NIRVAR is superior':proportion_list,
                      }

f = open("summary_statistics.txt", "w")
f.write("{\n")
for k in summary_statistics.keys():
    f.write("'{}':'{}'\n".format(k, summary_statistics[k]))
f.write("}")
f.close() 
