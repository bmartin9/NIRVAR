""" 
Script to calculate the overall MSE between predicted IP and realised IP for each model
"""

#!/usr/bin/env python3
# USAGE: ./overall-mse.py <DESIGN_MATRIX>.csv predictions-1.csv predictions-2.csv ... 

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

overall_mse_list = [] 
for i in range(len(predictions_list)):
    absolute_mse = np.sum((IP_targets[:] - predictions_list[i][:,0])**2)
    overall_mse_list.append(absolute_mse)

print(overall_mse_list[0])

print(f"overall mse list: {overall_mse_list}") 

# Get the proportion of times where the predictions of NIRVAR are better than FARM/FNETS/GNAR 
proportion_list = [] 
mse_list = [] 
for i in range(len(predictions_list)):
    absolute_mse = (IP_targets[:] - predictions_list[i][:,0])**2 
    mse_list.append(absolute_mse)

def calculate_proportion(listA, listB):
    if len(listA) != len(listB):
        raise ValueError("Both lists must have the same length")

    count = sum(a < b for a, b in zip(listA, listB))
    return count / len(listA)

for i in range(1,len(predictions_list)):
    proportion_list.append(calculate_proportion(mse_list[0],mse_list[i]))

print(proportion_list)
