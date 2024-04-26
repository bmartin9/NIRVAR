""" 
Define the GNAR network to be a 10 connected component graph where 
node i is connected to node j if the ith stock is in the same SIC sector as the jth stock. 
This script reads in the SIC codes and the design matrix and outputs the adjacency matrix. 
"""

#!/usr/bin/env python3 
# USAGE: ./SIC-adjacency.py <TRUE_SICCD>.csv labels_hat.csv <BACKTEST_DATA>.csv rows_to_delete.csv 
# e.g. <TRUE_SICCD> = ../../data/processed/20201231.csv , <BACKTEST_DATA>.csv = ../../data/processed/Matrix_Format_SubsetUniverse/pvCLCL_20000103_20201231.csv

import argparse 
import numpy as np 
import csv 
from numpy import genfromtxt
from sklearn.metrics.cluster import adjusted_rand_score
import plotly.io as pio
import pandas as pd
import plotly.express as px
import time
import numpy as np
import plotly.graph_objs as go



parser = argparse.ArgumentParser()
parser.add_argument("file1", help="path to file 1")
parser.add_argument("file2", help="path to file 2")
parser.add_argument("file3", help="path to file 3")
parser.add_argument("file4", help="path to file 4")
args = parser.parse_args()

true_siccd_path = args.file1
labels_hat_path = args.file2 
pvCLCL_path = args.file3
rows_to_delete_path = args.file4

# true_siccd = genfromtxt(true_siccd_path, delimiter=',', skip_header=1)
labels_hat = genfromtxt(labels_hat_path, delimiter=',', skip_header=1)
rows_to_delete = genfromtxt(rows_to_delete_path, delimiter=',', skip_header=0).astype(int)

N = labels_hat.shape[1]
print(f"N : {N}") 

ticker_to_sic_dict = {}
with open(true_siccd_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Assuming the CSV has a header row, you can skip it
    next(csv_reader)  # Skip the header
    
    for row in csv_reader:
        if len(row) >= 3:  # Check if there are at least 5 columns in the row
            key = row[1]  # Column 2 (Python uses 0-based indexing)
            value = row[13]  # Column 5
            ticker_to_sic_dict[key] = value
count = 0
for key, value in ticker_to_sic_dict.items():
    print(f'{key}: {value}')
    count += 1
    if count >= 4:
        break

data_ticker_values = []

# Open the CSV file for reading
with open(pvCLCL_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row
    next(csv_reader)
    
    # Iterate through the remaining rows and save the first column to the list
    for row in csv_reader:
        if not any("na" in item for item in row):
            data_ticker_values.append(row[0]) 

# Create a new dictionary containing only the specified keys
subset_dict = {key: ticker_to_sic_dict[key] for key in data_ticker_values if key in ticker_to_sic_dict} 

# Initialize a list to store keys that are not in the original_dict
key_error_list = []

# Iterate through the keys in the subset_dict
for key in data_ticker_values:
    if key in ticker_to_sic_dict:
        # print(f'{key}: {subset_dict[key]}')
        continue
    else:
        print(f'{key}: Not in ticker_to_sic_dict')
        key_error_list.append(key)

count = 0
for key, value in subset_dict.items():
    print(f'{key}: {value}')
    count += 1
    if count >= 4:
        break

print(f"Key Errors: {key_error_list}")

# Create a new dictionary with values converted to integers
int_subset_dict = {key: int(value) for key, value in subset_dict.items()}

# Create a new dictionary with the first two digits of each value
trimmed_int_subset_dict = {key: int(str(value)[:2]) for key, value in int_subset_dict.items()}

true_labels_dict = {}

for key, value in trimmed_int_subset_dict.items():
    if 1 <= value <= 9:
        true_labels_dict[key] = 0
    elif 10 <= value <= 14:
        true_labels_dict[key] = 1
    elif 15 <= value <= 17:
        true_labels_dict[key] = 2
    elif 20 <= value <= 39:
        true_labels_dict[key] = 3
    elif 40 <= value <= 49:
        true_labels_dict[key] = 4
    elif 50 <= value <= 51:
        true_labels_dict[key] = 5
    elif 52 <= value <= 59:
        true_labels_dict[key] = 6
    elif 60 <= value <= 67:
        true_labels_dict[key] = 7
    elif 70 <= value <= 89:
        true_labels_dict[key] = 8
    elif 91 <= value <= 99:
        true_labels_dict[key] = 9
    else:
        print(f"ERROR in SIC Map: {key}: {value}")

count = 0
for key, value in true_labels_dict.items():
    print(f'{key}: {value}')
    count += 1
    if count >= 4:
        break

# true_labels_array = np.array(list(true_labels_dict.values()))

def remove_items_by_indices(dictionary, indices_to_remove):
    # Create a new dictionary to store the filtered key-value pairs
    filtered_dict = {}
    
    # Convert the indices to a set for faster lookup
    indices_set = set(indices_to_remove)
    
    # Iterate through the dictionary and copy key-value pairs to the new dictionary
    for index, (key, value) in enumerate(dictionary.items()):
        if index not in indices_set:
            filtered_dict[key] = value

    return filtered_dict

true_labels_filtered = remove_items_by_indices(true_labels_dict, rows_to_delete) 
del true_labels_filtered['SPY']
true_labels_array = np.array(list(true_labels_filtered.values()))

print(f"True Labels Array: {true_labels_array.shape}") 

def groupings_to_2D(input_array : np.ndarray) -> np.ndarray:
        """ 
        Turn a 1d array of integers (groupings) into a 2d binary array, A, where 
        A[i,j] = 1 iff i and j have the same integer value in the 1d groupings array.

        Parameters
        ----------
        input_array : np.ndarray
            1d array of integers.

        Returns
        -------
        A : np.ndarray
            2d Representation. Shape = (len(input_array),len(input_array))
        """
        L = len(input_array)
        A = np.zeros((L,L)) 
        for i in range(L):
            for j in range(L): 
                if input_array[i] == input_array[j]:
                    A[i][j] = 1 
                else:
                    continue 
        
        return A 

adj = groupings_to_2D(true_labels_array) 
print(np.array_equal(adj,adj.T))  # Check if adj is symmetric

# Save adj to a CSV file
np.savetxt('adjacency.csv', adj, delimiter=',', fmt='%d')
