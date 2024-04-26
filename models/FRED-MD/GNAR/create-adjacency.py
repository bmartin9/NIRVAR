""" 
Create adjacency matrix for use in GNAR model.
"""

#!/usr/bin/env python
# USAGE: python create-adjacency.py FRED-MD-updated_appendix.csv fred-balanced.csv

import numpy as np 
import sys 
import csv
import pandas as pd

# Path to your CSV file
file_path = sys.argv[1] 
encoding = 'latin1'

# Usecols parameter is set to [2, 6] to read only the 3rd and 7th columns (0-indexed)
df = pd.read_csv(file_path, usecols=[2, 6], header=None ,encoding=encoding)

# Create a dictionary: keys from the 3rd column (index 2) and values from the 7th column (index 6)
data_dict = dict(zip(df[2], df[6]))

# for key, value in data_dict.items():
#     print(f"{key}: {value}")

# Path to your CSV file
file_path2 = sys.argv[2] 

# List to store the header
header = []

# Reading the CSV file
with open(file_path2, 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)

    # Read the header
    header = next(csvreader)

grouping_dict = {key: int(data_dict[key]) for key in header if key in data_dict} 
grouping_dict['IPB51222S'] = 1 # missing from FRED-MD-updated_appendix.csv
ordered_grouping_dict = {key: grouping_dict[key] for key in header} 
print(ordered_grouping_dict)

print(len(grouping_dict))
print(len(header))

# Convert your_list to a set for set operations
list_keys_set = set(header)

# Convert the keys of new_dict to a set
new_dict_keys_set = set(grouping_dict.keys())

# Find elements in your_list that are not in the keys of new_dict
elements_not_in_new_dict = list_keys_set - new_dict_keys_set

# Check if there are any elements in your_list that are not in the keys of new_dict and print them
if elements_not_in_new_dict:
    print("Elements in your_list that are not in the keys of new_dict:", elements_not_in_new_dict)
else:
    print("All elements in your_list are in the keys of new_dict.")
# Read the CSV file using pandas
df = pd.read_csv(file_path,header=None ,encoding=encoding)

# Number of rows
group_keys = list(ordered_grouping_dict.keys())
N = len(group_keys) 

# Initialize an NxN adjacency matrix with zeros
adjacency_matrix = np.zeros((N, N))

# Fill the adjacency matrix
for i in range(N):
    for j in range(N):
        if ordered_grouping_dict[group_keys[i]] == ordered_grouping_dict[group_keys[j]]:
            adjacency_matrix[i, j] = 1

adjacency_matrix = adjacency_matrix.astype(int)

is_symmetric = np.array_equal(adjacency_matrix, adjacency_matrix.T)

print("Is the array symmetric?", is_symmetric)


adjacency_matrix_df = pd.DataFrame(adjacency_matrix)

# Output the adjacency matrix to a new CSV file
output_file_path = 'adjacency.csv'
adjacency_matrix_df.to_csv(output_file_path, index=False, header=False)
