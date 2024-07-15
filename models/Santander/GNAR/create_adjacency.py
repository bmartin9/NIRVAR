"""
Script to create adjacency matrix for GNAR using an epsilon ball around each bike station for Santander bikes
"""
#!/usr/bin/env python3
#USAGE: python create_adjacency.py ../../../data/raw/Santander/santander_distances.npy ../../../data/raw/Santander/common_start_ids.csv 

import numpy as np
import pickle
import sys 

# Load the distances array
distances = np.load(sys.argv[1])
common_start_ids = np.loadtxt(sys.argv[2], delimiter=',', dtype=int)

print(distances[:10,:10])

###### PARAMETERS ######
R = 3

# Initialize the adjacency matrix with zeros
adjacency_matrix = np.zeros(distances.shape)

# Set entries to 1 where distance is less than R
adjacency_matrix[distances < R] = 1

sub_adjacency_matrix = adjacency_matrix[np.ix_(common_start_ids, common_start_ids)]

# Format the filename to include the threshold R
filename = f'735GNAR_adjacency_R{R}.csv'

# Save the adjacency matrix as a CSV file
np.savetxt(filename, sub_adjacency_matrix, delimiter=',', fmt='%d')


