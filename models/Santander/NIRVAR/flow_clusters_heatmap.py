""" 
Compute the number of rides between NIRVAR cluster i and NIRVAR cluster j. 
"""

#!/usr/bin/env python3
# USAGE: ./flow_clusters_heatmap.py santander_train.csv common_start_ids labels_hat.csv <BACKTEST_DAY>

import numpy as np
import pandas as pd
import sys

###### READ IN DATA ######
df = pd.read_csv(sys.argv[1])
common_start_ids = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=0).astype(int)
labels_hat = np.genfromtxt(sys.argv[3], delimiter=',', skip_header=0).astype(int)
backtest_day = int(sys.argv[4])

cluster_labels = labels_hat[backtest_day]
num_clusters = np.max(labels_hat) + 1

# Convert subset_ids to a set for faster membership testing
subset_ids_set = set(common_start_ids)

# Create a dictionary for cluster lookup
id_to_cluster = {id_: cluster_labels[i] for i, id_ in enumerate(common_start_ids)}

# Initialize the count matrix
M = np.zeros((num_clusters, num_clusters), dtype=int)

# Filter rows where both start_id and end_id are in subset_ids
filtered_df = df[df['start_id'].isin(subset_ids_set) & df['end_id'].isin(subset_ids_set)]

# Vectorized operation to count the occurrences
start_clusters = filtered_df['start_id'].map(id_to_cluster)
end_clusters = filtered_df['end_id'].map(id_to_cluster)

for start_cluster, end_cluster in zip(start_clusters, end_clusters):
    M[start_cluster, end_cluster] += 1

# Save the result to a CSV file
np.savetxt('flow_clusters.csv', M, delimiter=',', fmt='%d')
