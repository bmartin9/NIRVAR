#!/usr/bin/env python3
# USAGE: ./visualise_cluster.py commom_start_ids.csv santander_dictionary.pkl santander_locations.csv labels_hat.csv London.png

import numpy as np 
import pandas as pd 
import csv
import sys 
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load data
common_start_ids = np.loadtxt(sys.argv[1], delimiter=',')

with open(sys.argv[2], 'rb') as file:
    try:
        while True:
            # Load each object
            santander_dict = pickle.load(file)
    except EOFError:
        # End of file reached
        pass

santander_locations_df = pd.read_csv(sys.argv[3])
labels_hat = np.genfromtxt(sys.argv[4], delimiter=',', skip_header=0)[29].astype(int)

# Extract station names corresponding to common_start_ids
station_names = [santander_dict[int(id_)] for id_ in common_start_ids]

# Create a DataFrame with common_start_ids, station names, and labels
common_start_df = pd.DataFrame({
    'StationID': common_start_ids,
    'StationName': station_names,
    'ClusterLabel': labels_hat
})

# Merge with santander_locations_df to get latitude and longitude
merged_df = pd.merge(common_start_df, santander_locations_df, on='StationName')

# Define a list of distinct colors for the clusters
distinct_colors = [
    (55/255, 126/255, 184/255),   # Plotly Blue
    (228/255, 26/255, 28/255),    # Plotly Red
    (77/255, 175/255, 74/255),    # Plotly Green
    (152/255, 78/255, 163/255),   # Plotly Purple
    (255/255, 127/255, 0/255),    # Plotly Orange
    (0/255, 139/255, 139/255),    # Dark Cyan
    '#e377c2'                     # Plotly Pink (hex code)
]

# Create a mapping from cluster labels to colors
unique_labels = np.unique(labels_hat)
color_map = {label: distinct_colors[i % len(distinct_colors)] for i, label in enumerate(unique_labels)}

# Plotting settings
plt.rc('font', family='serif', size=11)
figsize = (10, 7)  # Width and height in inches

# Plotting
fig, ax = plt.subplots(figsize=figsize)
background_image = mpimg.imread(sys.argv[5])
ax.imshow(background_image, extent=[-0.25, 0.01, 51.45, 51.555], aspect='auto')

# Plot each cluster
for label in unique_labels:
    cluster_points = merged_df[merged_df['ClusterLabel'] == label]
    ax.scatter(cluster_points['longitude'], cluster_points['latitude'], 
               c=[color_map[label]], label=f'Cluster {label + 1}', 
               s=25, alpha=0.9, edgecolor='black', linewidth=0.4)

# Add labels and title
ax.set_xlim([-0.25, 0.01])
ax.set_ylim([51.45, 51.555])
ax.set_xlabel('Longitude', fontsize=11, labelpad=10)  # Adjust labelpad as needed
ax.set_ylabel('Latitude', fontsize=11, labelpad=10)   # Adjust labelpad as needed

# Position the legend outside the plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and add border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.xaxis.set_tick_params(width=1)
ax.yaxis.set_tick_params(width=1)
ax.xaxis.set_tick_params(direction='out')
ax.yaxis.set_tick_params(direction='out')

# Adjust subplot parameters to control the distance from the edges
plt.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.15)  # Adjust these values as needed

# Save the plot
plt.savefig('santander_bike_stations_clustering_plt.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

# Plot a single cluster or a couple of clusters
fig, ax = plt.subplots(figsize=figsize)
ax.imshow(background_image, extent=[-0.25, 0.01, 51.45, 51.555], aspect='auto')

# Plot specific clusters
for label in [2, 4]:
    cluster_points = merged_df[merged_df['ClusterLabel'] == label]
    ax.scatter(cluster_points['longitude'], cluster_points['latitude'], 
               c=[color_map[label]], label=f'Cluster {label + 1}', 
               s=45, alpha=0.9, edgecolor='black', linewidth=0.4)

# Add labels and title
ax.set_xlim([-0.25, 0.01])
ax.set_ylim([51.45, 51.555])
ax.set_xlabel('Longitude', fontsize=14, labelpad=23.5)  # Adjust labelpad as needed
ax.set_ylabel('Latitude', fontsize=14, labelpad=10)   # Adjust labelpad as needed

# Position the legend outside the plot area
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)

# Adjust layout and add border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.xaxis.set_tick_params(width=1)
ax.yaxis.set_tick_params(width=1)
ax.xaxis.set_tick_params(direction='out')
ax.yaxis.set_tick_params(direction='out')

# Adjust subplot parameters to control the distance from the edges
plt.subplots_adjust(left=0.15, right=0.75, top=0.888, bottom=0.15)  # Adjust these values as needed

# Save the plot
plt.savefig('santander_bike_stations_clustering_subset_plt.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
