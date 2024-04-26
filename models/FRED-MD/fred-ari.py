""" 
Script to compute the ARI between NIRVAR estimated groups and FRED-MD groups over time.
"""

#!/usr/bin/env python3
# USAGE: ./fred-ari.py <DESIGN_MATRIX>.csv labels_hat.csv FRED-MD_updated_appendix.csv 

import sys 
import csv 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio
import time
import plotly.express as px
import pandas as pd
from numpy import genfromtxt
from sklearn.metrics.cluster import adjusted_rand_score



design_file_name = sys.argv[1]
labels_hat_filename = sys.argv[2]
appendix_filename = sys.argv[3] 

# Step 1: Create a mapping from the first CSV file
mapping = {}
with open(appendix_filename, 'r',encoding='cp1252') as file1:
    reader = csv.reader(file1)
    next(reader)  # Skip header row
    for row in reader:
        name = row[2]  # Assuming 3rd column contains names
        group_label = row[6]  # Assuming 7th column contains group labels
        mapping[name] = group_label

# Step 2: Read header from the second CSV file and create the output dictionary
output_dict = {}
with open(design_file_name, 'r') as file2:
    reader = csv.reader(file2)
    header = next(reader)  # Read header row
    i = 0 
    for name in header:
        if name in mapping:
            output_dict[name] = mapping[name]
        else:
            print(name) 
            print(i)
        i += 1 


def insert_at_index(orig_dict, key, value, index):
    """ Insert a key-value pair into a dictionary at a specific index. """
    if not 0 <= index <= len(orig_dict):
        raise IndexError("Index out of range")

    new_dict = {}
    for i, (k, v) in enumerate(orig_dict.items()):
        if i == index:
            new_dict[key] = value
        new_dict[k] = v

    # If the index is equal to the length of the original dict, add the new item at the end.
    if index == len(orig_dict):
        new_dict[key] = value

    return new_dict

fred_labels_dict = insert_at_index(output_dict,"IPB51222S",1,16)
fred_labels_list = list(fred_labels_dict.values())
fred_labels_array = np.array([(int(e)-1) for e in fred_labels_list])
# print(fred_labels_array)

def get_kth_row_from_csv(file_path, k):
    """ Read a CSV file and return the k-th row as a list. """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == k:
                return row
    # Return None or raise an error if k is out of range
    return None

nirvar_labels = get_kth_row_from_csv(labels_hat_filename,k=1) 
nirvar_labels = [int(element) for element in nirvar_labels]

labels_hat = genfromtxt(sys.argv[2], delimiter=',')

T = labels_hat.shape[0]
print(T)
estimator_v_fred_ari = np.zeros((T)) 
for t in range(T):
    estimator_v_fred_ari[t] = adjusted_rand_score(labels_hat[t], fred_labels_array)

dates = pd.date_range(start='2000-01-01', end='2019-12-31',periods=T)

# Create a trace for the line plot
trace = go.Scatter(x=dates, y=estimator_v_fred_ari)

# Create the layout for the plot
layout = go.Layout(xaxis=dict(title='Month'),  yaxis=dict(title='ARI'))

# Create the figure and add the trace and layout
fig = go.Figure(data=[trace],layout=layout)

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

pio.write_image(fig, 'NIRVAR_vs_FRED_ari.eps')
time.sleep(1)
pio.write_image(fig, 'NIRVAR_vs_FRED_ari.eps')

ari_heatmap_array = np.zeros((T, T))
for i in range(T):
    print(f"i: {i}")
    for j in range(i, T):
        ari = adjusted_rand_score(labels_hat[i], labels_hat[j])
        ari_heatmap_array[i, j] = ari
        ari_heatmap_array[j, i] = ari  # Symmetric matrix, set the mirrored value

# The ARI for the same pair of label arrays is 1, so set the diagonal to 1
np.fill_diagonal(ari_heatmap_array, 1)

# np.savetxt('ari_heatmap.csv', ari_heatmap_array, delimiter=',', fmt='%.6f')

# Create a trace for the heatmap    
import plotly.graph_objs as go

# Create a heatmap using Plotly
fig = go.Figure(data=go.Heatmap(z=ari_heatmap_array, x=dates, y=dates, colorscale='RdBu',zmin=-1,zmax=1))

layout = go.Layout(xaxis=dict(title='Month'),  yaxis=dict(title='Month'))

fig.update_layout(layout)

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

# Show the plot
pio.write_image(fig, 'ari_heatmap_fred.eps')
time.sleep(1)
pio.write_image(fig, 'ari_heatmap_fred.eps')
