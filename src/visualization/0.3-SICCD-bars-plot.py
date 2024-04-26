# Script to produce bar plot comparing NIRVAR estimated clusters with SIC sectors

#!/usr/bin/env python3 
# USAGE: ./0.3-ARI-comparisons.py 20201231.csv labels_hat.csv pvCLCL.csv rows_to_delete.csv

import argparse 
import numpy as np 
import csv 
from numpy import genfromtxt
from sklearn.metrics.cluster import adjusted_rand_score
import plotly.io as pio
import pandas as pd
import plotly.express as px
import time
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

last_day_labels_hat = labels_hat[-1] 
estimated_d = len(set(last_day_labels_hat)) 
print(f"Estimated D: {estimated_d}")

# Create a new list of length 10*estimated_d and initialize with zeros
result_list = [0] * (10*estimated_d)

# Count the occurrences of combinations of values from list1 and list2
print(N)
for i in range(N):
    result_list[int(true_labels_array[i]) * estimated_d + int(last_day_labels_hat[i])] += 1 

sic_sector_names = [
    "Agriculture, Forestry, and Fishing",
    "Mining",
    "Construction",
    "Manufacturing",
    "Transportation and Public Utilities",
    "Wholesale Trade",
    "Retail Trade",
    "Finance, Insurance, and Real Estate",
    "Services",
    "Public Administration"
]

sic_sector_names = ["(i)","(ii)","(iii)","(iv)","(v)","(vi)","(vii)","(viii)","(ix)","(x)"]

sector_list = [] 
sector_names_list = []
cluster_list = []
for i in range(10):
    for j in range(estimated_d):
        sector_list.append(i)
        sector_names_list.append(sic_sector_names[i]) 
        cluster_list.append(j)

df = pd.DataFrame({"Sector" : sector_list, "SectorNames" : sector_names_list, 'EstimatedCluster' : cluster_list, "Count" : result_list })
print(df.head())

fig = px.bar(df, x="SectorNames", y="Count", color="EstimatedCluster", text_auto=True)
pio.write_image(fig, 'SIC_stacked_barplot.eps')
time.sleep(1)
pio.write_image(fig, 'SIC_stacked_barplot.eps')

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

fig = px.bar(df, x="EstimatedCluster", y="Count", color="SectorNames",color_discrete_sequence=colors)

dfs = df.groupby("EstimatedCluster").sum()

# fig.add_trace(go.Scatter(
#     x=dfs.index, 
#     y=dfs['Count'],
#     text=dfs['Count'],
#     mode='text',
#     textposition='top center',
#     textfont=dict(
#         size=12,
#     ),
#     showlegend=False
# ))

fig.update_xaxes(title_text="Estimated Cluster")
fig.update_layout(legend_title="SIC Sector")

layout = go.Layout(
    yaxis=dict( showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)
fig.update_layout(layout)

# Update the x-axis tick labels and tick values
tick_labels = [str(i) for i in range(1, 21)]
tick_values = list(range(0, 20))

fig.update_xaxes(ticktext=tick_labels, tickvals=tick_values)

pio.write_image(fig, 'SIC_stacked_barplot_20.eps')
time.sleep(1)
pio.write_image(fig, 'SIC_stacked_barplot_20.eps')
