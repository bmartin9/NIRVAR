""" 
Script to produce plots comparing NIRVAR estimated groups with the 8 FRED groups
"""

#!/usr/bin/env python3
# USAGE: ./groups-visualisation.py <DESIGN_MATRIX>.csv labels_hat.csv FRED-MD_updated_appendix.csv 

import sys 
import csv 
import numpy as np 
import plotly.graph_objects as go
import plotly.io as pio
import time
import plotly.express as px
import pandas as pd


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
print(fred_labels_list)

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


print(f"True Labels Array: {fred_labels_array.shape}") 

estimated_d = len(set(nirvar_labels)) 
print(f"Estimated d: {estimated_d}")

# Create a new list of length 10*estimated_d and initialize with zeros
result_list = [0] * (8*estimated_d)

# Count the occurrences of combinations of values from list1 and list2
N = 122
print(N)
for i in range(N):
    result_list[int(fred_labels_array[i]) * estimated_d + int(nirvar_labels[i])] += 1 

sic_sector_names = [
    "Output and income",
    "Labor market",
    "Housing",
    "Consumption, orders, and inventories",
    "Money and credit",
    "Interest and exchange rates",
    "Prices",
    "Stock market",
]

sic_sector_names = ["(i)","(ii)","(iii)","(iv)","(v)","(vi)","(vii)","(viii)"]


sector_list = [] 
sector_names_list = []
cluster_list = []
for i in range(8):
    for j in range(estimated_d):
        sector_list.append(i)
        sector_names_list.append(sic_sector_names[i]) 
        cluster_list.append(j)

df = pd.DataFrame({"Sector" : sector_list, "SectorNames" : sector_names_list, 'EstimatedCluster' : cluster_list, "Count" : result_list })
print(df.head())

fig = px.bar(df, x="SectorNames", y="Count", color="EstimatedCluster", text_auto=True)
pio.write_image(fig, 'fred_stacked_barplot.eps')
time.sleep(1)
pio.write_image(fig, 'fred_stacked_barplot.eps')

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

fig = px.bar(df, x="EstimatedCluster", y="Count", color="SectorNames",color_discrete_sequence=colors[:8])

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


fig.update_xaxes(title_text="Estimated Cluster")
fig.update_layout(legend_title="FRED Group")

# Update the x-axis tick labels and tick values
tick_labels = [str(i) for i in range(1, estimated_d+1)]
tick_values = list(range(0, estimated_d))

fig.update_xaxes(ticktext=tick_labels, tickvals=tick_values)

pio.write_image(fig, 'fred_stacked_barplot_k1.eps')
time.sleep(1)
pio.write_image(fig, 'fred_stacked_barplot_k1.eps')