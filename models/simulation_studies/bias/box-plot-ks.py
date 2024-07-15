""" 
Script to produce a boxplot of the KS statistic between the empirical distribution of NIRVAR estimates 
and the asymptotic distribution.
"""

#!/usr/bin/env python3
# USAGE: ./box-plot-ks.py ks_statistic.csv


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time 
import sys 

# Read data from the CSV file
file_path = sys.argv[1] 
data = pd.read_csv(file_path, header=None)

colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]

fig = px.box(data)
fig.update_traces(boxpoints='outliers', marker_color='rgb(55, 126, 184)')

# Update the layout based on the provided settings
fig.update_layout(
    yaxis=dict(title=r'$\delta^{(\text{KS})}$', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(title=r'$\hat{\gamma}(\hat{A})_{i}$', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width=500, 
    height=350,
    barmode='overlay'  # This will make the histograms overlap
)

fig.write_image("ks_box.pdf")
time.sleep(1)
fig.write_image("ks_box.pdf")
