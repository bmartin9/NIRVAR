""" 
Script to read in the daily volatility measures for each of the 46 stocks and
visualise the time series of volatility measures for each stock.
"""

#!/usr/bin/env python3 
# USAGE: python plot_santander_ts,py <DESIGN_MATRIX>.csv

import pandas as pd
import numpy as np
import sys
import plotly.express as px
import plotly.graph_objs as go
import time 
from statsmodels.tsa.stattools import acf
import statsmodels as sm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


# Read the CSV file
input_file = sys.argv[1]
df = pd.read_csv(input_file)

# Extract the time series of columns 2, 4, and 31
columns_to_plot = [555]
columns_data = df.iloc[:, columns_to_plot]
# columns_data = columns_data[7:] - columns_data[:-7] 
# columns_data = columns_data.diff(periods=7).dropna()
# columns_data = columns_data.diff(periods=1).dropna()

# Create a range of dates from 2000-01-01 to 2012-12-31
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2020-03-10')
date_range = pd.date_range(start=start_date, end=end_date,  periods=len(columns_data))

# Plot the time series using Plotly
fig = go.Figure()

# Add traces for each column
for i, col_idx in enumerate(columns_to_plot):
    fig.add_trace(go.Scatter(x=date_range, y=columns_data.iloc[:, i], mode='lines', name=f'Station {col_idx}'))

# Update layout
fig.update_layout(
                  xaxis_title='Date',
                  yaxis_title='Log Number of rides',)

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
# fig.show()

fig.write_image("santander_ts_example555.pdf")
time.sleep(1)
fig.write_image("santander_ts_example555.pdf")


sm.graphics.tsa.plot_acf(columns_data.values, lags=40)
plt.savefig("autocov_example555.pdf")
plt.close()