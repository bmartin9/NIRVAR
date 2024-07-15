""" 
Plot the number of factors over time
"""

#!/usr/bin/env python3 
# USAGE: ./line_plot factors.csv

import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np 
import plotly.io as pio
import sys
import time

# Read the CSV file
file_path = sys.argv[1] 
df = pd.read_csv(file_path,header=None)  # Replace 'your_file.csv' with your file path

# index_of_zero = np.where(df.values == 0)[0][]
# print(index_of_zero)

# Assuming the column name is 'Value', change it to your actual column name
# If your column doesn't have dates, generate date range
date_range = pd.date_range(start='2004-01-01', end='2020-12-31', periods=4032) 
df['Date'] = date_range


# Plotting using Plotly
fig = px.line(df, x='Date', y=df[0])
fig.update_yaxes(title_text="Number of Factors")
# Save the figure as a PNG file
pio.write_image(fig, 'factors.pdf')
time.sleep(2)
pio.write_image(fig, 'factors.pdf')
