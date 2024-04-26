# Script to create a line plot of the log mse of NIRVAR fred predictions as the embedding dimension d is varied.

import pandas as pd
import plotly.express as px
import numpy as np
import sys 
import plotly.io as pio 
import time 

filename = sys.argv[1]

df = pd.read_csv(filename )  
# Extract values from the columns
x_values = df.iloc[:, 0]
y_values = np.log(df.iloc[:, 1])

# Create a line plot using Plotly Express
fig = px.line(x=x_values, y=y_values, labels={'x': '$d$', 'y': 'Log MSE'})

pio.write_image(fig, '../../data/interim/meetings/17-01-24/FRE-mse-against-d.pdf')
time.sleep(1)
pio.write_image(fig, '../../data/interim/meetings/17-01-24/FRE-mse-against-d.pdf') 

