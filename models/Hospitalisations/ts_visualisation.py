import sys
import numpy as np 
import plotly.graph_objects as go
import time 
import pandas as pd

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',',skip_header=1)
X_diff = Xs[1:] - Xs[:-1]

dates = pd.date_range(start='2020-04-01', end='2021-07-31', freq='D')

fig = go.Figure()

for i in range(27,28):
    fig.add_trace(go.Scatter(x=dates, y=X_diff[:, i], mode='lines', name=f'Hospital {i+1}'))

fig.update_layout(
                  xaxis_title='Day',
                  yaxis_title='Ventilation Admissions (differenced)')

fig.write_image("hospital_ts_differenced.pdf")
time.sleep(1)
fig.write_image("hospital_ts_differenced.pdf")