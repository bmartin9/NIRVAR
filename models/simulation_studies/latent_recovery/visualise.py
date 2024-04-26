""" 
Visualise the marginal distributions of X_Sample_Covariance and compare with X_Gamma 
"""

#!/usr/bin/env python3
# USAGE: ./visualise.py hyperparameters.yaml X_Sample_Covariance.csv X_point1.csv  X_point2.csv X_point2.csv

import numpy as np
from numpy.random import default_rng
import sys
import yaml
from numpy import genfromtxt
import plotly.graph_objects as go
import time
from scipy.stats import norm
import plotly.express as px
from scipy import stats

with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
N = config['N1']
T = config['T']
Q = config['Q']
B = config['B']
d = config['d']
sigma = config['sigma']
SEED = config['SEED']
N_replicas = config['N_replicas']
H = config['H'] 
spectral_radius = config['spectral_radius']
n_iter = config['n_iter']

random_state = default_rng(seed=SEED)

###### READ IN DATA ###### 
X_Gamma = genfromtxt(sys.argv[3],delimiter=",")
X_Phi = genfromtxt(sys.argv[4],delimiter=",")
X_gt = genfromtxt(sys.argv[5],delimiter=",")
X_S = genfromtxt(sys.argv[2],delimiter=",")
X_S = np.reshape(X_S,(N_replicas,N,d)) 

colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]

component_to_visualise = 21
dimension_to_visualise = 0

X_S_hist_data = X_S[:,component_to_visualise,dimension_to_visualise]

X_S_mean = np.mean(X_S_hist_data) 
X_S_sdt = np.std(X_S_hist_data)
X_S_max = np.max(np.abs(X_S_hist_data))

X_Gamma_loc = X_Gamma[component_to_visualise,dimension_to_visualise]
X_Gamma_loc_d_subspace = X_Phi[component_to_visualise,dimension_to_visualise]
X_Phi_loc = X_gt[component_to_visualise,dimension_to_visualise]

X_S_hist = go.Histogram(x=X_S_hist_data,
                        histnorm='probability density',
                        nbinsx=100,
                        marker=dict(color=colors[0]),
                        name = r'$X_{S}^{(k)}$'
                        )

x_values = np.linspace(-X_S_max, X_S_max, 1000)  # Adjust range as needed
y_values = norm.pdf(x_values, loc=X_S_mean, scale=X_S_sdt) 
normal_distribution_trace = go.Scatter(x=x_values, y=y_values, mode='lines', name=r'$\mathcal{N}(<{X_{S}^{(k)}}>, \sigma_{X_{S}}^{2})$', line=dict(color=colors[1]))

fig = go.Figure(data=[X_S_hist])

# Add a red vertical line at X_Gamma_loc
# fig.add_vline(x=X_Gamma_loc, line=dict(color=colors[2], width=3),name=r'$X_{\Gamma -SVD}$')
fig.add_vline(x=X_Gamma_loc_d_subspace, line=dict(color=colors[3], width=2),name=r'$X_{\Phi}$')

layout = go.Layout(
    yaxis=dict(title='Probability', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title='Embedded Location',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)

fig.update_layout(layout)

fig.write_image("location-marginal-distribution.pdf")
time.sleep(1)
fig.write_image("location-marginal-distribution.pdf")

###### Contour Plot ######
X_S_contour_data = X_S[:,component_to_visualise,:d]
X_Gamma_contour_loc = X_Gamma[component_to_visualise,:d] 
X_Gamma_contour_loc_d_subspace = X_Phi[component_to_visualise,:d] 
X_gt_loc = X_gt[component_to_visualise,:d] 
X_gt_loc_block2 = X_gt[100,:d] 

X_S_block1_contour_data = X_S[0,:75,:d]
X_S_block2_contour_data = X_S[0,75:,:d]


# Prepare data for contour plot
X_contour = X_S_contour_data[:, 0]
Y_contour = X_S_contour_data[:, 1] 

X_contour_block1 = X_S_block1_contour_data[:, 0]
Y_contour_block1 = X_S_block1_contour_data[:, 1] 

X_contour_block2 = X_S_block2_contour_data[:, 0]
Y_contour_block2 = X_S_block2_contour_data[:, 1] 

# Create a scatter plot for X_S_contour_data in blue
fig_contour = go.Figure()
# fig_contour.add_trace(go.Scatter(x=X_contour, y=Y_contour, mode='markers', marker=dict(color=colors[0]), name='$(Y^{(S)})_{i}$'))

fig_contour.add_trace(go.Scatter(x=X_contour_block1, y=Y_contour_block1, mode='markers', marker=dict(color=colors[0]), name='$(Y^{(S)})_{1:150}$'))
fig_contour.add_trace(go.Scatter(x=X_contour_block2, y=Y_contour_block2, mode='markers', marker=dict(color=colors[2]), name='$(Y^{(S)})_{151:300}$'))

# Create a scatter plot for X_Gamma_contour_loc in red
fig_contour.add_trace(go.Scatter(x=[X_gt_loc[0]], y=[X_gt_loc[1]], mode='markers', marker=dict(color=colors[1]), name='$(Y_{B})_{1}$'))
fig_contour.add_trace(go.Scatter(x=[X_gt_loc_block2[0]], y=[X_gt_loc_block2[1]], mode='markers', marker=dict(color=colors[3]), name='$(Y_{B})_{2}$'))

layout = go.Layout(
    yaxis=dict(title=r'Second principle direction', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title=r'First principle direction',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)

fig_contour.update_layout(layout)

fig_contour.write_image("contour_marginal_dist.pdf")
time.sleep(1)
fig_contour.write_image("contour_marginal_dist.pdf")

# Plot of correlation matrix between dimensions 
corr = np.corrcoef(X_contour,Y_contour) 

max_corr = np.max(np.abs(corr))
fig = px.imshow(corr,zmin=-max_corr,zmax=max_corr,color_continuous_scale='RdBu')
fig.write_image("correlation_heatmap.eps")

###### Create a Q-Q plot ######
qq_data = stats.probplot((Y_contour_block2-np.mean(Y_contour_block2))/np.std(Y_contour_block2), dist="norm", plot=None)

# Extracting x and y coordinates for the Q-Q plot
x = qq_data[0][0]
y = qq_data[0][1]

# Create a scatter plot using Plotly
fig_qq = go.Figure()

# Add the reference line
fig_qq.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines',name=None, showlegend=False, line=dict(color=colors[6])))

# Add the Q-Q plot
fig_qq.add_trace(go.Scatter(x=x, y=y, mode='markers', name='$(Y^{(S)})_{1:75,0}$', marker=dict(color=colors[0])))


layout = go.Layout(
    yaxis=dict(title='Sample Quantiles', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title='Theoretical Quantiles',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350
)

fig_qq.update_layout(layout)
fig_qq.write_image("qq_plot_blocks.pdf")
time.sleep(1)
fig_qq.write_image("qq_plot_blocks.pdf")