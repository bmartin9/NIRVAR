""" 
Script to compare estimated VAR coefficients with ground truth coeffiecients.
Outputs two pdfs containing boxplots of RMSE and ARI
NOTE: It is assumed that phi.csv has shape (N*Q,N*Q) and phi_hat.csv has shape (n_backtest_days,N*N*Q) 
"""

#!/usr/bin/env python3 
# USAGE: ./compare_coefficients.py phi.csv phi_hat.csv backtesting_config.yaml labels_gt.csv labels_hat.csv 

import sys 
import yaml 
import numpy as np 
from sklearn.metrics.cluster import adjusted_rand_score
import plotly.graph_objs as go
import plotly.io as pio

with open(sys.argv[3], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 

###### CONFIG PARAMETERS ###### 
Q = config['Q']
target_feature = config['target_feature']
n_backtest_days = config['n_backtest_days'] 

###### READ IN DATA ######
phi_csv = sys.argv[1]
phi_hat_csv = sys.argv[2]
labels_gt_csv = sys.argv[4]
labels_hat_csv = sys.argv[5] 

phi_gt = np.genfromtxt(phi_csv, delimiter=',') 
N_times_Q = phi_gt.shape[1]
N = N_times_Q/Q 
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N) 
phi_gt = np.reshape(phi_gt,(N,Q,N*Q))
phi_gt_target = phi_gt[:,target_feature,:] 

phi_hat = np.genfromtxt(phi_hat_csv, delimiter=',') 
phi_hat = np.reshape(phi_hat,(n_backtest_days,N,N*Q)) 

labels_gt = np.genfromtxt(labels_gt_csv, delimiter=',') 

labels_hat = np.genfromtxt(labels_hat_csv, delimiter=',')  


# Estimation Accuracy: Frobenius Norm and ARI
ari = np.zeros((n_backtest_days))
RMSE = np.zeros((n_backtest_days))
norm_phi = np.linalg.norm(phi_gt_target)
for t in range(n_backtest_days):
    RMSE[t] = np.linalg.norm(phi_gt_target - phi_hat[t])/norm_phi
    ari[t] = adjusted_rand_score(labels_true = labels_gt,labels_pred = labels_hat[t])
# Produce Boxplots     
ari_trace = go.Box(y=ari) 
RMSE_trace = go.Box(y=RMSE) 

ari_layout = go.Layout(
    xaxis=dict(title='ARI')
)

RMSE_layout = go.Layout(
    xaxis=dict(title='RMSE')
)

ari_fig = go.Figure(data=[ari_trace], layout=ari_layout)
RMSE_fig = go.Figure(data=[RMSE_trace], layout=RMSE_layout) 

pio.write_image(ari_fig, 'ari.pdf')
pio.write_image(RMSE_fig, 'rmse.pdf')


