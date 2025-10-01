""" 
Simulation study showing a cross validation method for choosing K. 
A grid search of K values is performed. The PMSE for each K is averaged over a given number of 
backtesting days and the average value is plotted for each K. The lowest PMSE should correspond 
to the ground truth K.
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py hyperparameters.yaml 

from src.models import generativeVAR
from src.models import train_model
from src.models import predict_model
import numpy as np
from numpy.random import default_rng
import sys
import yaml
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy import stats
import time
from sklearn.metrics import adjusted_rand_score




with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
N = config["N"] 
T = config["T"] 
K = config["K"]
p_in = config["p_in"]
p_out = config["p_out"]
num_backtest_days = config["num_backtest_days"] 
first_prediction_day = config["first_prediction_day"]
lookback_window = config["lookback_window"] 
spectral_radius = config['spectral_radius']
num_replicas = config['num_replicas']
K_hat_list = config['K_hat_list'] 

num_K_hats = len(K_hat_list)
random_state = default_rng(SEED)

###### GENERATE DATA FROM NIRVAR ######
generator = generativeVAR.generativeVAR(T=T,
                          N=N,
                          B=K,
                          Q=1,
                          p_in = p_in,
                          p_out = p_out,
                          multiplier = spectral_radius,
                          random_state=random_state,
                          phi_distribution=np.ones((N,N))
)

X = generator.generate()
gt_cluster_labels = list(generator.categories.values())

###### DO NIRVAR PREDICTION WITH VARIOUS K_hat VALUES ######
PMSE_values = np.zeros((num_K_hats,num_backtest_days))
PnL_values = np.zeros((num_K_hats,num_backtest_days))

for k in range(num_K_hats):
    print(f"K_hat : {K_hat_list[k]}") 
    for t in range(num_backtest_days):
        X_train = X[first_prediction_day+t-lookback_window:first_prediction_day+t+1,:,:] 

        embedder = train_model.Embedding(y=X_train,
                                         d=K_hat_list[k],
                                         embedding_method='Pearson Correlation',
                                         cutoff_feature=0)
        corr_matrix = embedder.pearson_correlations()
        embedded_matrix = embedder.embed_corr_matrix(corr_matrix=corr_matrix,n_iter=7,random_state=453)
        fit_NIRVAR = train_model.fit(embedded_array=embedded_matrix,
                                     training_set=X_train,
                                     target_feature=0,
                                     UASE_dim=K_hat_list[k]
                                     )
        gmm_clusters, gmm_labels  = fit_NIRVAR.gmm(k=K_hat_list[k])

        ari = adjusted_rand_score(labels_true=gt_cluster_labels,labels_pred=gmm_labels[0])
        if t == 0:
            print(ari)

        phi_hat = fit_NIRVAR.ols_parameters(constrained_array=gmm_clusters)

        prediction = phi_hat[:,:,0]@X_train[-1,:,0] 
        target = X[first_prediction_day+t+1,:,0]

        PMSE = mean_squared_error(target,prediction)
        PMSE_values[k,t] = PMSE 

        PnL_object = predict_model.benchmarking(predictions=prediction,
                                                market_excess_returns=target,
                                                yesterdays_predictions=X[first_prediction_day+t,:,0])
        PnL = PnL_object.PnL(quantile=1)
        PnL_values[k,t] = PnL 


mean_PMSE = np.mean(PMSE_values,axis=-1)
std_errors_PMSE = np.std(a=PMSE_values,axis=-1,ddof=1)/np.sqrt(num_backtest_days)


###### PLOT PMSE AGAINST K_hat ######

x_vals = np.asarray(K_hat_list)

layout = go.Layout(
    yaxis=dict(title="PMSE",showline=True, linewidth=1, linecolor="black",
               ticks="outside", mirror=True),
    xaxis=dict(title=r"$\hat{K}$",showline=True, linewidth=1, linecolor="black",
               ticks="outside", mirror=True, automargin=True),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_family="Serif",
    font_size=14,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)

fig = go.Figure(
    data=[
        go.Scatter(
            x=x_vals,
            y=mean_PMSE,
            mode="lines+markers",
            name="PMSE",
            error_y=dict(
                type="data",
                array=std_errors_PMSE,
                visible=True
            )
        )
    ],
    layout=layout
)

fig.write_image("PMSE_K_hat.pdf")
time.sleep(1)
fig.write_image("PMSE_K_hat.pdf")

