""" 
Compare the RMSE, ARI for different values of \hat{K} and \hat{d} when K and d are fixed.

OUTPUT: lineplots of RMSE and ARI.
"""

#!/usr/bin/env python3
# USAGE: ./spectral-radius.py

from src.models import generativeVAR
from src.models import train_model
from src.models import predict_model
from src.visualization import utility_funcs
import numpy as np
from numpy.random import default_rng
import plotly.express as px
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import r2_score
from sklearn.decomposition import TruncatedSVD 
from scipy import stats
import time
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

###### SIMULATION PARAMETERS #######
SEED = 98927
random_state = default_rng(seed=SEED) 
T = 5000
N=100
Q=1
K=10
d = 2
n_iter = 7 
N_backtest = 1
target_feature = 0 
p_in = 1
p_out = 0
p_between = 0.5
rho = 0.9

# Create timeseries for each feature of each stock
stocks = ['{0}'.format(i) for i in range(N)] 
features = ['{0}'.format(q) for q in range(Q)]  

#Specify categories manually 
vals = sorted([x%K for x in range(N)])
keys = [str(x) for x in range(N)]
cat = dict(zip(keys,vals))

# Compare how the prediction MSE, RMSE and coefficient of determination vary with spectral radius
d_hats = [6,10,14]
num_ds = len(d_hats)
num_replicas = 45
K_hats = [5,6,7,8,9,10,11,12,13,14,15]
num_K_hats = len(K_hats)
plot_data = np.zeros((num_ds,num_K_hats,num_replicas,4)) 
svd = TruncatedSVD(n_components=d, n_iter=n_iter, random_state=343) 
uniform_array = random_state.uniform(low=0,high=1.0,size=(N,N))

for i in range(num_replicas):
    phi_dist = random_state.uniform(0,1,size=(N,N))
    # phi_dist = np.ones((N,N)) 
    generator = generativeVAR.generativeVAR(random_state,
                                    T=T,
                                    stock_names=stocks,
                                    feature_names=features,
                                    B=K,
                                    p_in=p_in,
                                    p_out=p_out,
                                    p_between=p_between,
                                    multiplier=rho,
                                    categories=cat,
                                    different_innovation_distributions=False,
                                    phi_distribution=phi_dist
                                    )
    groupings_list = list(generator.categories.values())
    Xs = generator.generate()
    A = generator.adjacency_matrix[:,0,:,0]
    phi = generator.phi_coefficients[:,0,:,0] 

    # Plot heatmap of phi using plotly
    import plotly.express as px

    fig_phi = px.imshow(phi, color_continuous_scale='Viridis', aspect='auto')
    fig_phi.update_layout(title="Heatmap of phi coefficients")
    # fig_phi.show()

    for j, K_hat in enumerate(K_hats):
        for l, d_hat in enumerate(d_hats): 

            #get current embedding
            current_embedding = train_model.Embedding(d=d_hat,y=Xs)
            current_corr = current_embedding.pearson_correlations() 
            current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=n_iter,random_state=235)

            #get ols params and neighbours
            trainer = train_model.fit(current_embedded_array,Xs,target_feature,UASE_dim=d_hat)
            neighbours , labels = trainer.gmm(k=K_hat) 
            ari = adjusted_rand_score(labels_true = vals,labels_pred = labels[0])
            ols_params = trainer.ols_parameters(neighbours)[:,:,0]
            M_hat = np.sum(np.where(ols_params==0,0,1))
            RMSE = np.linalg.norm(phi-ols_params)


            plot_data[l][j][i][0] = K_hat
            plot_data[l][j][i][1] = RMSE
            plot_data[l][j][i][2] = int(d_hat)
            plot_data[l][j][i][3] = ari
            
mean_plots = np.mean(plot_data,axis=2)
sem_plots = stats.sem(plot_data,axis=2)
mean_plots = np.reshape(mean_plots,(num_ds*num_K_hats,4))     
sem_plots = np.reshape(sem_plots,(num_ds*num_K_hats,4))  
pred_df = pd.DataFrame(mean_plots,columns=["K_hat","RMSE","d_hat","ARI"])    
pred_df["RMSE_sem"] = pd.DataFrame(sem_plots[:,1])
pred_df["ARI_sem"] = pd.DataFrame(sem_plots[:,3])

colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]

fig = utility_funcs.line(
        data_frame = pred_df,
        x = "K_hat",
        y = 'RMSE',
        error_y = 'RMSE_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        color = 'd_hat',
        markers = '.',
        color_discrete_sequence=colors
    )

fig.update_layout(xaxis_title=r"$\hat{K}$", yaxis_title="RMSE") 

for i, trace in enumerate(fig.data):
    trace.name = d_hats[i]

layout = go.Layout(
    xaxis=dict(title=r"$\hat{K}$", showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title="RMSE",showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
    legend_title_text=r"$\hat{d}$" 
)

fig.update_layout(layout)

fig.write_image("rmse.pdf")
time.sleep(1)
fig.write_image("rmse.pdf")

fig = utility_funcs.line(
        data_frame = pred_df,
        x = "K_hat",
        y = 'ARI',
        error_y = 'ARI_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        color = 'd_hat',
        markers = '.',
        color_discrete_sequence=colors
    )

fig.update_layout(xaxis_title=r"$\hat{K}$", yaxis_title="ARI") 

for i, trace in enumerate(fig.data):
    trace.name = d_hats[i]

layout = go.Layout(
    xaxis=dict(title=r"$\hat{K}$", showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title="ARI",showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
    legend_title_text=r"$\hat{d}$" 
)

fig.update_layout(layout)

fig.write_image("ari.pdf")
time.sleep(1)
fig.write_image("ari.pdf")

