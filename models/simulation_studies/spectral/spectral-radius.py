""" 
Compare the RMSE, ARI against the spectral radius of the VAR coefficient matrix for different values of T.

OUTPUT: lineplots of RMSE and ARI against the spectral radius of the VAR coefficient matrix for different values of T.
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
# T = 1000
N=100
Q=1
B=2
d = 2
n_iter = 7 
N_backtest = 1
target_feature = 0 
N_gaussian_mixtures = 2
p_in = 1
p_out = 0
p_between = 0.5

# Create timeseries for each feature of each stock
stocks = ['{0}'.format(i) for i in range(N)] 
features = ['{0}'.format(q) for q in range(Q)]  

#Specify categories manually 
vals = sorted([x%B for x in range(N)])
keys = [str(x) for x in range(N)]
cat = dict(zip(keys,vals))

# Compare how the prediction MSE, RMSE and coefficient of determination vary with spectral radius
observation_lengths = [250,500,750,1000]
num_Ts = len(observation_lengths)
num_radii = 10
num_replicas = 45
spectral_radii = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
plot_data = np.zeros((num_Ts,num_radii,num_replicas,6)) 
svd = TruncatedSVD(n_components=d, n_iter=n_iter, random_state=343) 
uniform_array = random_state.uniform(low=0,high=1.0,size=(N,N))

for r in range(num_radii):
    print(r)
    T = max(observation_lengths)
    rho = spectral_radii[r]
    print(f"rho : {rho}")
    for i in range(num_replicas):
        generator = generativeVAR.generativeVAR(random_state,
                                        T=T,
                                        stock_names=stocks,
                                        feature_names=features,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        p_between=p_between,
                                        multiplier=rho,
                                        categories=cat,
                                        different_innovation_distributions=False,
                                        phi_distribution=None 
                                        )
        groupings_list = list(generator.categories.values())
        Xs = generator.generate()
        A = generator.adjacency_matrix[:,0,:,0]
        for v in range(num_Ts): 

            ###### BACKTESTING ###### 
            N_backtest = 1
            first_prediction_day = observation_lengths[v] - 1
            phi0 = generator.phi_coefficients[:,0,:,:] 
            s_array = np.zeros((N_backtest,N)) 
            s_truth = np.zeros((N_backtest,N)) 
            realised_vals = np.zeros((N_backtest,N)) 
            ols_estimator = np.zeros((N_backtest,N,N,Q))
            ari_bar = 0
            for t in range(N_backtest):
                todays_date = first_prediction_day+t
                #get current embedding
                X_train = Xs[:todays_date,:,:] #Shape = (todays_date,N,Q) 
                current_embedding = train_model.Embedding(d=d,y=X_train)
                current_corr = current_embedding.pearson_correlations() 
                current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=n_iter,random_state=235)

                #get ols params and neighbours
                trainer = train_model.fit(current_embedded_array,X_train,target_feature,UASE_dim=d)
                neighbours , labels = trainer.gmm(k=N_gaussian_mixtures) 
                ari = adjusted_rand_score(labels_true = vals,labels_pred = labels[0])
                ari_bar += ari 
                ols_params = trainer.ols_parameters(neighbours) 
                ols_estimator[t] = ols_params

                #predict next day returns 
                todays_Xs = Xs[todays_date,:,:] 
                predictor = predict_model.predict(ols_params,todays_Xs=todays_Xs)
                s = predictor.next_day_prediction() 
                s_true = predictor.next_day_truth(phi0) 
                s_array[t] = s 
                s_truth[t] = s_true
                realised_vals[t] = todays_Xs[:,0]

            ari_bar = ari_bar/N_backtest 

            # Estimation Accuracy: Frobenius Norm 
            RMSE = 0 
            true_phi = np.reshape(phi0,(N,N*Q),order='F')

            for t in range(N_backtest):
                estimated_phi = np.reshape(ols_estimator[t],(N,N*Q),order='F')
                RMSE_t = np.linalg.norm((true_phi-estimated_phi))/rho
                # RMSE_t = np.linalg.norm(true_phi-np.where(estimated_phi==0,0,estimated_phi))
                M_hat = np.sum(np.where(estimated_phi==0,0,1))
                RMSE_t = RMSE_t/M_hat
                RMSE += RMSE_t
            RMSE = (1/N_backtest)*RMSE

            # Coefficient of determination 
            R2 = 0 
            if N_backtest > 1:
                for j in range(N):
                    y_true = realised_vals[:,j]
                    y_pred = s_array[:,j]
                    Ri2 = r2_score(y_true=y_true,y_pred=y_pred)
                    R2 += Ri2 
            R2 = (1/N)*R2

            # Prediction Error 
            MSE_pred = np.sum((s_truth-s_array)**2)*(1/(N*N_backtest))

            plot_data[v][r][i][0] = rho 
            plot_data[v][r][i][1] = RMSE
            plot_data[v][r][i][2] = int(observation_lengths[v])
            plot_data[v][r][i][3] = R2 
            plot_data[v][r][i][4] = MSE_pred
            plot_data[v][r][i][5] = ari_bar
            
mean_plots = np.mean(plot_data,axis=2)
sem_plots = stats.sem(plot_data,axis=2)
mean_plots = np.reshape(mean_plots,(num_Ts*num_radii,6))     
sem_plots = np.reshape(sem_plots,(num_Ts*num_radii,6))  
pred_df = pd.DataFrame(mean_plots,columns=["spectral_radius","RMSE","T","R2","PMSE","ARI"])    
pred_df["RMSE_sem"] = pd.DataFrame(sem_plots[:,1])
pred_df["R2_sem"] = pd.DataFrame(sem_plots[:,3])
pred_df["PMSE_sem"] = pd.DataFrame(sem_plots[:,4])
pred_df["ARI_sem"] = pd.DataFrame(sem_plots[:,5])

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
        x = "spectral_radius",
        y = 'RMSE',
        error_y = 'RMSE_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        color = 'T',
        markers = '.',
        color_discrete_sequence=colors
    )

fig.update_layout(xaxis_title="Spectral Radius", yaxis_title="RMSE") 

for i, trace in enumerate(fig.data):
    trace.name = [250,500,750,1000][i]

layout = go.Layout(
    xaxis=dict(title='Spectral Radius', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title=r'$\text{RMSE}^{(\text{normalised})}$',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
    legend_title_text="T" 
)

fig.update_layout(layout)

fig.write_image("rmse-spectral-radius.eps")
time.sleep(1)
fig.write_image("rmse-spectral-radius.eps")

fig = utility_funcs.line(
        data_frame = pred_df,
        x = "spectral_radius",
        y = 'ARI',
        error_y = 'ARI_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        color = 'T',
        markers = '.',
        color_discrete_sequence=colors
    )

fig.update_layout(xaxis_title="Spectral Radius", yaxis_title="ARI") 

for i, trace in enumerate(fig.data):
    trace.name = [250,500,750,1000][i]

layout = go.Layout(
    xaxis=dict(title='Spectral Radius', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title='ARI',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
)

fig.update_layout(layout)


fig.write_image("ari-spectral-radius.eps")
time.sleep(1)
fig.write_image("ari-spectral-radius.eps")
