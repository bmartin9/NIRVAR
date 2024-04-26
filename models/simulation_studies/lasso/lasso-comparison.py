""" 
Compare the RMSE of NIRVAR estimated VAR coefficient with a LASSO estimated VAR coefficient for different values of p_out. 

OUTPUT: lineplots
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import LassoLarsIC
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time 

###### SIMULATION PARAMETERS #######
SEED = 352
random_state = default_rng(seed=SEED) 
N = 150
T=1000
Q=1
B=10
d = 10
n_iter = 7 
target_feature = 0 
N_gaussian_mixtures = 10
spectral_radius = 0.9
p_between = 0.5
p_in = 1
T_test = 100
T_train = T - T_test
l1_penalty = 0.05

# Create timeseries for each feature of each stock
stocks = ['{0}'.format(i) for i in range(N)] 
features = ['{0}'.format(q) for q in range(Q)]  

#Specify categories manually 
vals = sorted([x%B for x in range(N)])
keys = [str(x) for x in range(N)]
cat = dict(zip(keys,vals))

# Compare how the prediction MSE, RMSE , ARI and coefficient of determination vary with p_out for block-VAR and LASSO 
p_out_list = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08]
num_p_out = len(p_out_list)
num_replicas = 15
plot_data = np.zeros((num_p_out,num_replicas,3)) 
svd = TruncatedSVD(n_components=d, n_iter=n_iter, random_state=343) 
a = np.zeros((N,N))
b = np.zeros((N,N))
uniform_array = random_state.uniform(low=0,high=1.0,size=(N,N))

for p in range(num_p_out):
    print(p)
    p_out = p_out_list[p]
    for k in range(num_replicas):
        generator = generativeVAR.generativeVAR(random_state,
                                        T=T,
                                        stock_names=stocks,
                                        feature_names=features,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        p_between=p_between,
                                        multiplier=spectral_radius,
                                        categories=cat,
                                        different_innovation_distributions=False,
                                        phi_distribution=uniform_array,
                                        )
        Xs = generator.generate()
        phi0 = generator.phi_coefficients[:,0,:,:] 
        adjacency = generator.adjacency_matrix 
        adjacency_matrix = np.reshape(adjacency,(N,N))
        ###### Block-VAR ###### 
        #get current embedding
        X_train = Xs[:T_train,:,:] #Shape = (T_train,N,Q) 
        current_embedding = train_model.Embedding(d=d,y=X_train)
        current_corr = current_embedding.pearson_correlations()
        current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=n_iter,random_state=235)

        #get ols params and neighbours
        trainer = train_model.fit(current_embedded_array,X_train,target_feature,UASE_dim=d)
        neighbours , labels = trainer.gmm(k=N_gaussian_mixtures)  
        ari = adjusted_rand_score(labels_true = vals,labels_pred = labels[0])
        ols_params = trainer.ols_parameters(neighbours)

        ###### LASSO ######
        clf = Lasso(alpha=l1_penalty,fit_intercept=False)
        # clfIC = LassoCV(fit_intercept=False,cv=5)
        # clf = MultiTaskLassoCV(fit_intercept=False,cv=5)
        # clf = MultiTaskLasso(alpha=l1_penalty,fit_intercept=False)
        clf.fit(X=X_train[:T_train-1,:,0], y=X_train[1:T_train,:,0]) 
        lasso_params = clf.coef_
        # lasso_predictions = clf.predict(X_test[:T_test-1,:,0]) 
        # lasso_score = clf.score(X_test[:T_test-1,:,0],X_test[1:T_test,:,0])

        # clfIC = LassoLarsIC(criterion = 'aic', fit_intercept=False, max_iter=200) # choose penalty using BIC
        # lasso_params = np.zeros((N,N))
        # lasso_predictions = np.zeros((T_test-1,N))
        # lasso_score = np.zeros((N))
        # mean_alpha = 0

        # Estimation Accuracy: Frobenius Norm 
        true_phi = np.reshape(phi0,(N,N*Q),order='F')
        norm_phi = np.linalg.norm(true_phi)
        estimated_phi = np.reshape(ols_params,(N,N*Q),order='F')
        RMSE = np.linalg.norm(true_phi-estimated_phi)/norm_phi 
        lasso_RMSE = np.linalg.norm(true_phi-lasso_params)/norm_phi 
        A_NIRVAR = np.where(estimated_phi ==0, 0, 1) 
        A_LASSO = np.where(lasso_params == 0, 0, 1) 
        count_errors_NIRVAR = 100*np.sum(np.where(A_NIRVAR != adjacency_matrix, 1, 0))/N**2
        count_errors_LASSO = 100*np.sum(np.where(A_LASSO != adjacency_matrix, 1, 0))/N**2 
        

        plot_data[p][k][0] = p_out 
        plot_data[p][k][1] = count_errors_NIRVAR
        plot_data[p][k][2] = count_errors_LASSO

mean_plots = np.mean(plot_data,axis=1)
sem_plots = stats.sem(plot_data,axis=1)
mean_plots = np.reshape(mean_plots,(num_p_out ,3))     
sem_plots = np.reshape(sem_plots,(num_p_out ,3))  
pred_df = pd.DataFrame(mean_plots,columns=[r"$p_{out}$","RMSE","RMSE_LASSO"])    
pred_df["RMSE_sem"] = pd.DataFrame(sem_plots[:,1])
pred_df["RMSE_LASSO_sem"] = pd.DataFrame(sem_plots[:,2])
pred_df["legend_color"] = pd.DataFrame([1 for _ in range(num_p_out)])

###### PLOT ######
fig1 = utility_funcs.line(
        data_frame = pred_df,
        x = r"$p_{out}$",
        y = 'RMSE',
        error_y = 'RMSE_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        markers = '.',
        color = 'legend_color'
    )
fig1.update_traces(line_color='rgb(55, 126, 184)') 

fig2 = utility_funcs.line(
        data_frame = pred_df,
        x = r"$p_{out}$",
        y = 'RMSE_LASSO',
        error_y = 'RMSE_LASSO_sem',
        error_y_mode = 'bar', # Here you say `band` or `bar`.
        markers = '.',
        color = 'legend_color'
    )

fig2.update_traces(line_color='rgb(228, 26, 28)') 


fig = go.Figure(data = fig1.data + fig2.data)
fig.update_xaxes(title=r"$p_{\text{out}}$")
fig.update_yaxes(title=r"RMSE")

legend_labels = ["NIRVAR","LASSO"]

# Apply custom legend labels
for i, trace in enumerate(fig.data):
    trace.name = legend_labels[i]


layout = go.Layout(
    xaxis=dict(title=r"$p^{(\text{out})}$", showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title=r"Variable Selection Error (%)",showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
    legend=dict(x=1.02, y=0.5) 
)

fig.update_layout(layout)
fig.write_image("lasso-comparison.eps")
time.sleep(1)
fig.write_image("lasso-comparison.eps")
