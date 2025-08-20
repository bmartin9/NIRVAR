""" 
Look at group recovery by computing the ARI between the NIRVAR estimated groups and the ground truth clusters for different values of spectral radius and T.
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
from numpy.linalg import eigvals
from scipy.stats import multivariate_t 
from scipy import stats
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import time
import plotly.graph_objects as go
from src.visualization import utility_funcs


with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
N = config["N"] 
T_list = config["T_list"] 
K = config["K"]
K_hat = config["K_hat"]
p_in = config["p_in"]
p_out = config["p_out"]
t_distribution_dof = config["t_distribution_dof"] 
NIRVAR_spectral_radius_list = config['NIRVAR_spectral_radius_list']
heavy_tailed_errors = config['heavy_tailed_errors']
std_VAR_coefficients = config['std_VAR_coefficients']
VAR_burnin = config['VAR_burnin']
NIRVAR_p = config['NIRVAR_p']
num_replicas = config['num_replicas']
estimated_VAR_p = config['estimated_VAR_p']
num_backtest_days = config['num_backtest_days']


num_T = len(T_list)
num_rho = len(NIRVAR_spectral_radius_list) 
rng = default_rng(SEED)


###### UTILITY FUNCTIONS ######
def generate_NIRVAR_coefficients(
    N: int,
    K: int,
    p_in: float,
    p_out: float,
    num_lags: int = 1,
    NIRVAR_spectral_radius: float = 0.9,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate coefficient matrices for NIRVAR model.

    Parameters
    ----------
    N : int
        Number of variables.
    K : int
        Number of equal-sized blocks along each axis (N must be divisible by K).
    p_in : float
        Probability that an entry *within* the same block is kept (1-mask).
    p_out : float
        Probability that an entry *between* different blocks is kept.
    num_lags : int, default 1
        Order of the VAR model (number of Phi matrices to generate).
    NIRVAR_spectral_radius : float, default 0.9
        Target spectral radius for the companion matrix.
    rng : np.random.Generator or None, default None
        NumPy random generator (uses `np.random.default_rng()` if omitted).

    Returns
    -------
    Phi : ndarray, shape (num_lags, N, N)
        The generated coefficient matrices, already rescaled so that the
        companion matrix has the requested spectral radius.
    """
    # ----- basic checks & setup ------------------------------------------------
    if rng is None:
        rng = np.random.default_rng()

    if N % K != 0:
        raise ValueError("K must divide N so each block is equally sized.")

    block_size = N // K
    Phi = np.empty((num_lags, N, N), dtype=float)

    # ----- build the (N,N) matrix of keep-probabilities -----------------------
    # For every (i, j), decide whether i and j lie in the same block.
    block_ids = np.repeat(np.arange(K), block_size)          # (N,)
    same_block = block_ids[:, None] == block_ids[None, :]    # (N, N) bool

    P_keep = np.where(same_block, p_in, p_out)               # (N, N) floats

    # ----- draw masks & fill Phi_j -------------------------------------------
    for j in range(num_lags):
        mask = rng.random((N, N)) < P_keep                   # Bernoulli mask
        # Phi_half = rng.normal(size=(N, N)) 
        # Phi_Wishart = Phi_half @ Phi_half.T                # Wishart ⇒ SPD
        # Phi[j] = (Phi_Wishart) * mask 
        # Phi[j] = rng.normal(size=(N, N)) * mask            # Uniform ⇒ [0, 1] matrix
        # random_negative_mean = rng.binomial(1,0.5,size=(K))
        # random_negative_mean = np.array([0,1])
        # random_negative_mean = np.where(random_negative_mean==1,-1,1)
        for i in range(N):
            # mean = random_negative_mean[block_ids]*block_ids[i]
            # mean = 3*mean 
            # Phi[j][i] = rng.normal(loc=mean,scale=0.1,size=(N)) * mask[i]
            Phi[j][i] = np.repeat(np.array([1,-1]),block_size) * mask[i] * rng.uniform(0,1,size=N) 

    # ----- rescale so companion matrix meets the spectral-radius target ------
    if num_lags > 0:
        # Build the (N*num_lags, N*num_lags) companion matrix F.
        #   F = [[Phi_1 … Phi_p],
        #        [I_N    0   ],
        #        [  …      … ]]
        p = num_lags
        row0 = np.concatenate(Phi, axis=1)                   # (N, N*p)
        if p == 1:
            F = row0
        else:
            eye = np.eye(N * (p - 1))
            F = np.zeros((N * p, N * p))
            F[:N, :] = row0
            F[N:, :-N] = eye

        # Current spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(F)))
        if rho > 0:                                          # avoid divide-by-0
            Phi *= (NIRVAR_spectral_radius / rho)

    return Phi

def generate_NIRVAR_data(
    T: int,
    N: int,
    Phi: np.ndarray,
    num_lags: int = 1,
    target_rho: float = 0.9,
    t_distribution_dof: float = 5,
    burnin: int = 200,
    heavy_tailed_errors: bool = False,
    Sigma_type: str = "Identity", # "Wishart" or "Identity"
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Given a set of coefficient matrices Phi, generate data from a NIRVAR(p) model.
    """
    companion = np.vstack(
        [np.hstack(Phi), np.eye(N * (num_lags - 1), N * num_lags)]
    )
    rho0 = max(abs(eigvals(companion)))          # current spectral radius
    for _ in range(20):
        if rho0 < target_rho + 0.03:
            break
        else:
            Phi *= 0.8*target_rho / rho0
            companion = np.vstack([np.hstack(Phi),
                                   np.eye(N * (num_lags - 1), N * num_lags)])
            rho0 = max(abs(eigvals(companion)))
    
    Phi *= target_rho / rho0                       # bring it down/up to target

    # sanity-check
    new_rho = max(abs(eigvals(
        np.vstack([np.hstack(Phi), np.eye(N * (num_lags - 1), N * num_lags)])
    )))
    print(new_rho)
    assert new_rho <= target_rho + 0.03, "VAR not stable to requested threshold!"

    # ----- 3. innovations ----------------------------------------------------
    if Sigma_type == "Identity":
        Sigma = np.eye(N)
    elif Sigma_type == "Wishart":
        Sigma = rng.standard_normal((N, N))
        Sigma = Sigma @ Sigma.T                     # Wishart ⇒ SPD
        Sigma /= np.max(np.diag(Sigma))             # rescale variances ≈ 1

    # ----- 4. simulate -------------------------------------------------------
    total_T = T + burnin
    y = np.zeros((total_T, N))

    if heavy_tailed_errors:
        mvt = multivariate_t(loc=np.zeros(N), shape=Sigma, df=t_distribution_dof,seed=rng)
        for t in range(num_lags, total_T):
            deterministic = sum(Phi[j] @ y[t - j - 1] for j in range(num_lags))
            eps_t = mvt.rvs(random_state=rng)    # Student-t shock
            y[t] = deterministic + eps_t 
            
    else:
        for t in range(num_lags, total_T):
            deterministic = sum(Phi[j] @ y[t - j - 1] for j in range(num_lags))
            eps_t = rng.multivariate_normal(mean=np.zeros((N)), cov=Sigma)    # Gaussian shock
            y[t] = deterministic + eps_t

    return y[burnin:]

vals = sorted([x%K for x in range(N)])
plot_data = np.zeros((num_T,num_rho,num_replicas,6)) 


for r in range(num_rho):
    print(r)
    T = max(T_list)
    rho = NIRVAR_spectral_radius_list[r]
    print(f"rho : {rho}")
    for i in range(num_replicas):

        # generator = generativeVAR.generativeVAR(rng,
        #                                 T=T,
        #                                 N = N,
        #                                 Q=1,
        #                                 B=K,
        #                                 p_in=p_in,
        #                                 p_out=p_out,
        #                                 multiplier=rho,
        #                                 different_innovation_distributions=False,
        #                                 phi_distribution=None 
        #                                 )
        # groupings_list = list(generator.categories.values())
        # X_generated = generator.generate()[:,:,0]
        # Phi = generator.phi_coefficients[:,0,:,0]

        Phi = generate_NIRVAR_coefficients(
                        N=N,
                        K=K,
                        p_in=p_in,
                        p_out=p_out,
                        num_lags=NIRVAR_p,
                        NIRVAR_spectral_radius=rho,
                        rng=rng
                    )
        
        # fig_phi = go.Figure(data=go.Heatmap(
        #     z=Phi[0],  # plot first lag's Phi matrix
        #     colorscale='Viridis',
        #     colorbar=dict(title='Phi[0] value'),
        # ))
        # fig_phi.update_layout(
        #     title=f'Heatmap of Phi[0], spectral radius={rho:.2f}',
        #     xaxis_title='Column',
        #     yaxis_title='Row',
        #     width=400,
        #     height=350,
        #     font_family="Serif",
        #     font_size=12,
        #     margin=dict(l=5, r=5, t=30, b=5),
        #     paper_bgcolor='white',
        #     plot_bgcolor='white',
        # )
        # fig_phi.write_image(f"Phi_heatmap_rho_{rho:.2f}_rep_{i}.pdf")
                    
        X_generated = generate_NIRVAR_data(
                        T=T,
                        N=N,
                        Phi=Phi,
                        num_lags=NIRVAR_p,
                        target_rho=rho,
                        burnin=VAR_burnin,
                        Sigma_type="Identity",
                        rng=rng
                    )
        
        for v in range(num_T): 
            print(f"T : {T_list[v]}")

            ###### BACKTESTING ###### 
            N_backtest = num_backtest_days
            first_prediction_day = T_list[v] - N_backtest -1
            predictions_gt_phi = np.zeros((N_backtest,N)) 
            predictions_NIRVAR1 = np.zeros((N_backtest,N)) 
            predictions_NIRVARp = np.zeros((N_backtest,N)) 
            realised_vals = np.zeros((N_backtest,N)) 
            ari_bar = 0
            for t in range(N_backtest):
                todays_date = first_prediction_day+t
                #get current embedding
                X_train = X_generated[:todays_date+1,:] 
                # Z = np.hstack([X_train[1:], X_train[:-1]])
                # autocorrelations = np.cov(Z.T)  # autocorrelations of the time series
                # current_embedding = train_model.Embedding(d=K,y=Z[:,:,np.newaxis])
                current_embedding = train_model.Embedding(d=K,y=X_train[:,:,np.newaxis])
                current_corr = current_embedding.pearson_correlations() 
                current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=7,random_state=235)
                # current_embedded_array = current_embedding.embed_corr_matrix(autocorrelations,n_iter=7,random_state=235)
                
                #get ols params and neighbours
                trainer = train_model.fit(current_embedded_array,X_train[:,:,np.newaxis],target_feature=0,UASE_dim=K)
                # trainer = train_model.fit(current_embedded_array,Z[:,:,np.newaxis],target_feature=0,UASE_dim=K)
                neighbours , labels = trainer.gmm(k=K_hat) 
                ari = adjusted_rand_score(labels_true = vals,labels_pred = labels[0])
                ari_bar += ari 
                ols_params_NIRVAR1 = trainer.ols_parameters(neighbours) 
                ols_params_NIRVARp = trainer.ols_parameters_NIRVARp(neighbours,p=NIRVAR_p) 

                #predict next day returns 
                todays_predictors = X_generated[todays_date-NIRVAR_p+1:todays_date+1,:] 
                todays_targets = X_generated[todays_date+1,:] 
                predictions_gt_phi[t] = sum(Phi[j] @ todays_predictors[NIRVAR_p -1 - j] for j in range(NIRVAR_p))
                # predictor_NIRVAR1 = predict_model.predict(ols_params_NIRVAR1,todays_Xs=todays_predictors[-1,:])
                # predictions_NIRVAR1[t] = predictor_NIRVAR1.next_day_prediction()
                # predictions_NIRVARp[t] = sum(ols_params_NIRVARp[:,:,j] @ todays_predictors[NIRVAR_p -1 - j] for j in range(NIRVAR_p))
                realised_vals[t] = todays_targets 

            ari_bar = ari_bar/N_backtest 

            # Prediction Error 
            MSPE_gt_phi = np.sum((predictions_gt_phi-realised_vals)**2)*(1/(N*N_backtest))
            MSPE_NIRVAR1 = np.sum((predictions_NIRVAR1-realised_vals)**2)*(1/(N*N_backtest))
            MSPE_NIRVARp = np.sum((predictions_NIRVARp-realised_vals)**2)*(1/(N*N_backtest))

            plot_data[v][r][i][0] = rho 
            plot_data[v][r][i][1] = int(T_list[v])
            plot_data[v][r][i][2] = MSPE_gt_phi 
            plot_data[v][r][i][3] = MSPE_NIRVAR1
            plot_data[v][r][i][4] = MSPE_NIRVARp
            plot_data[v][r][i][5] = ari_bar
            
mean_plots = np.mean(plot_data,axis=2)
sem_plots = stats.sem(plot_data,axis=2)
mean_plots = np.reshape(mean_plots,(num_T*num_rho,6))     
sem_plots = np.reshape(sem_plots,(num_T*num_rho,6))  
pred_df = pd.DataFrame(mean_plots,columns=["spectral_radius","T","MSPE_gt_phi","MSPE_NIRVAR1","MSPE_NIRVARp", "ARI"])    
pred_df["MSPE_gt_phi_sem"] = pd.DataFrame(sem_plots[:,2])
pred_df["MSPE_NIRVAR1_sem"] = pd.DataFrame(sem_plots[:,3])
pred_df["MSPE_NIRVARp_sem"] = pd.DataFrame(sem_plots[:,4])
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


fig.write_image("ari-spectral-radius.pdf")
time.sleep(1)
fig.write_image("ari-spectral-radius.pdf")


layout = go.Layout(
    xaxis=dict(title='T', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    yaxis=dict(title='MSPE',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
)

fig = go.Figure(layout=layout)

fig.add_trace(go.Scatter(
    x=pred_df["T"], y=pred_df["MSPE_gt_phi"],
    mode="lines+markers", name="MSPE_gt_phi",
    line=dict(color=colors[0]),
    error_y=dict(type="data", array=pred_df["MSPE_gt_phi_sem"], visible=True),
))

fig.add_trace(go.Scatter(
    x=pred_df["T"], y=pred_df["MSPE_NIRVAR1"],
    mode="lines+markers", name="MSPE_NIRVAR1",
    line=dict(color=colors[1]),
    error_y=dict(type="data", array=pred_df["MSPE_NIRVAR1_sem"], visible=True),
))

fig.add_trace(go.Scatter(
    x=pred_df["T"], y=pred_df["MSPE_NIRVARp"],
    mode="lines+markers", name="MSPE_NIRVARp",
    line=dict(color=colors[2]),
    error_y=dict(type="data", array=pred_df["MSPE_NIRVARp_sem"], visible=True),
))

fig.write_image("MSPE-T.pdf")
time.sleep(1)
fig.write_image("MSPE-T.pdf")