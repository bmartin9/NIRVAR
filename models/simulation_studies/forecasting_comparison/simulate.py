""" 
Script to simulate data using various models for as the data generation process.
Then compare the forecasting MSPE of various models over a backtesting period.
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py hyperparameters.yaml 

from src.models import generativeVAR
from src.models import train_model
from src.models import predict_model
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import Lasso
import sys
import yaml
from statsmodels.tsa.api import VAR
from numpy.linalg import eigvals
from scipy.stats import multivariate_t 


with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
N = config["N"] 
T = config["T"] 
K = config["K"]
p_in = config["p_in"]
p_out = config["p_out"]
VAR_p = config["VAR_p"] 
lasso_penalty = config["lasso_penalty"] 
num_backtest_days = config["num_backtest_days"] 
first_backtest_day = config["first_backtest_day"]
lookback_window = config["lookback_window"] 
t_distribution_dof = config["t_distribution_dof"] 
data_generating_process = config['data_generating_process']
prediction_model = config['prediction_model']
VAR_spectral_radius = config['VAR_spectral_radius']
NIRVAR_spectral_radius = config['NIRVAR_spectral_radius']
heavy_tailed_errors = config['heavy_tailed_errors']
std_VAR_coefficients = config['std_VAR_coefficients']
VAR_burnin = config['VAR_burnin']
NIRVAR_p = config['NIRVAR_p']

rng = default_rng(SEED)

###### UTILITY FUNCTIONS ######
def random_stable_var(
        dim: int,
        num_lags: int,
        num_time_points: int,
        target_rho: float = 0.98,
        coeff_sd: float = 0.05,
        func_rng: np.random.Generator | None = None,
        burnin: int = 200
):
    """
    Draws a stable unrestricted VAR(num_lags) and simulates data.

    Parameters
    ----------
    dim : int
        Number of variables (k).
    num_lags : int
        VAR order (p).
    num_time_points : int
        Length of the simulated series (T).
    target_rho : float, default 0.98
        Desired spectral radius of the companion matrix (< 1 for stability).
    coeff_sd : float, default 0.05
        Standard deviation used for *all* coefficient draws (no lag decay).
    func_rng : numpy.random.Generator or None
        Random-number generator for reproducibility.

    Returns
    -------
    y : ndarray, shape (T, k)
        Simulated data.
    A : ndarray, shape (p, k, k)
        Coefficient matrices after rescaling.
    Sigma : ndarray, shape (k, k)
        Innovation covariance matrix used in the simulation.
    """

    # ----- 1. draw unrestricted coefficients ---------------------------------
    A = func_rng.normal(scale=coeff_sd, size=(num_lags, dim, dim))

    # ----- 2. rescale so companion matrix hits target spectral radius --------
    companion = np.vstack(
        [np.hstack(A), np.eye(dim * (num_lags - 1), dim * num_lags)]
    )
    rho0 = max(abs(eigvals(companion)))          # current spectral radius
    for _ in range(20):
        if rho0 < target_rho + 0.03:
            break
        else:
            A *= 0.8*target_rho / rho0
            companion = np.vstack([np.hstack(A),
                                   np.eye(dim * (num_lags - 1), dim * num_lags)])
            rho0 = max(abs(eigvals(companion)))
            print(rho0)
    A *= target_rho / rho0                       # bring it down/up to target

    # sanity-check
    new_rho = max(abs(eigvals(
        np.vstack([np.hstack(A), np.eye(dim * (num_lags - 1), dim * num_lags)])
    )))
    print(new_rho)
    assert new_rho <= target_rho + 0.03, "VAR not stable to requested threshold!"

    # ----- 3. innovations ----------------------------------------------------
    Sigma = func_rng.standard_normal((dim, dim))
    Sigma = Sigma @ Sigma.T                     # Wishart ⇒ SPD
    Sigma /= np.max(np.diag(Sigma))             # rescale variances ≈ 1

    # ----- 4. simulate -------------------------------------------------------
    total_T = num_time_points + burnin
    y = np.zeros((total_T, dim))

    for t in range(num_lags, total_T):
        deterministic = sum(A[j] @ y[t - j - 1] for j in range(num_lags))
        eps_t = func_rng.multivariate_normal(mean=np.zeros((dim)), cov=Sigma)    # Gaussian shock
        y[t] = deterministic + eps_t

    return y[burnin:], A, Sigma

def random_stable_var_t(
        dim: int,
        num_lags: int,
        num_time_points: int,
        df: float = 5,
        target_rho: float = 0.98,
        coeff_sd: float = 0.05,
        burnin: int = 250,
        func_rng: np.random.Generator | int | None = None,
):
    """
    Simulate an unrestricted VAR(num_lags) with multivariate Student-t errors
    drawn via scipy.stats.multivariate_t.

    Parameters
    ----------
    dim, num_lags, num_time_points : int
        Dimensions, VAR order, and length of the final sample (after burn-in).
    df : float, default 5
        Degrees of freedom ( > 2 for finite variance).
    target_rho : float, default 0.98
        Desired spectral radius for the rescaled VAR (< 1 ⇒ stable).
    coeff_sd : float, default 0.05
        Standard deviation for every raw coefficient draw (no lag decay).
    burnin : int, default 250
        Extra observations discarded to wash out initial conditions.
    func_rng : np.random.Generator | int | None
        Seed or Generator for reproducibility.

    Returns
    -------
    y      : ndarray  shape (num_time_points, dim)
        Simulated data after burn-in.
    A      : ndarray  shape (num_lags, dim, dim)
        VAR coefficient matrices (already rescaled to `target_rho`).
    Sigma  : ndarray  shape (dim, dim)
        Scale matrix used in `multivariate_t`.  
        (Covariance of εₜ is  df/(df-2) · Sigma  when df>2.)
    """
    func_rng = np.random.default_rng(func_rng)

    # 1. draw unrestricted coefficients (no variance shrinkage across lags)
    A = func_rng.normal(scale=coeff_sd, size=(num_lags, dim, dim))

    # 2. rescale so the companion matrix has spectral radius == target_rho
    companion = np.vstack([np.hstack(A),
                           np.eye(dim * (num_lags - 1), dim * num_lags)])
    rho0 = max(abs(eigvals(companion)))
    for _ in range(20):
        if rho0 < target_rho:
            break
        else:
            A *= 0.8*target_rho / rho0
            companion = np.vstack([np.hstack(A),
                                   np.eye(dim * (num_lags - 1), dim * num_lags)])
            rho0 = max(abs(eigvals(companion)))
            print(rho0)
    
    companion2 = np.vstack([np.hstack(A),
                           np.eye(dim * (num_lags - 1), dim * num_lags)])


    # 3. choose an SPD scale matrix for the Student-t shocks
    Sigma = func_rng.standard_normal((dim, dim))
    Sigma = Sigma @ Sigma.T
    Sigma /= np.max(np.diag(Sigma))          

    # prepare the SciPy multivariate-t object
    mvt = multivariate_t(loc=np.zeros(dim), shape=Sigma, df=df)

    # 4. manual VAR recursion with t-distributed innovations
    total_T = num_time_points + burnin
    y = np.zeros((total_T, dim))

    for t in range(num_lags, total_T):
        deterministic = sum(A[j] @ y[t - j - 1] for j in range(num_lags))
        eps_t = mvt.rvs(random_state=func_rng)    # Student-t shock
        y[t] = deterministic + eps_t

    return y[burnin:], A, Sigma

###### DATA GENERATION ######
if heavy_tailed_errors:
    if data_generating_process == "VARp":
        X_generated, _, _ = random_stable_var_t(
            dim=N,
            num_lags=VAR_p,
            num_time_points=T,
            target_rho=VAR_spectral_radius,
            coeff_sd=std_VAR_coefficients,
            df=t_distribution_dof,
            func_rng=rng
        )

    elif data_generating_process == "NIRVAR1": 
        generator = generativeVAR.generativeVAR(random_state=rng,
                                  T=T,
                                  N=N,
                                  B=K,
                                  Q=1,
                                  p_in = p_in,
                                  p_out = p_out,
                                  different_innovation_distributions=True,
                                  multiplier = NIRVAR_spectral_radius,
                                  t_distribution = True,
                                  t_dist_dof=t_distribution_dof
                                  )
        
        X_generated = generator.generate()[:,:,0] # shape (T, N) 

    elif data_generating_process == "NIRVARp": 
        Phi = np.zeros((VAR_p, N, N))
        for j in range(VAR_p):
            generator = generativeVAR.generativeVAR(random_state=rng,
                                    T=T,
                                    N=N,
                                    B=K,
                                    Q=1,
                                    p_in = p_in,
                                    p_out = p_out,
                                    different_innovation_distributions=False,
                                    multiplier = NIRVAR_spectral_radius,
                                    t_distribution = True,
                                    t_dist_dof=t_distribution_dof
                                    )
            Phi_j = generator.phi()[:,0,:,0] # shape (N, N)
            Phi[j] = Phi_j

        
        companion = np.vstack([np.hstack(Phi),np.eye(N * (NIRVAR_p - 1), NIRVAR_p * N)])
        rho0 = max(abs(eigvals(companion)))
        Phi  *= NIRVAR_spectral_radius / rho0 

        for _ in range(20):
            if rho0 < NIRVAR_spectral_radius + 0.03:
                break
            else:
                Phi *= 0.8*NIRVAR_spectral_radius / rho0
                companion = np.vstack([np.hstack(Phi),
                                    np.eye(N * (NIRVAR_p - 1), N * NIRVAR_p)])
                rho0 = max(abs(eigvals(companion)))
                print(rho0)
        Phi *= NIRVAR_spectral_radius / rho0   

        Sigma = rng.standard_normal((N,N))
        Sigma = Sigma @ Sigma.T                     # Wishart ⇒ SPD
        Sigma /= np.max(np.diag(Sigma))             # rescale variances ≈ 1

        total_T = T + VAR_burnin
        y = np.zeros((total_T, N))

        mvt = multivariate_t(loc=np.zeros(N), shape=Sigma, df=t_distribution_dof)

        for t in range(NIRVAR_p, total_T):
            deterministic = sum(Phi[j] @ y[t - j - 1] for j in range(NIRVAR_p))
            eps_t = mvt.rvs(random_state=rng)    # Student-t shock
            y[t] = deterministic + eps_t

        X_generated= y[VAR_burnin:]


else:
    if data_generating_process == "VARp":
        X_generated, _, _ = random_stable_var(
            dim=N,
            num_lags=VAR_p,
            num_time_points=T,
            target_rho=VAR_spectral_radius,
            coeff_sd=std_VAR_coefficients,
            func_rng=rng
        )

    elif data_generating_process == "NIRVAR1": 
        generator = generativeVAR.generativeVAR(random_state=rng,
                                  T=T,
                                  N=N,
                                  B=K,
                                  Q=1,
                                  p_in = p_in,
                                  p_out = p_out,
                                  different_innovation_distributions=False,
                                  multiplier = NIRVAR_spectral_radius
                                  )
        
        X_generated = generator.generate()[:,:,0] # shape (T, N) 

    elif data_generating_process == "NIRVARp": 
        Phi = np.zeros((NIRVAR_p, N, N))
        for j in range(NIRVAR_p):
            generator = generativeVAR.generativeVAR(random_state=rng,
                                    T=T,
                                    N=N,
                                    B=K,
                                    Q=1,
                                    p_in = p_in,
                                    p_out = p_out,
                                    different_innovation_distributions=False,
                                    multiplier = NIRVAR_spectral_radius
                                    )
            Phi_j = generator.phi()[:,0,:,0] # shape (N, N)
            Phi[j] = Phi_j

        
        companion = np.vstack([np.hstack(Phi),np.eye(N * (NIRVAR_p - 1), NIRVAR_p * N)])
        rho0 = max(abs(eigvals(companion)))
        Phi  *= NIRVAR_spectral_radius / rho0 

        for _ in range(20):
            if rho0 < NIRVAR_spectral_radius + 0.03:
                break
            else:
                Phi *= 0.8*NIRVAR_spectral_radius / rho0
                companion = np.vstack([np.hstack(Phi),
                                    np.eye(N * (NIRVAR_p - 1), N * NIRVAR_p)])
                rho0 = max(abs(eigvals(companion)))
                print(rho0)
        Phi *= NIRVAR_spectral_radius / rho0   

        Sigma = rng.standard_normal((N,N))
        Sigma = Sigma @ Sigma.T                     # Wishart ⇒ SPD
        Sigma /= np.max(np.diag(Sigma))             # rescale variances ≈ 1

        total_T = T + VAR_burnin
        y = np.zeros((total_T, N))

        for t in range(NIRVAR_p, total_T):
            deterministic = sum(Phi[j] @ y[t - j - 1] for j in range(NIRVAR_p))
            eps_t = rng.multivariate_normal(mean=np.zeros((N)), cov=Sigma)    # Gaussian shock
            y[t] = deterministic + eps_t

        X_generated= y[VAR_burnin:]


###### VISUALIZATION ######
import plotly.graph_objects as go

fig = go.Figure(
    data=[go.Scatter(y=X_generated[:, 0], mode='lines', line=dict(color='black'))],
    layout=go.Layout(
        yaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
        xaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )
)
fig.show()

###### BACKTESTING ###### 

