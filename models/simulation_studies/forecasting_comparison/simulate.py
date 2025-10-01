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
from numpy.linalg import eigvals
from scipy.stats import multivariate_t 
from typing import List, Tuple, Optional, Union
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV


with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
N_list = config["N_list"] 
T = config["T"] 
K = config["K"]
p_in = config["p_in"]
p_out_list = config["p_out_list"]
VAR_p = config["VAR_p"] 
lasso_penalty = config["lasso_penalty"] 
num_backtest_days = config["num_backtest_days"] 
first_prediction_day = config["first_prediction_day"]
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
num_replicas = config['num_replicas']
estimated_VAR_p = config['estimated_VAR_p'] 

num_N = len(N_list)
num_pout = len(p_out_list)

###### UTILITY FUNCTIONS ######
def random_stable_var(
        dim: int,
        num_lags: int,
        num_time_points: int,
        target_rho: float = 0.9,
        coeff_sd: float = 0.05,
        func_rng: np.random.Generator | None = None,
        burnin: int = 200,
        phi_distribution: str = "Gaussian"  # "Gaussian" or "Wishart"
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
    target_rho : float, default 0.9
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
    if phi_distribution == "Gaussian":
        A = func_rng.normal(scale=coeff_sd, size=(num_lags, dim, dim))
    elif phi_distribution == "Wishart":
        A = np.empty((num_lags, dim, dim), dtype=float)
        for j in range(num_lags):
            A_half = rng.normal(size=(N, N)) 
            A_Wishart = A_half @ A_half.T                # Wishart ⇒ PSD
            A[j] = (A_Wishart/np.max(np.diag(A_Wishart)))  
        

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
        Phi_half = rng.normal(size=(N, N)) 
        Phi_Wishart = Phi_half @ Phi_half.T                # Wishart ⇒ SPD
        Phi[j] = (Phi_Wishart) * mask 

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

    

###### FUNCTIONS TO ESTIMATE MODELS ######
def estimate_var_no_intercept(data: np.ndarray, p: int
                              ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    VAR(p) estimation WITHOUT an intercept and 1-step-ahead prediction.

    Parameters
    ----------
    data : ndarray, shape (T, N)
        Observed time-series (rows = time, columns = variables).
    p : int
        Lag order of the VAR.

    Returns
    -------
    A_list : list[ndarray]         # length p, each (N, N)
        Coefficient matrices A₁ … A_p such that
            y_t = A₁ y_{t−1} + … + A_p y_{t−p} + u_t .
    y_hat_next : ndarray, shape (N,)
        Conditional mean E[y_T | y_{T−1}, … , y_{T−p}].
    """
    if p < 1:
        raise ValueError("p must be ≥ 1.")
    T, N = data.shape
    if T <= p:
        raise ValueError("Need at least p+1 observations.")

    # ----------  build regressors and dependent variable  ------------------
    Y = data[p:]                                    # (T-p) × N
    X = np.empty((T - p, N * p))                    # (T-p) × (N p)
    for lag in range(1, p + 1):
        X[:, (lag - 1) * N : lag * N] = data[p - lag : T - lag]

    # ----------  OLS estimate (no intercept)  ------------------------------
    # beta_hat: (N p) × N   with stacked [A₁', …, A_p']'
    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]

    # unpack into A₁ … A_p
    A_list = [
        beta_hat[(i * N) : (i + 1) * N].T   # (N, N)
        for i in range(p)
    ]

    # ----------  1-step-ahead forecast -------------------------------------
    y_lags = data[-p:][::-1]               # y_{T−1}, y_{T−2}, … , y_{T−p}
    y_hat_next = sum(A @ y for A, y in zip(A_list, y_lags))

    return A_list, y_hat_next

RandomStateLike = Union[int, np.random.RandomState, np.random.Generator, None]

def estimate_var_lasso(
    data: np.ndarray,
    p: int,
    *,
    alpha: Optional[float] = None,
    cv: bool = True,
    cv_folds: int = 5,
    max_iter: int = 1000,
    random_state: RandomStateLike = None,
) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """
    LASSO-penalised VAR(p) without intercept + 1-step forecast.

    Parameters
    ----------
    data : (T, N) array
    p    : lag order
    alpha: penalty (None → chosen by CV)
    cv   : if True and alpha is None, run MultiTaskLassoCV
    random_state : int | RandomState | Generator | None
        Forwarded directly to scikit-learn.

    Returns
    -------
    A_list     : list of A₁ … A_p  (each (N, N))
    y_hat_next : 1-step forecast   (N,)
    alpha_used : penalty actually employed
    """
    if p < 1:
        raise ValueError("p must be ≥ 1")
    T, N = data.shape
    if T <= p:
        raise ValueError("Need at least p+1 observations")

    # ---------- build design matrices --------------------------------------
    Y = data[p:]                                        # (T-p, N)
    X = np.empty((T - p, N * p))                        # (T-p, Np)
    for lag in range(1, p + 1):
        X[:, (lag - 1) * N : lag * N] = data[p - lag : T - lag]

    # ---------- fit LASSO ---------------------------------------------------
    if alpha is None and not cv:
        raise ValueError("Supply `alpha` or set `cv=True` for CV selection")

    if alpha is None:
        model = MultiTaskLassoCV(
            cv=cv_folds,
            fit_intercept=False,
            max_iter=max_iter,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X, Y)
        alpha_used = model.alpha_
    else:
        model = MultiTaskLasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(X, Y)
        alpha_used = alpha

    # ---------- unpack coefficients ----------------------------------------
    beta_hat = model.coef_.T                           # (Np, N)

    A_list: List[np.ndarray] = [
        beta_hat[i * N : (i + 1) * N].T               # (N, N)
        for i in range(p)
    ]

    # ---------- 1-step forecast --------------------------------------------
    y_lags = data[-p:][::-1]                           # y_{T-1}, … , y_{T-p}
    y_hat_next = sum(A @ y for A, y in zip(A_list, y_lags))

    return A_list, y_hat_next, alpha_used

mspe_values = np.zeros((num_N,num_pout,num_replicas))
std_values = np.zeros((num_N,num_pout,num_replicas)) 

rng = default_rng(SEED)

for i, N in enumerate(N_list):
    for j, p_out in enumerate(p_out_list):
        for replica in range(num_replicas):
            # rng = default_rng(SEED)

            print(f"Running simulation with N={N}, p_out={p_out}, replica={replica+1}/{num_replicas}")

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

                elif data_generating_process == "NIRVARp": 
                    Phi = generate_NIRVAR_coefficients(
                        N=N,
                        K=K,
                        p_in=p_in,
                        p_out=p_out,
                        num_lags=1,
                        NIRVAR_spectral_radius=NIRVAR_spectral_radius,
                        rng=rng
                    )
                    
                    X_generated = generate_NIRVAR_data(
                        T=T,
                        N=N,
                        Phi=Phi,
                        num_lags=NIRVAR_p,
                        target_rho=NIRVAR_spectral_radius,
                        burnin=VAR_burnin,
                        Sigma_type="Wishart",
                        rng=rng,
                        heavy_tailed_errors=True,
                        t_distribution_dof=t_distribution_dof
                    )


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

                elif data_generating_process == "NIRVARp": 

                    Phi = generate_NIRVAR_coefficients(
                        N=N,
                        K=K,
                        p_in=p_in,
                        p_out=p_out,
                        num_lags=NIRVAR_p,
                        NIRVAR_spectral_radius=NIRVAR_spectral_radius,
                        rng=rng
                    )
                    
                    X_generated = generate_NIRVAR_data(
                        T=T,
                        N=N,
                        Phi=Phi,
                        num_lags=1,
                        target_rho=NIRVAR_spectral_radius,
                        burnin=VAR_burnin,
                        Sigma_type="Wishart",
                        rng=rng
                    )


            ###### VISUALIZATION ######
            # import plotly.graph_objects as go

            # fig = go.Figure(
            #     data=[go.Scatter(y=X_generated[:, 0], mode='lines', line=dict(color='black'))],
            #     layout=go.Layout(
            #         yaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
            #         xaxis=dict(showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
            #         paper_bgcolor='white',
            #         plot_bgcolor='white',
            #         font_family="Serif",
            #         font_size=11,
            #         margin=dict(l=5, r=5, t=5, b=5),
            #         width=500,
            #         height=350
            #     )
            # )
            # fig.show()


            ###### BACKTESTING ###### 
            # Get a list of days to do backtesting on
            days_to_backtest = [int(first_prediction_day + i) for i in range(num_backtest_days)]
            # print(f"Days to backtest: {days_to_backtest}")

            if prediction_model == "VARp":
                predictions = np.zeros((num_backtest_days,N)) 
                for t, day in enumerate(days_to_backtest):
                    X_train = X_generated[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
                    A_list, y_hat_next = estimate_var_no_intercept(X_train, estimated_VAR_p) 
                    predictions[t] = y_hat_next
                

            elif prediction_model == "VARp_LASSO":
                predictions = np.zeros((num_backtest_days,N)) 
                for t, day in enumerate(days_to_backtest):
                    # print(t)
                    alpha_cv = 0.32
                    if t==-1:
                        X_train = X_generated[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
                        A_list, y_hat_next, alpha_used = estimate_var_lasso(X_train, VAR_p, cv=True, cv_folds=5, random_state=4266) 
                        print(alpha_used)
                        alpha_cv += alpha_used
                        predictions[t] = y_hat_next
                    else:
                        print(t)
                        X_train = X_generated[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
                        A_list, y_hat_next, alpha_used = estimate_var_lasso(X_train, VAR_p, cv=False, alpha=alpha_cv, random_state=4266) 
                        predictions[t] = y_hat_next

            elif prediction_model == "NIRVAR1":
                predictions = np.zeros((num_backtest_days,N)) 
                for t, day in enumerate(days_to_backtest):
                    X_train = X_generated[day-lookback_window:day+1, :, np.newaxis] # day is the day on which you predict tomorrow's returns from 
                    NIRVAR_embedding = train_model.Embedding(y=X_train,
                                                            d=K,
                                                            embedding_method='Pearson Correlation',
                                                            cutoff_feature=0)
                    d_hat = NIRVAR_embedding.d 
                    correlation_matrix = NIRVAR_embedding.covariance_matrix()
                    embedded_array = NIRVAR_embedding.embed_corr_matrix(corr_matrix=correlation_matrix,n_iter=20,random_state=345)
                    fit_NIRVAR = train_model.fit(embedded_array=embedded_array,
                                                training_set=X_train,
                                                target_feature=0,
                                                UASE_dim=d_hat)
                    gmm_groups  = fit_NIRVAR.gmm(k=d_hat)[0]
                    ols_params = fit_NIRVAR.ols_parameters(constrained_array=gmm_groups)[:,:,0]
                    y_hat_next = ols_params@X_train[-1,:,0].T
                    predictions[t] = y_hat_next
                

            elif prediction_model == "BayesianVAR": 
                import pandas as pd
                import rpy2.robjects as ro
                from rpy2.robjects.packages import importr
                from rpy2.robjects import pandas2ri, conversion, default_converter
                from rpy2.robjects.conversion import localconverter

                
                BVAR = importr("BVAR")                       

                # ------------------------------------------------------------------
                # 1.  Helper: fit BVAR & get forecast for a single training window --
                # ------------------------------------------------------------------
                def _bvar_one_step(x_train: np.ndarray, lags: int) -> np.ndarray:
                    """
                    Fit a Minnesota-prior BVAR on x_train (T×N) and return the
                    posterior-mean 1-step-ahead forecast as a NumPy vector (N,).
                    """
                    df_py = pd.DataFrame(
                        x_train,
                        columns=[f"y{i+1}" for i in range(x_train.shape[1])]
                    )

                    # ---- Python → R -------------------------------------------
                    with localconverter(default_converter + pandas2ri.converter):
                        ro.globalenv["Y_train"] = conversion.py2rpy(df_py)

                    ro.globalenv["p_lags"] = lags

                    # ---- R code -----------------------------------------------
                    ro.r("""
                        library(BVAR)
                        minnesota_prior <- bv_priors(hyper = "auto",mn=bv_mn(lambda = bv_lambda(mode = 0.2, sd = 0.4, min = 0.0001, max = 5)))
                        fit_bvar  <- bvar(Y_train, lags = p_lags, n_draw = 1000, n_burn = 500, priors = minnesota_prior
                                        )
                        pred_bvar <- predict(fit_bvar, horizon = 1)$fcast
                        fc_mean <- apply(pred_bvar, 3, mean)  
                    """)

                    # ---- R → Python -------------------------------------------
                    with localconverter(default_converter + pandas2ri.converter):
                        fc_py = conversion.rpy2py(ro.r("fc_mean"))

                    return np.asarray(fc_py, dtype=float)    # shape (N)
                
                predictions = np.zeros((num_backtest_days, N))

                for t, day in enumerate(days_to_backtest):
                    X_train = X_generated[day - lookback_window : day + 1, :]
                    predictions[t] = _bvar_one_step(X_train, lags=estimated_VAR_p)

            ###### EVALUATION ######
            targets = X_generated[[j+1 for j in days_to_backtest], :] # tomorrow's returns
            mspe = np.sum((targets - predictions)**2)/((N*num_backtest_days))
            print(f"MSPE : {mspe}")
            mspe_values[i,j,replica] = mspe 
            std = np.std(targets - predictions)
            print(f"STD : {std}")
            std_values[i,j,replica] = std 

            # x = np.arange(1, len(targets[:,0]) + 1)

            # Create a figure and add traces for each array
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(x=x, y=targets[:,0], mode='lines', name='Targets'))
            # fig.add_trace(go.Scatter(x=x, y=predictions[:,0], mode='lines', name='Predictions'))
            # fig.show()
if num_replicas>1:
    mean_mspe = np.mean(mspe_values, axis=-1)
    std_mspe = np.std(std_values, axis=-1)
else:
    mean_mspe = mspe_values[:,:,0]
    std_mspe = std_values[:,:,0]

np.savetxt(f"mspe_mean_{data_generating_process}_{prediction_model}.csv", mean_mspe.T, delimiter=",", fmt="%.6f")
np.savetxt(f"mspe_std_{data_generating_process}_{prediction_model}.csv", std_mspe.T, delimiter=",", fmt="%.6f") 



