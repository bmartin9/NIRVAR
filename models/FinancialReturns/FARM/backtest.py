""" 
Do FARM returns prediction using my implementation of the FARM model.
"""

#!/usr/bin/env python3 
# USAGE: ./backtest.py <DESIGN_MATRIX>.csv backtesting_config.yaml 

import sys
import yaml
import numpy as np
from numpy.random import default_rng
import fcntl
import os 
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import eigs
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config['SEED']
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
num_lags = config['num_lags']
target_feature = config['target_feature']
Q = config['Q']
lookback_window = config['lookback_window']

###### ENVIRONMENT VARIABLES ###### 
# PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
PBS_ARRAY_INDEX = 1
# NUM_ARRAY_INDICES = int(os.environ['NUM_ARRAY_INDICES'])
NUM_ARRAY_INDICES = 1

# Re-define n_backtest_days to be total number of backtesting days divided by the number of array indices 
n_backtest_days = int(n_backtest_days_tot/NUM_ARRAY_INDICES)

# Get a list of days to do backtesting on
days_to_backtest = [int(i + (n_backtest_days)*(PBS_ARRAY_INDEX-1)) for i in range(n_backtest_days)]

random_state = default_rng(seed=SEED)

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N)

Xs = np.reshape(Xs,(T,N,Q),order='F') 

###### BACKTESTING ###### 
s_array = np.zeros((n_backtest_days,N)) #predictions 

for index in range(n_backtest_days):
    t = days_to_backtest[index]
    print(t)
    todays_date = first_prediction_day+t
    furthest_lookback_day = todays_date - lookback_window
    X_train = Xs[furthest_lookback_day:todays_date,:,target_feature] #Shape = (lookback_window,N)

    # Compute factor and idiosyncratic design matrices 
    evals, evecs = eigs(X_train.T@X_train, k = 1) 
    evals = evals.real 
    evecs = evecs.real
    loadings = np.multiply(np.sqrt(evals),evecs)
    factors = X_train@loadings@np.linalg.inv(np.diag(evals)) 
    U = X_train - factors@loadings.T 
    # Factor regression OLS
    factor_design = np.zeros(((lookback_window-num_lags),num_lags))
    for tau in range(lookback_window-num_lags): 
        covariate_tau = factors[tau:tau+num_lags].flatten()
        factor_design[tau] = covariate_tau
    factor_targets = factors[num_lags:T] 
    ols_reg_object = LinearRegression(fit_intercept=False)
    ols_fit = ols_reg_object.fit(factor_design,factor_targets) 
    factor_coef = ols_fit.coef_  
    # U regression LASSO
    idiosyncratic_design = np.zeros((lookback_window-num_lags,N*num_lags))
    for l in range(lookback_window-num_lags): 
        covariate_l = U[l:l+num_lags].flatten() 
        idiosyncratic_design[l] = covariate_l 
    idiosyncratic_targets = U[num_lags:lookback_window,:] 
    # lasso_reg_object = LassoCV(alphas=[0.0001,0.0005,0.001,0.0015,0.002,0.005,0.01,0.015,0.02,0.05],fit_intercept=False)
    lasso_reg_object = Lasso(alpha = 0.00005,fit_intercept=False)
    lasso_fit = lasso_reg_object.fit(idiosyncratic_design,idiosyncratic_targets) 
    idio_coef = lasso_fit.coef_  
    print(idio_coef[idio_coef!=0].size)

    #predict next day returns 
    estimated_factor = factor_coef@factors[-num_lags:] 
    estimated_idio = idio_coef.T@U[lookback_window-num_lags:].flatten()
    s = estimated_factor*loadings + estimated_idio[:,None]
    s_array[index] = s[:,0]

    print ("\033[A                             \033[A") 
    sys.stdout.write("\033[F")  # \033[F is the ANSI escape sequence to move up one line
    sys.stdout.write("\033[K")  # \033[K is the ANSI escape sequence to clear the line from cursor to the end


###### OUTPUT s_array, labels_hat AND phi_hat TO FILE ###### 
predictions_path = f"predictions-{PBS_ARRAY_INDEX}.csv"


np.savetxt(predictions_path, s_array, delimiter=',', fmt='%.6f')

###### OUTPUT BACKTESTING HYPERPARAMETERS TO FILE ######

f = open("backtesting_hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()