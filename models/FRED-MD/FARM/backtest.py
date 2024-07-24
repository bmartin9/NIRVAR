""" 
Predict the monthly Industrial Production using FARM.
NOTE: <DESIGN_MATRIX>.csv should be a transformed version of the FRED-MD dataset such that 
the time series are stationary.
"""

#!/usr/bin/env python3 
# USAGE: ./backtest.py <DESIGN_MATRIX>.csv backtesting_config.yaml 

import sys
import yaml
import numpy as np
from src.models import train_model
from src.models import predict_model
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
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols=range(1,123))
T = Xs.shape[0]
N = Xs.shape[1]

###### BACKTESTING ###### 
s_array = np.zeros((n_backtest_days,1)) #predictions 

for index in range(n_backtest_days):
    t = days_to_backtest[index]
    print(t)
    todays_date = first_prediction_day+t
    # print(todays_date)
    furthest_lookback_day = todays_date - lookback_window 
    X_train = Xs[furthest_lookback_day:todays_date,:] #Shape = (lookback_window,N) 
    # Normalize the data to be in the range [-1,1] 
    scaler = MinMaxScaler(feature_range=(-1,1)) 
    scaler.fit(X_train.reshape((lookback_window,N))) 
    # print(scaler.data_max_)
    X_train = scaler.transform(X_train.reshape((lookback_window,N)))
    X_train_mean = np.mean(X_train,axis=0)
    X_train -= X_train_mean

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
    idiosyncratic_targets = U[num_lags:lookback_window,5] 
    # lasso_reg_object = LassoCV(alphas=[0.0001,0.0005,0.001,0.0015,0.002,0.005,0.01,0.015,0.02,0.05],fit_intercept=False)
    lasso_reg_object = Lasso(alpha = 0.002,fit_intercept=False)
    lasso_fit = lasso_reg_object.fit(idiosyncratic_design,idiosyncratic_targets) 
    idio_coef = lasso_fit.coef_  
    print(idio_coef[idio_coef!=0].size)

    #predict next day returns 
    # todays_Xs = Xs[todays_date,:]
    estimated_factor = factor_coef@factors[-num_lags:] 
    estimated_idio = idio_coef.T@U[lookback_window-num_lags:].flatten()
    s = estimated_factor*loadings[5] + estimated_idio 
    s += X_train_mean[5]
    inv_transform_vec = np.zeros((N)) 
    inv_transform_vec[5] = s 
    s = scaler.inverse_transform(inv_transform_vec.reshape(1,-1)) 
    s_array[index] = s[0,5]

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
