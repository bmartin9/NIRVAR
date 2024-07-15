""" 
Script to do backtesting on Santander bike station dataset using FARM.
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
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler


with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config['SEED']
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
num_lags = config['num_lags']
lookback_window = config['lookback_window'] 
num_factors = config['num_factors']

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
N = Xs.shape[1]

###### BACKTESTING ###### 
s_array = np.zeros((n_backtest_days,N)) #predictions 

for index in range(n_backtest_days):
    t = days_to_backtest[index]
    print(t)
    todays_date = first_prediction_day+t
    furthest_lookback_day = todays_date - lookback_window 
    X_train = Xs[furthest_lookback_day:todays_date+1,:] 
    # X_train_diff = X_train[7:] - X_train[:-7]  # First order differences to remove weekly seasonality
    X_train_diff = X_train[1:] - X_train[:-1]  # First order differences to remove weekly seasonality

    # Normalize the data to be in the range [-1,1] 
    # scaler = MinMaxScaler(feature_range=(-1,1)) 
    # scaler.fit(X_train.reshape((lookback_window+1,N))) 
    # X_train_scaled = scaler.transform(X_train.reshape((lookback_window+1,N))).reshape(lookback_window+1,N,1)
    # X_train_mean = np.mean(X_train_scaled,axis=0)
    # X_train_scaled -= X_train_mean 
    # X_train_scaled = X_train_scaled.reshape(lookback_window+1,N) 

    # Compute factor and idiosyncratic design matrices 
    evals, evecs = eigs(X_train_diff.T@X_train_diff, k = num_factors) 
    evals = evals.real 
    evecs = evecs.real
    loadings = np.multiply(np.sqrt(evals),evecs)
    factors = X_train_diff@loadings@np.linalg.inv(np.diag(evals)) 
    U = X_train_diff - factors@loadings.T 
    # Factor regression OLS
    factor_design = np.zeros(((lookback_window +0-num_lags),num_lags,num_factors))
    for tau in range(lookback_window +0-num_lags): 
        for f in range(num_factors):
            covariate_tau = factors[tau:tau+num_lags,f].flatten()
            factor_design[tau,:,f] = covariate_tau
    factor_targets = factors[num_lags:] 
    ols_reg_object = LinearRegression(fit_intercept=False)
    factor_coef = np.zeros((num_lags,num_factors))
    for f in range(num_factors):
        ols_fit = ols_reg_object.fit(factor_design[:,:,f],factor_targets[:,f]) 
        factor_coef[:,f] = ols_fit.coef_
    # U regression LASSO
    idiosyncratic_design = np.zeros((lookback_window +0-num_lags,N*num_lags))
    for l in range(lookback_window +0-num_lags): 
        covariate_l = U[l:l+num_lags].flatten() 
        idiosyncratic_design[l] = covariate_l 
    idiosyncratic_targets = U[num_lags:lookback_window +0,:] 
    # lasso_reg_object = MultiTaskLassoCV(alphas=[0.0001,0.0005,0.001,0.0015,0.002,0.005,0.01,0.015,0.02,0.05],fit_intercept=False)
    # lasso_reg_object = MultiTaskLassoCV(alphas=[0.005,0.01,0.02,0.05],fit_intercept=False)
    lasso_reg_object = Lasso(alpha = 0.01,fit_intercept=False)
    lasso_fit = lasso_reg_object.fit(idiosyncratic_design,idiosyncratic_targets) 
    # print(f"LASSO CV alpha: {lasso_fit.alpha_}")
    idio_coef = lasso_fit.coef_  
    print(idio_coef.shape) 
    print(np.sum(np.where(idio_coef!=0,1,0)))

    #predict next day returns 
    # todays_Xs = Xs[todays_date,:]
    # estimated_factor = factor_coef@factors[-num_lags:] 
    estimated_factor = factor_coef.T@factors[-num_lags:] 
    estimated_idio = idio_coef@U[lookback_window +0-num_lags:].flatten()
    s = np.sum(estimated_factor@loadings.T,axis=0) + estimated_idio 
    # s += X_train_mean[:,0]
    # s = scaler.inverse_transform(s.reshape(1,-1))
    # s += X_train[-1,:] #add back the weekly seasonality
    s_array[index] = s 

    # print ("\033[A                             \033[A") 
    # sys.stdout.write("\033[F")  # \033[F is the ANSI escape sequence to move up one line
    # sys.stdout.write("\033[K")  # \033[K is the ANSI escape sequence to clear the line from cursor to the end


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
