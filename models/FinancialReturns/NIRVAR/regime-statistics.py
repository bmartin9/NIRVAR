""" 
Script to compute various backtesting statistics given an input file of predicted stock returns values
for different dates within the backtesting period 
NOTE: It is assumed that the backtest_design input file is clean: no NA values and has shape (T,N*Q) 
"""

#!/usr/bin/env python3 
# USAGE: ./backtest_statistics.py <BACKTEST_DESIGN>.csv predictions.csv backtesting_config.yaml 

import sys 
import yaml 
import numpy as np 
from src.models import predict_model
from numpy.random import default_rng 
from scipy import stats 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

with open(sys.argv[3], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 

###### CONFIG PARAMETERS ###### 
SEED = config['SEED'] 
Q = config['Q']
n_backtest_days_tot = config['n_backtest_days'] 
first_prediction_day = config['first_prediction_day']
target_feature = config['target_feature']
SVD_niter = config['SVD_niter'] 
SVD_random_state = config['SVD_random_state']
quantile = config['quantile'] #The top quantile stocks with the strongest predictions 
target_feature = config['target_feature']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',') #read in full design matrix 
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N) 
print(f"N : {N}")
print(f"T : {T}")

Xs = np.reshape(Xs,(T,N,Q),order='F') 

predictions = np.genfromtxt(sys.argv[2], delimiter=',') #read in predictions. shape = (n_backtest_days_tot,N) 

first_fret = first_prediction_day   # first day we predict for
last_fret = first_fret + n_backtest_days_tot # last day we predict for
targets = Xs[first_fret:last_fret,:,target_feature] 
print(targets[0,0])

####### DAILY BACKTESTING STATISTICS ###### 
num_splits = 5 
regime_sharpes = np.zeros((num_splits))
for s in range(num_splits):
    regime_backest_days = int(n_backtest_days_tot/num_splits)
    PnL = np.zeros((regime_backest_days))
    for t in range(s*regime_backest_days,(s+1)*regime_backest_days): 
        weightings = np.ones((N)) #equal weightings
        daily_bench = predict_model.benchmarking(predictions=predictions[t],market_excess_returns=targets[t],yesterdays_predictions=predictions[t-1])  
        daily_PnL = daily_bench.weighted_PnL_transactions(weights=weightings, quantile=quantile) 
        PnL[t - s*regime_backest_days] = daily_PnL 

    sharpe_ratio = (np.mean(PnL)/np.std(PnL))*np.sqrt(252)
    regime_sharpes[s] = sharpe_ratio 

summary_statistics = {'Regime Sharpes': regime_sharpes}
f = open("regime_sharpes.txt", "w")
f.write("{\n")
for k in summary_statistics.keys():
    f.write("'{}':'{}'\n".format(k, summary_statistics[k]))
f.write("}")
f.close()