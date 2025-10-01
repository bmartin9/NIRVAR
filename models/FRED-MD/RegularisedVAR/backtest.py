""" 
Predict the monthly Industrial Production using VAR(p).
NOTE: <DESIGN_MATRIX>.csv should be a transformed version of the FRED-MD dataset such that 
the time series are stationary.
"""

#!/usr/bin/env python3 
# USAGE: ./backtest.py <DESIGN_MATRIX>.csv backtesting_config.yaml 

import sys
import yaml
import numpy as np
from src.models import predict_model
from numpy.random import default_rng
import fcntl
import os 
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso

with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if len(sys.argv) > 3 : 
    manual_d = sys.argv[3] 
    specified_d = int(sys.argv[4])
else: 
    manual_d = False

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
embedding_method = config['embedding_method']
lookback_window = config['lookback_window']
p = config['p'] # VAR(p) lag order
alpha_grid = config['alpha_grid']

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
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N)

Xs = np.reshape(Xs,(T,N,Q),order='F')

###### BACKTESTING ###### 
s_array = np.zeros((n_backtest_days,1)) #predictions 

for index in range(n_backtest_days):
    t = days_to_backtest[index]
    print(t)
    todays_date = first_prediction_day+t
    # print(todays_date)
    #get current embedding
    furthest_lookback_day = todays_date - lookback_window 
    X_train = Xs[furthest_lookback_day:todays_date,:,:] #Shape = (todays_date,N,Q) 
    # Normalize the data to be in the range [-1,1] 
    scaler = MinMaxScaler(feature_range=(-1,1)) 
    scaler.fit(X_train.reshape((lookback_window,N))) 
    # print(scaler.data_max_)
    X_train = scaler.transform(X_train.reshape((lookback_window,N))).reshape(lookback_window,N,1)
    X_train_mean = np.mean(X_train,axis=0)
    X_train -= X_train_mean
    X_train = X_train.reshape(lookback_window,N_times_Q)
    Y = X_train[p:]
    Z = np.concatenate([X_train[p-k:-k] for k in range(1, p+1)], axis=1)
    # mtl = MultiTaskLassoCV(alphas=alpha_grid, cv=5, fit_intercept=False,random_state=543, n_jobs=None)
    mtl = MultiTaskLasso(alpha=alpha_grid[0],  fit_intercept=False,random_state=543)
    mtl.fit(Z,Y) 
    # print(mtl.alpha_)
    print(1 - np.count_nonzero(mtl.coef_)/(N*N*p))
    # ols_params = VAR_fit.coefs
    
    #predict next day returns 
    todays_Xs = X_train[-p:,:]
    Z_next = np.concatenate([todays_Xs[-k] for k in range(1, p+1)], axis=0).reshape(1, -1)
    s = Z_next @ mtl.coef_.T 
    # todays_Xs = scaler.transform(todays_Xs.reshape(1,N)).reshape(N,1) # Make sure to scale your targets!
    # todays_Xs -= X_train_mean
    s += X_train_mean[:,0]
    s = scaler.inverse_transform(s.reshape(1,-1))
    s_array[index] = s[:,5]

    # print ("\033[A                             \033[A") 
    # sys.stdout.write("\033[F")  # \033[F is the ANSI escape sequence to move up one line
    # sys.stdout.write("\033[K")  # \033[K is the ANSI escape sequence to clear the line from cursor to the end


###### OUTPUT s_array, labels_hat AND phi_hat TO FILE ###### 
predictions_path = f"predictions-{PBS_ARRAY_INDEX}.csv"


print(s_array[-1,-1])
np.savetxt(predictions_path, s_array, delimiter=',', fmt='%.6f')


###### OUTPUT BACKTESTING HYPERPARAMETERS TO FILE ######

f = open("backtesting_hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()
