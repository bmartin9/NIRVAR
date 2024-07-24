""" 
Script to do backtesting. Outputs csv files containing predictions, estimated transition matrices and hyperparameters.txt 
NOTE: It is assumed that the backtest_design input file is clean: no NA values and has shape (T,N*Q) 
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


with open(sys.argv[2], "r") as f:
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
embedding_method = config['embedding_method']
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
phi_hat = np.zeros((n_backtest_days,N,N,2))
labels_hat = np.zeros((n_backtest_days,N))

for index in range(n_backtest_days):
    t = days_to_backtest[index]
    todays_date = first_prediction_day+t
    #get current embedding
    furthest_lookback_day = todays_date - lookback_window 
    X_train = Xs[furthest_lookback_day:todays_date,:,0:2] #Shape = (todays_date,N,Q) 
    current_embedding = train_model.Embedding(X_train,embedding_method=embedding_method) #If d is not specified, then it is computed via marchenko-pastur method
    d_hat = current_embedding.d
    if embedding_method == 'Pearson Correlation':
        current_corr = current_embedding.pearson_correlations()
    elif embedding_method == 'Full UASE Correlation':
        current_corr = current_embedding.pearson_correlations()
    elif embedding_method == 'Precision Matrix':
        current_corr = current_embedding.precision_matrix()
        for q in range(2): 
            diagonal_sqrt = np.sqrt(np.diag(current_corr[q]))
            current_corr[q] = current_corr[q] / np.outer(diagonal_sqrt, diagonal_sqrt) 
    elif embedding_method == 'Spearman Correlation':   
        current_corr = current_embedding.spearman_correlations()
    elif embedding_method == 'Kendall Correlation':
        current_corr = current_embedding.kendall_correlations()
    elif embedding_method == 'Covariance Matrix':
        current_corr = current_embedding.covariance_matrix()
    else:
        print("ERROR: Invalid embedding method")
        sys.exit()
 
    current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=SVD_niter,random_state=SVD_random_state)

    #get ols params and neighbours 
    trainer = train_model.fit(current_embedded_array[0:2,:,:],X_train[:,:,0:2],target_feature,UASE_dim=d_hat)
    if embedding_method == 'Full UASE Correlation':
        neighbours , labels = trainer.full_UASE_gmm(k=d_hat)
    else:
        neighbours , labels = trainer.gmm(k=d_hat)
    ols_params = trainer.ols_parameters(neighbours)
    phi_hat[index] = ols_params
    labels_hat[index] = labels[target_feature]

    #predict next day returns 
    todays_Xs = X_train[-1,:,0:2]
    predictor = predict_model.predict(ols_params,todays_Xs=todays_Xs)
    s = predictor.next_day_prediction()
    s_array[index] = s

#    print ("\033[A                             \033[A") 


###### OUTPUT s_array, labels_hat AND phi_hat TO FILE ###### 
phi_hat_path = f"phi_hat-{PBS_ARRAY_INDEX}.csv"
phi_hat = np.reshape(phi_hat,(n_backtest_days,N*N*2))

predictions_path = f"predictions-{PBS_ARRAY_INDEX}.csv"

labels_hat_path = f"labels_hat-{PBS_ARRAY_INDEX}.csv"

print(s_array[-1,-1])
np.savetxt(predictions_path, s_array, delimiter=',', fmt='%.6f')
np.savetxt(phi_hat_path, phi_hat, delimiter=',', fmt='%.6f')
np.savetxt(labels_hat_path, labels_hat, delimiter=',', fmt='%d')


###### OUTPUT BACKTESTING HYPERPARAMETERS TO FILE ######

f = open("backtesting_hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()
