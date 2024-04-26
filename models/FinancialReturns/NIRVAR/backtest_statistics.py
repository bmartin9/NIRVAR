""" 
Script to compute various backtesting statistics given an input file of predicted stock returns values
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
PnL = np.zeros((n_backtest_days_tot))
hit_ratios  = np.zeros((n_backtest_days_tot))
long_ratios = np.zeros((n_backtest_days_tot)) 
corr_SP = np.zeros((n_backtest_days_tot))
daily_rmse = np.zeros((n_backtest_days_tot))
daily_turnover = np.zeros((n_backtest_days_tot))
daily_r_squared = np.zeros((n_backtest_days_tot))

volume_array = Xs[:,:,2]


for t in range(n_backtest_days_tot): 
    print(t) 
    #Compute Portfolio weights 

    weightings = np.ones((N)) #equal weightings
    
    daily_bench = predict_model.benchmarking(predictions=predictions[t],market_excess_returns=targets[t],yesterdays_predictions=predictions[t-1])  
    daily_PnL = daily_bench.weighted_PnL_transactions(weights=weightings, quantile=quantile) 
    PnL[t] = daily_PnL
    daily_hit_ratio = daily_bench.hit_ratio()
    hit_ratios[t] = daily_hit_ratio
    daily_long = daily_bench.long_ratio() 
    long_ratios[t] = daily_long
    daily_corr_SP = daily_bench.corr_SP()
    corr_SP[t] = daily_corr_SP
    daily_rmse[t] = np.sqrt(mean_squared_error(predictions[t],targets[t]))
    daily_turnover[t] = (1/N)*np.sum(daily_bench.transaction_indicator())
    daily_r_squared = r2_score(y_true=targets[t],y_pred=predictions[t]) 

    print ("\033[A                             \033[A") 


def deflated_SR(PnL_vector): 
    SR = np.mean(PnL_vector)/np.std(PnL_vector) 
    Ti = len(PnL_vector) 
    g3 = stats.skew(PnL_vector) 
    g4 = stats.kurtosis(PnL_vector) 
    denominator = 1 - g3*SR + (g4-1)*((SR**2)/4)
    test_statistic = SR/(np.sqrt(denominator/(Ti-1))) 
    p1 = stats.norm.cdf(test_statistic) 
    pval = np.minimum(p1, 1 - p1)*2 
    return pval 

deflated_sharpe_ratio = deflated_SR(PnL) 

sharpe_ratio = (np.mean(PnL)/np.std(PnL))*np.sqrt(252)
mean_spearman_corr = np.mean(corr_SP)
print(f"sum PnL : {np.sum(PnL)}") 
PPT = np.sum(PnL)/(n_backtest_days_tot*N)
print(f"n_backtest_days_tot : {n_backtest_days_tot}") 
mean_daily_PnL = np.mean(PnL) 
total_hit_ratio = np.mean(hit_ratios)
total_long_ratio = np.mean(long_ratios)
mean_rmse = np.mean(daily_rmse)
mean_turnover = np.mean(daily_turnover)
mean_r_squared = np.mean(daily_r_squared)
MAE = mean_absolute_error(y_true=targets,y_pred=predictions,multioutput="uniform_average")

# Calculate Max draw down
cum_PnL_bpts = 10000*np.cumsum(PnL)/(n_backtest_days_tot) 
def max_drawdown(X):
    #return max draw down in percentage terms
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd 
max_dd = max_drawdown(cum_PnL_bpts)

#Sotino Ratio
sortino_ratio = (np.mean(PnL)/np.std(np.where(PnL<=0,PnL,0)))*np.sqrt(252)

###### WRITE TO OUTPUT FILES ######
np.savetxt('PnL.csv', PnL, delimiter=',', fmt='%.6f')
np.savetxt('hit.csv', hit_ratios, delimiter=',', fmt='%.6f')
np.savetxt('long.csv', PnL, delimiter=',', fmt='%.6f')
np.savetxt('spearman_corr.csv', corr_SP, delimiter=',', fmt='%.6f')

summary_statistics = {'Deflated Sharpe Ratio': deflated_sharpe_ratio,
                      'Sharpe Ratio': sharpe_ratio,
                      'Mean Spearman Correlation' : mean_spearman_corr,
                      'PnL Per Trade' : PPT,
                      'Mean Daily PnL' : mean_daily_PnL,
                      'Hit Ratio' : total_hit_ratio,
                      'Long Ratio' : total_long_ratio,
                      'Max Draw Down' : max_dd,
                      'Sortino Ratio' : sortino_ratio,
                      'Mean RMSE' : mean_rmse,
                      'Mean R Squared' : mean_r_squared,
                      'Mean Turnover' : mean_turnover,
                      'MAE' : MAE} 

f = open("summary_statistics.txt", "w")
f.write("{\n")
for k in summary_statistics.keys():
    f.write("'{}':'{}'\n".format(k, summary_statistics[k]))
f.write("}")
f.close()
 
