""" 
Script to clean universe of stocks data and subtract the market. 
"""

#!/usr/bin/env python3 
# USAGE: ./clean_stocks.py </PATH/TO/STOCKS_DATA>[0]-[3].csv 

import numpy as np 
import pandas as pd
import sys 

OPCL_path = sys.argv[1] # e.g. ../../data/processed/Matrix_Format_SubsetUniverse/OPCL_20000103_20201231.csv'
pvCLCL_path = sys.argv[2]
volMM_path = sys.argv[3]
volume_path = sys.argv[4] 

###### Import data from ../data/processed/*.csv ###### 
opcl_df = pd.read_csv(OPCL_path)
pvclcl_df = pd.read_csv(pvCLCL_path) 
volMM_df = pd.read_csv(volMM_path) 
volume_df = pd.read_csv(volume_path)

#Convert data frames to numpy arrays - not including ticker
opcl_array = opcl_df.iloc[:,1:].to_numpy(copy=True)
pvclcl_array = pvclcl_df.iloc[:,1:].to_numpy(copy=True) 
volMM_array = volMM_df.iloc[:,1:].to_numpy(copy=True) 
volume_array = volume_df.iloc[:,1:].to_numpy(copy=True) 

# Get indices of rows that contain nan values
opcl_nan_rows = np.argwhere(np.isnan(opcl_array).any(axis=1))
pvclcl_nan_rows = np.argwhere(np.isnan(pvclcl_array).any(axis=1))
rows_to_delete = np.concatenate((opcl_nan_rows,pvclcl_nan_rows),axis=None) 
# np.savetxt("rows_to_delete.csv",rows_to_delete,delimiter=",",fmt='%d')

###### PARAMETERS OF DATASET ######
N_total = opcl_array.shape[0]
Q = 4
T = opcl_array.shape[1]

# Create data array with 4 features
tickers = opcl_df.iloc[:,0].to_numpy()
X = np.zeros((T,N_total,Q))
X[:,:,0] = opcl_array.T
X[:,:,1] = pvclcl_array.T
X[:,:,2] = volMM_array.T
X[:,:,3] = volume_array.T
X = np.delete(X,rows_to_delete,axis=1) # delete rows containing nan values 
new_tickers = np.delete(tickers,rows_to_delete)
SPY_index = np.argwhere(new_tickers=='SPY')[0,0] 

# Number of stocks with no 'NA' values 
N = X.shape[1]

# remove the market 
X_SPY = X[:,SPY_index,:] 
X_no_SPY = np.delete(X[:,:,0:4],SPY_index,axis=1) 
X_no_SPY[:,:,0:2] = X_no_SPY[:,:,0:2] - X_SPY[:,None,0:2] 
# X_no_SPY = np.delete(X[:,:,0:2],SPY_index,axis=1)  - (X[:,SPY_index,0:2])[:,None,:]

# X_no_SPY = X_no_SPY[:,:,:]
X_no_SPY_reshaped = np.reshape(X_no_SPY,(T,(N-1)*Q),order='F')  

file_name = "stocks_no_market_cleaned.csv"

# Save the numpy array to the CSV file
np.savetxt(file_name, X_no_SPY_reshaped, delimiter=',', fmt='%.6f', header='', comments='')

