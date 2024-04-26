# Test whether the sample points are from a MVN distribution 

#!/usr/bin/env python3
# USAGE: ./visualise.py hyperparameters.yaml  X_Sample_Covariance.csv 

import numpy as np
from numpy.random import default_rng
import sys
import yaml
from numpy import genfromtxt
import plotly.graph_objects as go
import time
from scipy.stats import norm
import pingouin as pg 


with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
N = config['N1']
T = config['T']
Q = config['Q']
B = config['B']
d = config['d']
sigma = config['sigma']
SEED = config['SEED']
N_replicas = config['N_replicas']
H = config['H'] 
spectral_radius = config['spectral_radius']
n_iter = config['n_iter']

random_state = default_rng(seed=SEED)

###### READ IN DATA ###### 
X_S = genfromtxt(sys.argv[2],delimiter=",")
X_S = np.reshape(X_S,(N_replicas,N,d)) 

component_to_visualise = 4

X_S_contour_data = X_S[:,component_to_visualise,:]

hz_results = pg.multivariate_normality(X_S_contour_data, alpha=.05)

print(hz_results)

