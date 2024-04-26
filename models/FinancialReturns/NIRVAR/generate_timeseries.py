""" 
Script to generate time series from a SBM VAR model.
"""

#!/usr/bin/env python3 
# USAGE: ./generate_timeseries.py <CONFIG>.yaml 

import sys 
import yaml 
import numpy as np 
from src.models import generativeVAR
from numpy.random import default_rng

with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 

###### CONFIG PARAMETERS ###### 
SEED = config['SEED'] 
N = config['N']
T = config['T']
Q = config['Q']
B = config['B']
spectral_radius = config['spectral_radius']
p_in = config['p_in']
p_out = config['p_out'] 

random_state = default_rng(seed=SEED) 

generator = generativeVAR.generativeVAR(random_state,
                                        T=T,
                                        N = N,
                                        Q = Q,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        multiplier=spectral_radius,
                                        different_innovation_distributions=False
                                        )

Xs = generator.generate()
Xs = np.reshape(Xs,(T,N*Q),order = 'F')
phi_ground_truth = generator.phi_coefficients
phi_ground_truth = np.reshape(phi_ground_truth,(N*Q,N*Q),order = 'F') 
labels_gt = np.array(list(generator.categories.values())) 

###### WRITE TO OUTPUT FILES ######
f = open("generating_hyperparameters.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close() 

# Specify the file path where you want to save the CSV files
Xs_path = "design_matrix.csv"
phi_path = "phi.csv"
labels_gt_path = "labels_gt.csv"

# Use np.savetxt to save the NumPy array to a CSV file with 3 decimal places
np.savetxt(Xs_path, Xs, delimiter=',', fmt='%.3f', comments='')
np.savetxt(phi_path, phi_ground_truth, delimiter=',', fmt='%.3f', comments='')
np.savetxt(labels_gt_path, labels_gt, delimiter=',', fmt='%.3f', comments='')