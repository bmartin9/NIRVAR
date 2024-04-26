""" 
Script to generate plots to show the bias of the restricted estimator when the restrictions
are incorrect.
"""

#!/usr/bin/env python3
# USAGE: ./0.3-restricted-bias.py 

import numpy as np
from numpy.random import default_rng
import pandas as pd
import plotly.express as px
import time
import plotly.graph_objects as go
from src.visualization import utility_funcs
from src.models import generativeVAR
import plotly.express as px
from src.models import train_model
from scipy.stats import norm 
from scipy.stats import kstest 


##### PARAMETERS ######
N = 50
T = 5000
Q=1
SEED = 94032
random_state = default_rng(seed=SEED) 
spectral_radius = 0.9
B=5
d = 5
n_iter = 7 
target_feature = 0 
N_gaussian_mixtures = 5
p_in = 0.75
p_out = 0.2
p_between = 0.5
var = 1 
num_replicas = 10000

# Specify true Adjacency matrix
generator = generativeVAR.generativeVAR(random_state=random_state,
                                        N=N,
                                        T=T,
                                        Q=Q,
                                        multiplier=spectral_radius,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        p_between=p_between)
A = generator.adjacency_matrix 
phi_true = generator.phi_coefficients 
R = utility_funcs.get_R(np.reshape(A,(N,N)))

# Compute the Covariance Matrix, \Gamma 
def get_gamma(phi : np.ndarray, noise_variance : float) -> np.ndarray: 
    n = phi.shape[0]
    sigma = noise_variance*np.identity(n)
    vec_sigma = np.reshape(sigma,(n**2),order='F')
    phi_kron = -np.kron(phi,phi)
    np.fill_diagonal(phi_kron, phi_kron.diagonal() + 1)
    vec_gamma = np.linalg.inv(phi_kron)@vec_sigma 
    gamma = np.reshape(vec_gamma,(n,n),order='F')
    return gamma 

true_Gamma = get_gamma(phi_true[:,0,:,0],noise_variance=1)

# Specify incorrect Adjacency matrix
generator2 = generativeVAR.generativeVAR(random_state=random_state,
                                        N=N,
                                        T=T,
                                        Q=Q,
                                        multiplier=spectral_radius,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        p_between=p_between)
A_incorrect = generator2.adjacency_matrix 
A_incorrect = np.reshape(A_incorrect,(Q,N,N))
R_hat = utility_funcs.get_R(np.reshape(A_incorrect,(N,N)))

R_hat_ols_estimates = np.zeros((num_replicas,N,N))
for k in range(num_replicas): 
    print(k)
    generator = generativeVAR.generativeVAR(random_state=random_state,
                                        N=N,
                                        T=T,
                                        Q=Q,
                                        adjacency_matrix=A,
                                        phi_coefficients=phi_true,
                                        multiplier=spectral_radius,
                                        B=B,
                                        p_in=p_in,
                                        p_out=p_out,
                                        p_between=p_between) 
    X = generator.generate() 
    current_embedding = train_model.Embedding(y=X,d=d) 
    current_corr = current_embedding.pearson_correlations() 
    current_embedded_array = current_embedding.embed_corr_matrix(current_corr,n_iter=n_iter,random_state=235)
    trainer = train_model.fit(current_embedded_array,X,target_feature,UASE_dim=d)
    ols_params_R_hat = trainer.ols_parameters(A_incorrect) 
    R_hat_ols_estimates[k] = ols_params_R_hat[:,:,target_feature] 

    print ("\033[A                             \033[A") 

# Compute [R'[\Gamma \otimes \Sigma]R]^{-1} 
unrestricted_variance = np.kron(true_Gamma,var*np.identity(N))
restricted_variance_inv = R_hat.T@unrestricted_variance@R_hat
restricted_variance = np.linalg.inv(restricted_variance_inv) # shape = (M_hat,M_hat) 

unrestricted_variance_inv = np.linalg.inv(unrestricted_variance)

# compute bias factor 
C = restricted_variance@(R_hat.T@unrestricted_variance@R) # shape = (M_hat,M) 

def get_nonzero_index(v,i):
    """ 
    Parameters
    ----------
    v :np.ndarray
        1-d numpy array 
    i : int
        index of original array 

    Returns
    -------
    j : int 
        index of original array when the zeros have been removed 
    """ 
    num_zeros_previous = np.count_nonzero(v[:i]==0)
    j = i - num_zeros_previous 
    return j 

def get_nonzero_index_inverse(v,j):
    count = 0 
    return_index = 0
    for i in range(len(v)):
        if v[i]!=0:
            count+=1
            if count == j+1: 
                return_index = i
                break
    return return_index

phi_vec = np.reshape(phi_true,(N**2),order='F') 
R_hat_ols_estimates_vec = np.reshape(R_hat_ols_estimates,(num_replicas,N**2),order='F') 

# compute \gamma and \gamma_{R_hat} 
phi_vec_nonzero = phi_vec[phi_vec!=0]
R_hat_ols_estimates_vec_nonzero = R_hat_ols_estimates_vec[R_hat_ols_estimates_vec!=0]
M = phi_vec_nonzero.shape[0]
M_hat = int(R_hat_ols_estimates_vec_nonzero.shape[0]/num_replicas)
R_hat_ols_estimates_vec_nonzero = np.reshape(R_hat_ols_estimates_vec_nonzero,(num_replicas,M_hat),order='F')

# Compute C \gamma 
C_gamma = C@phi_vec_nonzero # shape = (M_hat,)  

# Plot histogram and normal curve 
non_zero_index = 0
index = get_nonzero_index_inverse(R_hat_ols_estimates_vec[0],non_zero_index) 

asy_var_hat = restricted_variance[non_zero_index,non_zero_index] 
unrestricted_asy_var_hat = unrestricted_variance_inv[index,index] 
x_normal = np.linspace(-6,1,num=500)
y_normal = norm.pdf(x_normal,loc=np.sqrt(T)*C_gamma[non_zero_index],scale=np.sqrt(asy_var_hat))
y_normal_unrestricted = norm.pdf(x_normal,loc=np.sqrt(T)*C_gamma[non_zero_index],scale=np.sqrt(unrestricted_asy_var_hat))  
histogram_data1 = go.Histogram(x=np.sqrt(T)*R_hat_ols_estimates_vec[:,index],histnorm='probability density',nbinsx=50,name='$\sqrt{T} (\mathbf{\hat{\gamma}}(\hat{A}))_{i}$',opacity = 0.7, marker=dict(color='blue'))
scatter_line1 = go.Scatter(x=x_normal, y=y_normal, mode='lines',name='$\mathcal{N}(\sqrt{T} (C_{\infty} \mathbf{\gamma}(A))_{i},([R(\hat{A})\'(\Gamma \otimes \Sigma^{-1}) R(\hat{A})]^{-1})_{ii})$', line=dict(color='red'))


layout = go.Layout(
    yaxis=dict(title='Probability', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title='NIRVAR Estimate',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width =500, 
    height=350,
    barmode='overlay'  # This will make the histograms overlap
)
layout['legend'] = dict(
    orientation='v',  # 'h' for horizontal, 'v' for vertical
    x=0,  # Adjust as needed
    y=1.3,  # Adjust as needed
    bgcolor='rgba(255, 255, 255, 0.5)',  # Set legend background color
    bordercolor='white',  # Set legend border color
    borderwidth=1,  # Set legend border width
)
fig = go.Figure(data=[ histogram_data1, scatter_line1], layout=layout)

fig.write_image("bias-asymptotics.pdf")
time.sleep(1)
fig.write_image("bias-asymptotics.pdf")

###### KS Test ######
ks_statistics = np.zeros((M_hat))
ks_pvals = np.zeros((M_hat))
print(f"M_hat: {R_hat_ols_estimates_vec_nonzero.shape}")
print(f"restricted variance: {restricted_variance.shape}")
for i in range(M_hat):
    ks_index = get_nonzero_index_inverse(R_hat_ols_estimates_vec[0],i) 
    ks_asy_var_hat = restricted_variance[i,i] 
    ks_statistic, ks_p_value = kstest(np.sqrt(T)*R_hat_ols_estimates_vec[:,ks_index], 'norm', args=(np.sqrt(T)*C_gamma[i], np.sqrt(ks_asy_var_hat)))
    ks_statistics[i] = ks_statistic
    ks_pvals[i] = ks_p_value

np.savetxt('ks_statistic.csv', ks_statistics, delimiter=',', fmt='%.4f')
np.savetxt('ks_pvals.csv', ks_pvals, delimiter=',', fmt='%.4f')
np.savetxt('gamma_hat_nonzero.csv', R_hat_ols_estimates_vec_nonzero, delimiter=',', fmt='%.4f')
