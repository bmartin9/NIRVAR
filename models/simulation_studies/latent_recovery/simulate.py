""" 
Script to simulate data for the latent recovery simulation study.
Inputs: Simulation Hyperparameters 
Outputs: X_{Gamma}, X_{\hat{S}}, X, Q 
"""

#!/usr/bin/env python3
# USAGE: ./simulate.py hyperparameters.yaml 

from src.models import generativeVAR 
import numpy as np
from numpy.random import default_rng
import sys
import yaml
from sklearn.decomposition import TruncatedSVD 
from procrustes import Procrustes 
# from numpy.linalg import svd 
import plotly.graph_objects as go


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

def generate_points_on_unit_circle(B):
    angles = np.linspace(0, 2*np.pi, B, endpoint=False)  # Equally spaced angles
    points = np.column_stack((np.cos(angles), np.sin(angles)))  # Convert angles to points

    return points

def generate_points_on_unit_sphere(B):
    # Generate points on a unit sphere using Fibonacci lattice method
    points = np.zeros((B, 3))
    offset = 2.0 / B
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(B):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y**2)
        phi = ((i + 1) % B) * increment
        points[i] = [np.cos(phi) * r, y, np.sin(phi) * r]

    return points

def generate_points_on_hypersphere_randomly(B, d):
    # Generate B points uniformly distributed on a hypersphere of dimension d
    points = np.random.randn(B, d)
    norms = np.linalg.norm(points, axis=1)
    normalized_points = points / norms[:, np.newaxis]

    return normalized_points

def generate_points_in_hypersphere(dimensions, num_points):
    points = []
    for _ in range(num_points):
        while True:
            # Generate a random point within the positive quadrant
            point = np.random.rand(dimensions)
            # Ensure the magnitude is less than or equal to 1
            if np.linalg.norm(point) <= 1:
                points.append(point)
                break
    return np.array(points)


def scale_points(points,spectral_rad):
    # Step 1: Calculate inner products
    inner_products = np.dot(points, points.T)

    # Step 2: Find the maximum inner product
    max_inner_product_eig = np.max(np.abs(np.linalg.eigvals(inner_products)))

    # Step 3: Scale points
    scaling_factor = np.sqrt(spectral_rad / max_inner_product_eig)
    scaled_points = points * scaling_factor

    return scaled_points

if d==2:
    B_distinct_points = np.array([[0.05,0.95],[0.95,0.05]]) 
elif d==3:
    B_distinct_points = generate_points_on_unit_sphere(B)
else:
    B_distinct_points = generate_points_on_hypersphere_randomly(B, d)

B_probability_matrix = np.dot(B_distinct_points,B_distinct_points.T)

gt_generator = generativeVAR.generativeVAR(random_state=random_state,
                                           N=N,
                                           T=T,
                                           B=B,
                                           Q=Q,
                                           multiplier=spectral_radius,
                                           global_noise=sigma
                                           )

block_assignments_dict = gt_generator.categories
blocks_array = np.array(list(block_assignments_dict.values())) 
X = B_distinct_points[blocks_array]

X_inner_product = np.dot(X,X.T) 
X_inner_eigs = np.linalg.eigvals(X_inner_product)
max_eval_k = np.max(np.abs(X_inner_eigs))
small_phi = spectral_radius*(1/max_eval_k)
expected_value_Phi = small_phi*X_inner_product 

###### Compute SVD of Gamma (theoretical covariance) ###### 
Gamma = np.linalg.inv(np.identity(N) - expected_value_Phi@expected_value_Phi.T)
svd = TruncatedSVD(n_components=d, n_iter=n_iter)  
Gamma_evals_EV , Gamma_evecs_EV = np.linalg.eig(Gamma)
sorted_indices_EV = np.argsort(np.abs(Gamma_evals_EV))[::-1]
Gamma_evals = Gamma_evals_EV[sorted_indices_EV]
largest_indices_Gamma = sorted_indices_EV[:d]
X_Gamma = np.real(Gamma_evecs_EV[:,largest_indices_Gamma]@np.diag(np.sqrt(Gamma_evals_EV[largest_indices_Gamma])))


# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(expected_value_Phi)

# Find indices of eigenvalues sorted by magnitude
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

# Extract the indices of the two largest magnitude eigenvalues
largest_indices = sorted_indices[:d]

# Get the two largest magnitude eigenvalues and corresponding eigenvectors
Lam = eigenvalues[largest_indices]
U = eigenvectors[:,largest_indices]
Lam_tilde = np.diag(1/(1-Lam**2))
Gamma_subspace_d = U@Lam_tilde@U.T 

X_Phi = np.real(U@np.diag(np.sqrt(Lam)))

Gamma_subspace_d_orthogonal = np.identity(N) - U@U.T 
orth_evals , orth_evecs = np.linalg.eig(Gamma_subspace_d_orthogonal)

Gamma_full = Gamma_subspace_d + Gamma_subspace_d_orthogonal
full_evals, full_evecs = np.linalg.eig(Gamma_full)

X_Gamma_subspace_d = U@np.sqrt(Lam_tilde)
is_equal = np.allclose(Gamma, Gamma_full)
Phi_svd = U@np.diag(Lam)@U.T
if is_equal:
    print("Gamma and Gamma_svd are equal")
else:
    print("Gamma and Gamma_svd are not equal")

# Plot Gamma heatmap
fig1 = go.Figure(data=go.Heatmap(z=Gamma, colorscale='RdBu', reversescale=True, zmin = -1, zmax = 1))
fig1.update_layout(title='Gamma Heatmap', xaxis_title='Column', yaxis_title='Row')
# fig1.show()

# Plot Gamma_svd heatmap
fig2 = go.Figure(data=go.Heatmap(z=np.real(Gamma_subspace_d), colorscale='RdBu', reversescale=True , zmin = -1, zmax = 1))
fig2.update_layout(title='Gamma_svd Heatmap', xaxis_title='Column', yaxis_title='Row')
# fig2.show()

# Plot Gamma_svd heatmap
fig3 = go.Figure(data=go.Heatmap(z=np.real(Gamma_subspace_d_orthogonal), colorscale='RdBu', reversescale=True, zmin = -1, zmax = 1))
fig3.update_layout(title='Gamma_svd Heatmap', xaxis_title='Column', yaxis_title='Row')
# fig3.show()

# Plot Gamma_svd heatmap
fig4 = go.Figure(data=go.Heatmap(z=Gamma_full, colorscale='RdBu', reversescale=True, zmin = -1.2, zmax = 1.2))
fig4.update_layout(title='Gamma_svd Heatmap', xaxis_title='Column', yaxis_title='Row')
# fig4.show()


###### Sample from WRDP N_replicas times ###### 
X_S_no_alignment = np.zeros((N_replicas,N,d))
X_S_Gamma = np.zeros((N_replicas,N,d)) 
X_S_Phi = np.zeros((N_replicas,N,d)) 
X_S_gt = np.zeros((N_replicas,N,d)) 
Q_Gamma = np.zeros((N_replicas,d,d))
Q_Phi = np.zeros((N_replicas,d,d))
Q_gt = np.zeros((N_replicas,d,d))
if H == "Gaussian":
    gaussian_cov = np.identity(N)
    Phi = np.repeat(expected_value_Phi[np.newaxis, :, :], N_replicas, axis=0) +  random_state.normal(loc = 0,scale=1,size=(N_replicas,N,N)) #each entry of Phi is iid normal with mean expected_value_Phi_ij
elif H == "Bernoulli":
    random_uniform = random_state.uniform(size=(N,N))
    Phi = np.zeros((N_replicas,N,N))
    for s in range(N_replicas):
        random_uniform = random_state.uniform(size=(N,N))
        Phi1 = np.zeros((N,N))
        Phi1[X_inner_product > random_uniform] = 1
        Phi[s] = Phi1

proc_Gamma = Procrustes(X_Gamma) 
proc_X = Procrustes(X) 
proc_Phi = Procrustes(X_Phi) 
for k in range(N_replicas):
    Phi_k = Phi[k]
    #You need to rescale the Phi sample so that the spectral radius is less than 1
    Phi_k = spectral_radius*(1/np.max(np.abs(X_inner_eigs)))*Phi_k
    Phi_k_eigs = np.linalg.eigvals(Phi_k)
    max_eval_k = np.max(np.abs(Phi_k_eigs))
    if max_eval_k>= 1:
        continue
    else:

        timeseries_generator = generativeVAR.generativeVAR(random_state=random_state,
                                                            N=N,
                                                            T=T,
                                                            B=B,
                                                            Q=Q,
                                                            multiplier=spectral_radius,
                                                            global_noise=sigma,
                                                            phi_coefficients=Phi_k[:,np.newaxis, :, np.newaxis]
                                                )

        X_timeseries = timeseries_generator.generate().reshape(T,N) 
        X_cov = X_timeseries.T@X_timeseries/T 
        X_S_k_evals , X_S_k_evecs = np.linalg.eig(X_cov) 
        sorted_indices_S_k = np.argsort(np.abs(np.real(X_S_k_evals)))[::-1]
        largest_indices_Gamma = sorted_indices_S_k[:d]
        X_S_k_Gamma = np.real(X_S_k_evecs[:,largest_indices_Gamma]@np.diag(np.sqrt(X_S_k_evals[largest_indices_Gamma]))) 
        X_S_no_alignment[k] = X_S_k_Gamma
        X_S_k_Gamma_aligned = proc_Gamma.rotated_Y(X_S_k_Gamma)
        X_S_Gamma[k] = X_S_k_Gamma_aligned
        Q_Gamma_k = proc_Gamma.orthogonal_matrix(X_S_k_Gamma)
        Q_Gamma[k] = Q_Gamma_k

        inverted_evals = np.sqrt(1-(1/X_S_k_evals[largest_indices_Gamma]))
        X_S_k_Phi = np.real(X_S_k_evecs[:,largest_indices_Gamma]@np.diag(np.sqrt(inverted_evals)))
        X_S_k_Phi_aligned = proc_Phi.rotated_Y(X_S_k_Phi)
        X_S_Phi[k] = X_S_k_Phi_aligned
        Q_Phi_k = proc_Phi.orthogonal_matrix(X_S_k_Phi)
        Q_Phi[k] = Q_Phi_k

        X_S_k_gt_aligned = proc_X.rotated_Y((1/np.sqrt(small_phi))*X_S_k_Phi_aligned)
        X_S_gt[k] = X_S_k_gt_aligned
        Q_gt_k = proc_X.orthogonal_matrix((1/np.sqrt(small_phi))*X_S_k_Phi_aligned)
        Q_gt[k] = Q_gt_k


###### SAVE TO CSV FILES ######
X_S_no_alignment = np.reshape(X_S_no_alignment,(N_replicas*N,d))
X_S_Gamma = np.reshape(X_S_Gamma,(N_replicas*N,d))
X_S_Phi = np.reshape(X_S_Phi,(N_replicas*N,d))
X_S_gt = np.reshape(X_S_gt,(N_replicas*N,d))
Q_Gamma = np.reshape(Q_Gamma,(N_replicas*d,d))
Q_Phi = np.reshape(Q_Phi,(N_replicas*d,d))
Q_gt = np.reshape(Q_gt,(N_replicas*d,d))
np.savetxt('X_Gamma.csv', X_Gamma, delimiter=',', fmt='%.5f')
np.savetxt('X_Phi.csv', X_Phi, delimiter=',', fmt='%.5f')
np.savetxt('X_gt.csv', X, delimiter=',', fmt='%.5f')
np.savetxt('X_S_no_align.csv', X_S_no_alignment, delimiter=',', fmt='%.5f')
np.savetxt('X_S_Gamma.csv', X_S_Gamma, delimiter=',', fmt='%.5f')
np.savetxt('X_S_Phi.csv', X_S_Phi, delimiter=',', fmt='%.5f')
np.savetxt('X_S_gt.csv', X_S_gt, delimiter=',', fmt='%.5f')
np.savetxt('Q_Gamma.csv', Q_Gamma, delimiter=',', fmt='%.3f')
np.savetxt('Q_Phi.csv', Q_Phi, delimiter=',', fmt='%.3f')
np.savetxt('Q_gt.csv', Q_gt, delimiter=',', fmt='%.3f')


###### OUTPUT BACKTESTING HYPERPARAMETERS TO FILE ######

f = open("hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()
