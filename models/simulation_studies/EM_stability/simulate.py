""" 
For fixed N and T, perform EM of the GMM likelihood for different SEED values M times and 
create a M \times M matrix of ARIs
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py hyperparameters.yaml 

from src.models import generativeVAR
from src.models import train_model
from src.models import predict_model
import numpy as np
from numpy.random import default_rng
import sys
import yaml
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from scipy import stats
import time
from sklearn.metrics import adjusted_rand_score




with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
N = config["N"] 
T = config["T"] 
K = config["K"]
p_in = config["p_in"]
p_out = config["p_out"]
spectral_radius = config['spectral_radius']
num_SEEDS= config['num_SEEDS']
DGP_SEED = config["DGP_SEED"]

random_state = default_rng(DGP_SEED)


###### GENERATE DATA FROM NIRVAR ######
generator = generativeVAR.generativeVAR(T=T,
                          N=N,
                          B=K,
                          Q=1,
                          p_in = p_in,
                          p_out = p_out,
                          multiplier = spectral_radius,
                          random_state=random_state,
                          phi_distribution=np.ones((N,N))
)

X = generator.generate()
gt_cluster_labels = list(generator.categories.values())

estimated_clusters = np.zeros((num_SEEDS,N))
gt_ari = np.zeros((num_SEEDS))
for i, seed in enumerate(random_state.integers(2,10000,num_SEEDS)):
    # print(f"SEED : {seed}") 

    embedder = train_model.Embedding(y=X,
                                        d=K,
                                        embedding_method='Pearson Correlation',
                                        cutoff_feature=0)
    corr_matrix = embedder.pearson_correlations()
    embedded_matrix = embedder.embed_corr_matrix(corr_matrix=corr_matrix,n_iter=7,random_state=453)

    if i ==0:
        U = embedded_matrix[0] 
        # Define a color palette (extend as needed for your number of clusters)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        fig_U = go.Figure()
        for n in range(U.shape[0]):
            fig_U.add_trace(go.Scatter(
            x=[U[n, 0]],
            y=[U[n, 1]],
            mode='markers',
            textposition='top center',
            marker=dict(size=10, color=colors[gt_cluster_labels[n] % len(colors)]),
            ))
        fig_U.update_layout(showlegend=False)
        layout = go.Layout(
        yaxis=dict(title=r'First principal direction', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
        xaxis=dict(title=r'Second principal direction',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=18,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
        )
        fig_U.update_layout(layout)

        fig_U.write_image(f"U_scatter_{i}.pdf")
        time.sleep(1)
        fig_U.write_image(f"U_scatter_{i}.pdf")

    fit_NIRVAR = train_model.fit(embedded_array=embedded_matrix,
                                    training_set=X,
                                    target_feature=0,
                                    UASE_dim=K,
                                    kmeans_random=seed
                                    )
    gmm_clusters, gmm_labels  = fit_NIRVAR.gmm(k=K)

    estimated_clusters[i] = gmm_labels[0]

    gt_ari_i = adjusted_rand_score(labels_true = gt_cluster_labels,labels_pred=gmm_labels[0])
    gt_ari[i] = gt_ari_i


ari_matrix = np.identity(num_SEEDS)
for i in range(num_SEEDS):
    for j in range(i+1,num_SEEDS):
        ari = adjusted_rand_score(labels_true = estimated_clusters[i],labels_pred=estimated_clusters[j])
        ari_matrix[i,j] = ari 


ari_matrix = (ari_matrix + ari_matrix.T) - np.identity(num_SEEDS)

min_ari = np.min(ari_matrix)
print(f"min ARI: {min_ari}")

mean_gt_ari = np.mean(gt_ari)
std_gt_ari = np.std(gt_ari)

print(f"mean gt ari: {mean_gt_ari}")
print(f"std gt ari: {std_gt_ari}")

colorscale = [[0,'snow'],[1,'rgb(55, 126, 184)']]

heatmap = go.Heatmap(
    z=ari_matrix,
    colorscale=colorscale,
    zmin=0.7,
    zmax=1,
    colorbar=dict(title='ARI', tickvals=[0.7, 1], ticktext=[f'0.7', '1'])
)

layout = go.Layout(
    yaxis=dict(title=r'Seed index', showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True),
    xaxis=dict(title=r'Seed index',showline=True, linewidth=1, linecolor='black',ticks='outside',mirror=True,automargin=True),
    paper_bgcolor='white',
    plot_bgcolor='white',
    font_family="Serif",
    font_size=18,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)

fig = go.Figure(data=[heatmap], layout=layout)
fig.write_image("seed_ari_heatmap.pdf")
time.sleep(1)
fig.write_image("seed_ari_heatmap.pdf")