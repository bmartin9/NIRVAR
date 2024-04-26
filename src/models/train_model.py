#!/usr/bin/env python3 
# USAGE: ./train_model.py 

# Script to find the best OLS estimated coefficients of the NIRVAR model subject to a regularisation via variable selection.
# The variable selection is done by embedding each stock into a latent space and choosing nearby stocks as predictors.

import numpy as np 
from sklearn.decomposition import TruncatedSVD 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.signal import fftconvolve  
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh
import plotly.express as px
from sklearn.mixture import BayesianGaussianMixture
from numpy.linalg import svd

class Embedding(): 
    """
        :param d: Embedding Dimension 
        :type d: int 

        :param y: The training data. Must be in the form of a numpy array of size (T,N,Q)
        :type y: numpy.ndarray

        :param embedding_method: The possible embedding methods are: 'Pearson Correlation' (Default), 'Precision Matrix', 'Spearman Correlation', 'Kendall Correlation', 'Covariance Matrix"
        :type embedding_method: str

        :param cutoff_feature: The feature on which to calculate dhat via MP of embedding method
        :type cutoff_feature: int

        :return: None
        :rtype: None
    """

    def __init__(
        self,
        y : np.ndarray,
        d : int = None,
        embedding_method : str = 'Pearson Correlation',
        cutoff_feature : int = 1
    ) -> None:

        self.y = y 
        self.embedding_method = embedding_method
        self.cutoff_feature = cutoff_feature
        self.d = d if d is not None else self.marchenko_pastur_estimate()  

    @property
    def N(self):
        N = self.y.shape[1] 
        return N
    
    @property
    def Q(self):
        Q = self.y.shape[2] 
        return Q
    
    @property
    def T(self):
        T = self.y.shape[0] 
        return T

    def pearson_correlations(self) -> np.ndarray: 
        """
        :return: A row concatenated array of the (N x N) Pearson correlation matrix for each feature. Size = (Q,N,N)
        :rtype: numpy.ndarray 
        """


        corr_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:,:,q].T) 
            corr_mat[q] = p_corr
        return corr_mat 
    
    def covariance_matrix(self) -> np.ndarray:
        """
            :return: A row concatenated array of the (N x N) covariance matrix for each feature. Size = (Q,N,N)
            :rtype: numpy.ndarray 
        """
            
        cov_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            cov_mat[q] = np.cov(self.y[:,:,q].T) 
        return cov_mat
    
    def precision_matrix(self) -> np.ndarray:
        """
            :return: A row concatenated array of the (N x N) precision matrix for each feature. Size = (Q,N,N)
            :rtype: numpy.ndarray 
        """

        prec_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:,:,q].T) 
            prec_mat[q] = np.linalg.inv(p_corr)
        return prec_mat 
    
    # Marchenko Pastur Function
    def marchenko_pastur_estimate(self) -> int:
        """
            :return: Estimated number of "significant" dimensions
            :rtype: int
        """


        if self.embedding_method == 'Pearson Correlation':
            Sigma = self.pearson_correlations()
            eigenvalues = np.linalg.eigvals(Sigma[self.cutoff_feature])
            cutoff = (1 + np.sqrt(self.N/self.T))**2
            d_hat = np.count_nonzero(eigenvalues > cutoff) 
        elif self.embedding_method == 'Precision Matrix':
            Sigma = self.precision_matrix()
            eigenvalues = np.linalg.eigvals(Sigma[self.cutoff_feature])
            ratio_limit = self.N/self.T 
            cutoff = ((1 - np.sqrt(ratio_limit))/(1 - ratio_limit))**2 
            d_hat = np.count_nonzero(eigenvalues < cutoff) 
        elif self.embedding_method == 'Full UASE Correlation':
            flat_X = np.reshape(self.y[:,:,:2],(self.T,self.N*2)) # include only those features that are being used as predictors
            Sigma = np.corrcoef(flat_X.T) 
            print(f"Sigma shape: {Sigma.shape}")
            ratio_limit = 2*self.N/self.T
            cutoff = (1 + np.sqrt(ratio_limit))**2
            eigenvalues = np.linalg.eigvals(Sigma)
            d_hat = np.count_nonzero(eigenvalues > cutoff) 
        else:
            print("ERROR : Embedding method must be one of Pearson Correlation or Precision Matrix")
        
        return d_hat 

    
    def marcenko_pastur_denoised_correlation(self) -> np.ndarray: 
        """
            :return: A row concatenated array of the (N x N) Pearson correlation matrix for each feature.
            :rtype: np.ndarray
        """

        corr_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            p_corr = np.corrcoef(self.y[:,:,q].T) 
            cutoff = (1+np.sqrt(self.N/self.T))**2
            signal_eigs_all , signal_eigv_all = eigh(p_corr,subset_by_value=(cutoff,np.inf)) 
            signal_corr = signal_eigv_all@np.diag(signal_eigs_all)@signal_eigv_all.T
            if np.array_equal(signal_corr,np.zeros((self.N,self.N))):
                print("WARNING: All eigenvalues are less than Marcenko-Pastur cutoff")
                corr_mat[q] = p_corr 
            else:
                corr_mat[q] = signal_corr 
        return corr_mat 
    
    
    def spearman_correlations(self) -> np.ndarray: 
        """
            :return: A row concatenated array of the (N x N) Spearman correlation matrix for each feature.
            :rtype: np.ndarray
        """

        corr_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            rho_corr, p = spearmanr(self.y[:,:,q]) 
            corr_mat[q] = rho_corr
        return corr_mat 
    
    def kendall_correlations(self) -> np.ndarray: 
        """
            :return: A row concatenated array of the (N x N) Kendall correlation matrix for each feature.
            :rtype: np.ndarray
        """

        corr_mat = np.zeros((self.Q,self.N,self.N)) 
        for q in range(self.Q):
            tau = np.ones((self.N,self.N))
            i, j = np.triu_indices(self.N, k=1)
            row_samples = self.y[:,:,q].T[i] 
            column_samples = self.y[:,:,q].T[j] 
            kendall_func = np.vectorize(kendalltau,signature='(n),(n)->(),()') 
            res = kendall_func(row_samples,column_samples)[0] 
            tau[i,j] = res[0]
            tau += tau.T 
            corr_mat[q] = tau 

        return corr_mat 
    
    def embed_corr_matrix(self,corr_matrix : np.ndarray, n_iter : int, random_state : int) -> np.ndarray: 
        """
        :param corr_matrix: Correlation matrix to be embedded. Size = (Q,N,N). Note you can also pass the precision matrix in here.
        :type corr_matrix: numpy.ndarray

        :param n_iter: Number of iterations to run randomized SVD solver.
        :type n_iter: int

        :param random_state: Used during randomized svd. Pass an int for reproducible results across multiple function calls.
        :type random_state: int

        :return: embedded_array. UASE for each stock-feature vector. Size = (Q,N,d)
        :rtype: numpy.ndarray
        """
        flat_dist_corr = np.reshape(corr_matrix,(self.N*self.Q,self.N)) 
        svd = TruncatedSVD(n_components=self.d, n_iter=n_iter, random_state=random_state)  
        embedded_array = svd.fit_transform(flat_dist_corr)
        embedded_array = np.reshape(embedded_array,(self.Q,self.N,self.d)) 
        return embedded_array 
    
    def embed_design_matrix(self,X : np.ndarray, n_iter : int, random_state : int) -> np.ndarray: 
        """
            :param X: Design matrix to be embedded. Size = (N,Q*T)
            :type X: np.ndarray

            :param n_iter: Number of iterations to run randomized SVD solver
            :type n_iter: int

            :param random_state: Used during randomized svd. Pass an int for reproducible results across multiple function calls
            :type random_state: int

            :return: UASE for each stock vector. Size = (N,d)
            :rtype: np.ndarray
        """

        svd = TruncatedSVD(n_components=self.d, n_iter=n_iter, random_state=random_state)  
        embedded_array = svd.fit_transform(X)
        return embedded_array 
    

class fit(): 

    """ 
    Compute the neighbours of stock i, compute the corresponding covariates for stock i at time t and do regression against the 
    actual value of stock i at time t. The estimated regression coefficients are computed using OLS.

    :param embedded_array: UASE for each stock-feature vector. Size = (Q,N,d)
    :type embedded_array: np.ndarray

    :param training_set: Array containing the training data. Size = (T_train,N,Q)
    :type training_set: np.ndarray

    :param target_feature: The feature that you are trying to predict e.g. pvCLCL returns
    :type target_feature: int

    :param cutoff_distance: The radius within which stocks are included in the VAR predictive model. It is the same for all stock-features
    :type cutoff_distance: float

    :param UASE_dim: Embedding Dimension
    :type UASE_dim: int

    :param alpha: Exponential parameter
    :type alpha: float

    :param lookback_window: How many previous days to use to predict tomorrow's returns
    :type lookback_window: int

    :param weights: The EMA weights. They must sum to 1. Shape = (lookback_window)
    :type weights: np.ndarray

    :return: None
    :rtype: None
    """


    def __init__(self,
                embedded_array : np.ndarray,
                training_set : np.ndarray,
                target_feature : int,
                cutoff_distance : float = 1,
                UASE_dim : int = 3,
                alpha : float = 0.4,
                lookback_window : int = 1,
                weights : np.ndarray = None,
                kmeans_random : int = 253,
                ) -> None:

        self.embedded_array = embedded_array
        self.training_set = training_set
        self.target_feature = target_feature
        self.cutoff_distance = cutoff_distance
        self.UASE_dim = UASE_dim
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.weights = weights if weights is not None else self.compute_weights() 
        self.kmeans_random = kmeans_random 

    @property
    def N(self):
        N = self.training_set.shape[1]
        return N
    
    @property
    def Q(self):
        Q = self.training_set.shape[2]
        return Q
    
    @property
    def T_train(self):
        T_train = self.training_set.shape[0]
        return T_train
    
    def compute_weights(self) -> np.ndarray:
        weights = np.zeros((self.lookback_window))
        for t in range(self.lookback_window):
            weights[t] = self.alpha**t 
        return weights  
    
    def euclid_distances(self) -> np.ndarray:
        """
            :return: For every feature, the NxN euclidean pairwise distances between each stock. Shape=(Q,N,N)
            :rtype: np.ndarray
        """

        euclid_distances = np.zeros((self.Q,self.N,self.N))

        for q in range(self.Q):
            feature_embedding = self.embedded_array[q,:,:]
            feature_distances = np.zeros((self.N,self.N))
            i, j = np.triu_indices(self.N, k=1)
            row_samples = feature_embedding[i]
            column_samples = feature_embedding[j]
            diff = row_samples - column_samples
            ssd = np.sum(diff**2, axis=1)
            distances =  np.sqrt(ssd)
            feature_distances[i,j] = distances
            feature_distances += feature_distances.T
            euclid_distances[q,:,:] = feature_distances 

        return euclid_distances 

    def epsilon_constraint(self) -> np.ndarray: 
        """
            :return: A binary array with value 1 if the euclidean distance is less than cutoff_distance and 0 otherwise. Shape = (Q,N,N) 
            :rtype: np.ndarray
        """
        euclidean_distances = self.euclid_distances() 
        constrained_distances = np.where(euclidean_distances <= self.cutoff_distance, 1 , 0) 
        return constrained_distances

    def k_nearest_neighbours_constraint(self,K : int) -> np.ndarray: 
        """
            :return: A binary array with value 1 for the K nearest stocks and 0 otherwise. Shape=(Q,N,N)
            :rtype: np.ndarray
        """

        constrained_distances = np.zeros((self.Q,self.N,self.N))
        euclid_dist = self.euclid_distances()
        for q in range(self.Q): 
            feature_distances = euclid_dist[q,:,:] 
            ranks = rankdata(feature_distances,axis=1,method='min') 
            feature_constraints = np.where(ranks<=K,1,0) # there may be more than K 1's per row if two neighbouring stocks have exactly the same position
            constrained_distances[q,:,:] = feature_constraints

        return constrained_distances
    
    @staticmethod
    def groupings_to_2D(input_array : np.ndarray) -> np.ndarray:
        """ 
        Turn a 1d array of integers (groupings) into a 2d binary array, A, where 
        A[i,j] = 1 iff i and j have the same integer value in the 1d groupings array.

        :param input_array: 1d array of integers.
        :type input_array: np.ndarray

        :return: 2d Representation. Shape = (len(input_array),len(input_array))
        :rtype: np.ndarray
        """

        L = len(input_array)
        A = np.zeros((L,L)) 
        for i in range(L):
            for j in range(L): 
                if input_array[i] == input_array[j]:
                    A[i][j] = 1 
                else:
                    continue 
        
        return A 

    def k_means(self,k : int) -> np.ndarray:
        """ 
        k means clustering.

        Parameters
        ----------
        :param k: the number of clusters 
        :type k: int

        :return: A list with the first element being a binary array with value 1 for the neighbouring stocks in the same cluster and 0 otherwise
          having shape = (Q,N,N)  The second element is an array of integers where each integer labels a kmeans cluster. Shape = (Q,N) 
        :rtype: list
        """
        constrained_distances = np.zeros((self.Q,self.N,self.N))
        all_labels = np.zeros((self.Q,self.N))
        for q in range(self.Q): 
            feature_embedding = self.embedded_array[q,:,:]
            kmeans = KMeans(n_clusters=k, random_state=self.kmeans_random, n_init="auto").fit(feature_embedding)
            labels = kmeans.labels_ 
            all_labels[q] = labels
            similarity_matrix = self.groupings_to_2D(labels) 
            constrained_distances[q] = similarity_matrix

        return constrained_distances , all_labels 
    
    def gmm(self, k: int) -> np.ndarray:
        """
        GMM clustering. Number of clusters must be pre-specified. EM algorithm is then run.

        :param k: The number of clusters.
        :type k: int

        :return: A binary array with value 1 for the neighboring stocks in the same cluster and 0 otherwise.
                    Shape = (Q, N, N)
        :rtype: np.ndarray

        :return: Array of integers where each integer labels a k-means cluster. Shape = (Q, N)
        :rtype: np.ndarray
        """
        constrained_distances = np.zeros((self.Q, self.N, self.N))
        all_labels = np.zeros((self.Q, self.N))
        for q in range(self.Q):
            feature_embedding = self.embedded_array[q, :, :]
            gmm_labels = GaussianMixture(n_components=k, random_state=self.kmeans_random, init_params='k-means++').fit_predict(feature_embedding)
            labels = gmm_labels
            all_labels[q] = labels
            similarity_matrix = self.groupings_to_2D(labels)
            constrained_distances[q] = similarity_matrix

        return constrained_distances, all_labels
    
    def full_UASE_gmm(self, k: int) -> np.ndarray:
        """
        GMM clustering. Multiple features are clustered at once. Number of clusters must be pre specified. EM algorithm is then run.

        :param k: The number of clusters
        :type k: int

        :return: A binary array with value 1 for the neighbouring stocks in the same cluster and 0 otherwise (Shape = (Q,N,N))
        :rtype: np.ndarray

        :return: Array of integers where each integer labels a kmeans cluster (Shape = (Q,N))
        :rtype: np.ndarray
        """
        feature_embedding = self.embedded_array  # shape = (Q,N,d)
        feature_embedding = np.reshape(feature_embedding, (self.Q * self.N, self.UASE_dim))  # shape = (Q*N,d)
        gmm_labels = GaussianMixture(n_components=k, random_state=self.kmeans_random, init_params='k-means++').fit_predict(feature_embedding)
        labels = gmm_labels
        similarity_matrix = self.groupings_to_2D(labels)
        all_labels = np.reshape(labels, (self.Q, self.N))
        target_similarity = similarity_matrix[self.target_feature * self.N:self.target_feature * self.N + self.N, :]  # we only care about the predictors of the target feature
        constrained_distances = np.reshape(target_similarity, (self.N, self.Q, self.N)).transpose(1, 0, 2)  # shape = (Q,N,N)
        return constrained_distances, all_labels
        

    def covariates(self,constrained_array : np.ndarray) -> np.ndarray: 
        """ 
        :param constrained_array: Shape = (Q,N,N) Some constraint on which neighbours to sum up to get a predictor along that feature.
        :type: np.ndarray 

        :rtype: np.ndarray 
        :return: Shape = (N,N,Q,T_train) For each stock, we have a maximum (this max is not reached do to clustering regularisation) of NQ predictors. There are T_train training values for each predictor.
        :rtype: np.ndarray 
        """
        covariates = np.zeros((self.N,self.N,self.Q,self.T_train),dtype=np.float32)
        for i in range(self.N):
            c = constrained_array[:,i,:,None]*(self.training_set.transpose(2,1,0)) 
            c = c.transpose(1,0,2)
            covariates[i] = c
        return covariates
    
    def ols_parameters(self,constrained_array : np.ndarray) -> np.ndarray:
        """ 
        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape= (Q,N,N)
        :type: np.ndarray

        :return: ols_params Shape = (N,N,Q)
        :rtype: np.ndarray
        """
        ols_params = np.zeros((self.N,self.N*self.Q)) 
        covariates = self.covariates(constrained_array=constrained_array)
        targets = self.training_set[:,:,self.target_feature] # shape = (T_train,N)
        for i in range(self.N):
            ols_reg_object = LinearRegression(fit_intercept=False)
            x = covariates[i].reshape(-1,covariates[i].shape[-1],order='F').T[:-1,:] #shape = (T_train-1,NQ)
            non_zero_col_indices = np.where(x.any(axis=0))[0] #only do ols on stocks that are connected to node i
            x_reg = x[:,non_zero_col_indices]
            y = targets[1:,i] 
            ols_fit = ols_reg_object.fit(x_reg,y) 
            ols_params[i,non_zero_col_indices] = ols_fit.coef_  
        ols_params = np.reshape(ols_params,(self.N,self.N,self.Q),order='F') 
        return ols_params 
    
    def lasso_parameters(self,constrained_array : np.ndarray) -> np.ndarray:
        """ 
        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape= (Q,N,N)
        :type: np.ndarray

        :return: lasso_params Shape = (N,N,Q)
        :rtype: np.ndarray
        """
        lasso_params = np.zeros((self.N,self.N*self.Q)) 
        covariates = self.covariates(constrained_array=constrained_array)
        targets = self.training_set[:,:,self.target_feature] # shape = (T_train,N)
        for i in range(self.N):
            lasso_reg_object = Lasso(alpha=0.05,fit_intercept=False)
            x = covariates[i].reshape(-1,covariates[i].shape[-1],order='F').T[:-1,:] #shape = (T_train-1,NQ)
            non_zero_col_indices = np.where(x.any(axis=0))[0] #only do ols on stocks that are connected to node i
            x_reg = x[:,non_zero_col_indices]
            y = targets[1:,i] 
            lasso_fit = lasso_reg_object.fit(x_reg,y) 
            lasso_params[i,non_zero_col_indices] = lasso_fit.coef_  
        lasso_params = np.reshape(lasso_params,(self.N,self.N,self.Q),order='F') 
        return lasso_params 
    
    def covariates_subset(self,constrained_array : np.ndarray, subset_to_predict : np.ndarray) -> np.ndarray: 
        """ 
        Find the covariates for only a subset of the N panel components.

        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape = (Q,N,N)
        :type: np.ndarray

        :param subset_to_predict: An M dimensional array of indices. Must be a subset of {0,...,(N-1)}
        :type: np.ndarray

        :return: covariates. Shape = (N,N,Q,T_train) For each stock, we have a maximum (this max is not reached do to clustering regularisation) of NQ predictors. There are T_train training values for each predictor. Only stocks in subset_to_predict will be non zero
        :rtype: np.ndarray

        """
        covariates = np.zeros((self.N,self.N,self.Q,self.T_train),dtype=np.float32)
        for i in range(self.N): 
            if i in subset_to_predict:
                c = constrained_array[:,i,:,None]*(self.training_set.transpose(2,1,0)) 
                c = c.transpose(1,0,2)
                covariates[i] = c
        return covariates

    def ols_parameters_subset(self,constrained_array : np.ndarray, subset_to_predict : np.ndarray) -> np.ndarray:
        """ 
        Do OLS only on a subset of the N panel components.

        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape = (Q,N,N)
        :type: np.ndarray

        :param subset_to_predict: An M dimensional array of indices. Must be a subset of {0,...,(N-1)}
        :type: np.ndarray

        :return: ols_params.  Shape = (N,N,Q)
        :rtype: np.ndarray
        """
        ols_params = np.zeros((self.N,self.N*self.Q)) 
        covariates = self.covariates_subset(constrained_array=constrained_array,subset_to_predict=subset_to_predict)
        targets = self.training_set[:,:,self.target_feature] # shape = (T_train,N)
        for i in range(self.N): 
            if i in subset_to_predict:
                ols_reg_object = LinearRegression(fit_intercept=False)
                x = covariates[i].reshape(-1,covariates[i].shape[-1],order='F').T[:-1,:] #shape = (T_train-1,NQ)
                non_zero_col_indices = np.where(x.any(axis=0))[0] #only do ols on stocks that are connected to node i
                x_reg = x[:,non_zero_col_indices]
                y = targets[1:,i] 
                ols_fit = ols_reg_object.fit(x_reg,y) 
                ols_params[i,non_zero_col_indices] = ols_fit.coef_  
        ols_params = np.reshape(ols_params,(self.N,self.N,self.Q),order='F') 
        return ols_params 
