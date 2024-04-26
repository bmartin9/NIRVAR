#!/usr/bin/env python3 
# USAGE: ./generativeSBM.py 

# Script to generate timeseries from the NIRVAR model 

import numpy as np
from sklearn.decomposition import TruncatedSVD 
from scipy.stats import invgamma
from scipy.stats import t as t_dist

class generativeVAR():
    """ 
    :param random_state: Random State object. The seed value is set by the user upon instantiation of the Random State object.
    :type random_state: np.random.RandomState

    :param T: Number of observations (time points)
    :type T: int

    :param N: Number of Stocks
    :type N: int

    :param Q: Number of Features
    :type Q: int

    :param stock_names: List of stock names 
    :type stock_names: list

    :param feature_names: A list of feature names 
    :type feature_names: list

    :param B: The number of blocks in the SBM 
    :type B: int

    :param p_in: Probability of an edge forming between two in the same group for a particular feature
    :type p_in: float

    :param p_out: Probability of an edge forming between two in the different group for a particular feature
    :type p_out: float

    :param p_between: Probability of an edge forming between two over different features
    :type p_between: float

    :param categories: Dictionary with keys being the stock names and values being the corresponding groups
    :type categories: dict

    :param adjacency_matrix: Shape = (N,Q,N,Q). Gives the connections between stock-features. Entries are 1s or 0s.
    :type adjacency_matrix: np.ndarray

    :param phi_coefficients: Shape = (N,Q,N,Q). Gives the weighted connections between stock-features. Defines the VAR generative model. 
        Entries are real numbers. Has a spectral radius of <1.
    :type phi_coefficients: np.ndarray

    :param uniform_range: Distribution from which phi is sampled is U(-uniform_range,uniform_range)
    :type uniform_range: float

    :param innovations_variance: Variance of innovations, 
    :type innovations_variance: np.ndarray

    :param multiplier: Spectral radius of Phi
    :type multiplier: float

    :param global_noise: Variance of each innovation - set this if you want the same std for each stock innovation
    :type global_noise: float

    :param different_innovation_distributions: If False, the innovation distribution of each stock will be Normal(0,self.global_noise)
        If True, the innovation distribution of each stock will be Normal(0,sigma) with sigma ~ Inv-Gamma(3,2)
    :type different_innovation_distributions: bool

    :param phi_distribution: Shape = (NQ,NQ)
        A dense array of values for each Phi_{ij}^{(q)}. For example each Phi_{ij}^{(q)} could be drawn 
        from a some distribution that depends on the block membership of i and j. 
    :type phi_distribution: np.ndarray

    :param t_distribution: Whether you want t distributed innovations instead of normally distributed distributions
    :type t_distribution: bool

    :return: None
    :rtype: None
    """

    def __init__(self,
                 random_state : np.random.RandomState,
                 T : int, 
                 B : int,
                 N : int = None,
                 Q : int = None,
                 stock_names : list = None,
                 feature_names : list = None,
                 p_in : float = 0.9,
                 p_out : float = 0.05,
                 p_between : float = 0,
                 categories : dict = None,
                 adjacency_matrix : np.ndarray = None, 
                 phi_coefficients : np.ndarray = None,
                 uniform_range : float = 10,
                 innovations_variance : np.ndarray = None,
                 multiplier : float = 1,
                 global_noise : float = 1,
                 different_innovation_distributions : bool = False,
                 phi_distribution : np.ndarray = None,
                 t_distribution : bool = False
                 ) -> None:
 
        self.random_state = random_state
        self.T = T 
        self.B = B
        self.p_in = p_in
        self.p_out = p_out
        self.p_between = p_between
        self.uniform_range = uniform_range
        self.multiplier = multiplier
        self.different_innovation_distributions = different_innovation_distributions  
        self.global_noise = global_noise 
        self.t_distribution = t_distribution

        if N is None and stock_names is None:
            raise ValueError("You must specify either 'N' or 'stock_names'")
        if N is not None and stock_names is not None:
            if len(stock_names) != N:
                raise ValueError("Length of stock_names must be equal to N")
            self._N = N
            self._stock_names = stock_names 
        elif N is not None:
            self._N = N
            self._stock_names = ['{0}'.format(i) for i in range(N)] 
        elif stock_names is not None:
            self._stock_names = stock_names
            self._N = len(stock_names) 

        if Q is None and feature_names is None:
            raise ValueError("You must specify either 'Q' or 'feature_names'")
        if Q is not None and feature_names is not None:
            if len(feature_names) != Q:
                raise ValueError("Length of feature_names must be equal to Q")
            self._Q = Q
            self._feature_names = feature_names 
        elif Q is not None:
            self._Q = Q
            self._feature_names = ['{0}'.format(q) for q in range(Q)]  
        elif feature_names is not None:
            self._feature_names = feature_names
            self._Q = len(feature_names)  
        self.categories = categories if categories is not None else self.manual_categories()
        self.adjacency_matrix = adjacency_matrix if adjacency_matrix is not None else self.adjacency() 
        self.phi_distribution = phi_distribution if phi_distribution is not None else self.phi_blocks_distribution()
        self.phi_coefficients = phi_coefficients if phi_coefficients is not None else self.phi() 
        self.innovations_variance = innovations_variance if innovations_variance is not None else self.innovations_var()

    @property
    def N(self):
        return self._N 

    @N.setter
    def N(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N must be a positive integer") 
        if value != len(self.stock_names):
            raise ValueError("Length of stock_names must be equal to N")
        self._N = value
        self._stock_names = ['{0}'.format(i) for i in range(value)] 

    @property
    def stock_names(self):
        return self._stock_names

    @stock_names.setter
    def stock_names(self, value):
        if not isinstance(value, list):
            raise ValueError("stock_names must be a list")
        if len(value) != self.N:
            raise ValueError("Length of stock_names must be equal to N")
        self._stock_names = value

    @property
    def Q(self):
        return self._Q 

    @Q.setter
    def Q(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Q must be a positive integer") 
        if value != len(self.feature_names):
            raise ValueError("Length of feature_names must be equal to Q")
        self._Q = value
        self._feature_names = ['{0}'.format(q) for q in range(value)] 

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        if not isinstance(value, list):
            raise ValueError("feature_names must be a list")
        if len(value) != self.Q:
            raise ValueError("Length of feature_names must be equal to Q")
        self._feature_names = value
    
    def random_categories(self):
        """ 
        Returns
        -------
        SBM_groupings : dict
        """
        block_labels = [np.argmax(self.random_state.multinomial(1,[1/self.B]*self.B)) for _ in range(self.N)]
        SBM_groupings = dict(zip(self.stock_names,block_labels)) 
        return SBM_groupings 
    
    def blocks(self) -> np.ndarray:
        """

        :return: blocks. Shape = (N,N). Represents the blocks defined by self.categories as a binary matrix
        :rtype: np.ndarray
        """
        SBM_groupings_matrix = np.zeros((self.N,self.N))
        SBM_groupings_values = list(self.categories.values())
        for i in range(self.N):
            for j in range(self.N):
                if SBM_groupings_values[i] == SBM_groupings_values[j]:
                    SBM_groupings_matrix[i][j] = 1 
                else:
                    continue 
        return SBM_groupings_matrix 
    
    def adjacency(self) -> np.ndarray:
        """ 

        :returns: adjacency_matrix 
            Shape = (N,Q,N,Q). Gives the adjacency matrix of the SBM defined by self.categories. 
            Connections within blocks are 1 with probability p_in.
            Connections in different blocks are 1 with probability p_out. 
            Connections between different features are 1 with probability p_between.
        :rtype: np.ndarray
        """
        blocks = self.blocks() 
        block_probabilities = np.where(blocks==1,self.p_in,self.p_out)
        P = self.p_between*np.ones((self.N,self.Q,self.N,self.Q)) 
        for q in range(self.Q):
            P[:,q,:,q] = block_probabilities 
        adjacency_matrix = self.random_state.binomial(1,P) 
        return adjacency_matrix
    
    def manual_categories(self):
        """ 
        :return: cat
            keys are the stock names, values are the block memberships
        :rtype: dict
        """
        vals = sorted([x%self.B for x in range(self.N)])
        keys = [str(x) for x in range(self.N)]
        cat = dict(zip(keys,vals))
        return cat
 
    def phi_blocks_distribution(self):
        """ 
        :return:  phi_dense.
            Shape = (NQ,NQ)
            Each Phi_{ij} ~ N(mean,1) where mean depends on the block membership of node i
        :rtype: np.ndarray
        """
        phi_dense = np.zeros((self.N,self.Q,self.N,self.Q))
        random_negative_mean = self.random_state.binomial(1,0.5,size=(self.B))
        random_negative_mean = np.where(random_negative_mean==1,-1,1)
        for i in range(self.N):
            mean = random_negative_mean[list(self.categories.values())[i]]*list(self.categories.values())[i]
            mean = 3*mean +10 
            phi_dense[i] = self.random_state.normal(loc=mean,scale=1,size=(self.Q,self.N,self.Q))
        phi_dense = np.reshape(phi_dense,(self.N*self.Q,self.N*self.Q),order='F')
        return phi_dense

    def phi(self) -> np.ndarray:
        """
        :return: phi 
            Shape = (N,Q,N,Q). Keeping zero edges, sample phi from a uniform distribution such that 
            the spectral radius of phi is <1 (for stationary solution of VAR model).
        :rtype: np.ndarray
        """
        connections = self.adjacency_matrix 
        connections = np.reshape(connections,(self.N*self.Q,self.N*self.Q),order='F')  
        phi = connections*self.phi_distribution 
        phi_eigs = np.linalg.eig(phi)[0]
        phi = (1/abs(np.max(phi_eigs)))*phi
        phi = self.multiplier*phi 
        phi_eigs = np.linalg.eig(phi)[0]
        phi = np.reshape(phi,(self.N,self.Q,self.N,self.Q),order='F')
        return phi 
    
    def innovations_var(self) -> np.ndarray: 
        """ 
        :return: var 
            Shape = (N,Q). Variance for each innovation
        :rtype: np.ndarray
        """
        if self.different_innovation_distributions:
            var = invgamma.rvs(a=3,loc=0,scale=2,size=(self.N,self.Q),random_state=self.random_state)
            return var 
        else:
            var = self.global_noise*np.ones((self.N,self.Q))
            return var 

    def generate(self) -> np.ndarray: 
        """ 
        :return: X_stored 
            Shape = (T,N,Q). Generated Time Series from VAR model.
        :rtype: np.ndarray
        """
        X_stored = np.zeros((self.T,self.N,self.Q))
        X = np.zeros((self.N,self.Q))
        if t_dist:
            for t in range(self.T): 
                Z = t_dist.rvs(df=1,scale=self.global_noise,size=(self.N,self.Q))
                X = np.sum(np.sum(self.phi_coefficients*X,axis=2),axis=2) + Z 
                X_stored[t] = X

        else:
            for t in range(self.T): 
                Z = self.random_state.normal(0,np.sqrt(self.innovations_variance)) 
                X = np.sum(np.sum(self.phi_coefficients*X,axis=2),axis=2) + Z 
                X_stored[t] = X
        return X_stored 