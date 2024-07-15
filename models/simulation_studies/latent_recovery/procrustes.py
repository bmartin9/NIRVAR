""" 
Script containing class to do Procrustes Alignment 
"""

#!/usr/bin/env python3 
# USAGE: ./procrustes.py 

import numpy as np
from numpy.linalg import svd 

class Procrustes():

    def __init__(self,
                 X_ref : np.ndarray,
                 Y_bar : np.ndarray = None,
                 t : int = 1
                 ) -> None:
        """
        Parameters
        ----------
        X_ref : np.ndarray
            Reference shape with which to compare observations

        Y_bar : np.ndarray 
            Sample mean of orthogonally transformed observations 

        t : int 
            Number of observations seen so far

        Returns
        -------
        None
        """
        self.X_ref = X_ref
        self.Y_bar = Y_bar if Y_bar is not None else X_ref 
        self.t = t  


    @property
    def n_datapoints(self) -> int:
        return self.Y_bar.shape[0] 
    
    @property
    def n_dimensions(self) -> int:
        return self.Y_bar.shape[1]

    def __str__(self) -> str:
        message = (f"Number of Datapoints:  {self.n_datapoints}, "
                   f"Dimension of each Datapoint:  {self.n_dimensions}, "
                   f"Number of Observations:  {self.t}")
        return message
    
    def __repr__(self) -> str:
        message = (f"Number of Datapoints:  {self.n_datapoints}, "
                   f"Dimension of each Datapoint:  {self.n_dimensions}, "
                   f"Number of Observations:  {self.t}")
        return message

    def orthogonal_matrix(self,Y : np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Y : np.ndarray
            Array to transform.
            Y.shape = self.X_ref.shape

        Returns
        -------
        R : np.ndarray
            Optimal Orthogonal transformation. 
            R.shape = (self.n_dimensions,self.n_dimensions) 
        """
        #compute svd 
        inner_product = Y.T@self.X_ref 
        U, S, Vh = svd(inner_product,full_matrices = True) 
        R = U@Vh 
        return R 

    def rotated_Y(self,Y : np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Y : np.ndarray
            Array to transform.
            Y.shape = self.X_ref.shape

        Returns
        -------
        rotated_Y : np.ndarray
            Y@R 
        """
        rotated_Y = Y@self.orthogonal_matrix(Y)
        return rotated_Y 
    
    def dist_Y_bar(self) -> float:
        """
        Parameters
        ----------
        Y : np.ndarray
            Array to transform.
            Y.shape = self.X_ref.shape

        Returns
        -------
        dist : float
            Frobenius norm ||Y_bar - X_ref||_{F} 
        """
        dist = np.linalg.norm(self.Y_bar-self.X_ref) 
        return dist
    
    def update_Y_bar(self,Y : np.ndarray) -> None: 
        """
        Parameters
        ----------
        Y : np.ndarray
            Array to transform.
            Y.shape = self.X_ref.shape

        Returns
        -------
        None
        """
        self.t += 1 
        self.Y_bar = (1-1/self.t)*self.Y_bar + (1/self.t)*self.rotated_Y(Y) 
        return None 
    
    def update_reference(self,threshold:float) -> None:
        """
        Parameters
        ----------
        threshold:float
            Threshold for updating reference shape.

        Returns
        -------
        None
        """
        dist_Y_bar = self.dist_Y_bar()
        if dist_Y_bar > threshold:
            self.X_ref = self.Y_bar
        

    