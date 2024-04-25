#!/usr/bin/env python3 
# USAGE: ./predict_model.py 

# Script to do NIRVAR predictions.

import numpy as np 
from scipy.stats import spearmanr
from scipy.stats import rankdata

def getCharacteristics(self, startHnd=1, endHnd=0xFFFF, uuids=None):
    """Returns a list containing :class:`bluepy.btle.Characteristic`
    objects for the peripheral. If no arguments are given, will return all
    characteristics. If startHnd and/or endHnd are given, the list is
    restricted to characteristics whose handles are within the given range.

    :param startHnd: Start index, defaults to 1
    :type startHnd: int, optional
    :param endHnd: End index, defaults to 0xFFFF
    :type endHnd: int, optional
    :param uuids: a list of UUID strings, defaults to None
    :type uuids: list, optional
    :return: List of returned :class:`bluepy.btle.Characteristic` objects
    :rtype: list
    """

    pass

def moving_average_predictors(X_test : np.ndarray, alpha : float) -> np.ndarray:
    """ 
    Utility function to compute the exponentially weighted average stock returns over the past l days.

    Parameters
    ----------
    X_test : np.ndarray 
        Last l days of stock values. Shape = (Q,N,l) 

    alpha : float
        exponential parameter

    Returns
    -------
    weighted_average : np.ndarray 
        exponential smoothed predictors. Shape = (Q,N) 

    """
    l = X_test.shape[2] 
    w = np.flip(np.array([alpha**t for t in range(l)])) 
    normalisation = np.sum(w) 
    weighted_y = w*X_test 
    weighted_average = np.sum(weighted_y,axis=2)
    weighted_average = weighted_average/normalisation
    return weighted_average

class predict(): 
    """ 
    Given the set of neighbouring stocks and the coefficients for each feature, predict the next day returns of a stock using the prediction model:

    $$X_{i,q,t+1} = \sum_{q'=1}^{Q}  \sum_{j = 1}^{N} \hat{A}_{ij}^{(qq')} \hat{\Phi}_{ij}^{(qq')} X_{j,q',t}$$ 
    """

    def __init__(self,
                 ols_params  : np.ndarray,
                 todays_Xs :np.ndarray,
                 ) -> None:
        
        """ 

        ols_params  : np.ndarray 
            \Phi coefficients. Should contain zeros for stocks not in neighbourhood
            Shape = (N,N,Q)

        todays_Xs :np.ndarray 
            The value of X_{i,q} for each feature of each stock today
            Shape = (N,Q) 
        """
        self.ols_params = ols_params 
        self.todays_Xs = todays_Xs

    def next_day_prediction(self) -> np.ndarray:
        """ 
        Parameters
        ----------

        Returns
        -------
        s : np.ndarray 
            Next day predictions. Shape = (N) 
        """
        s = np.sum(np.sum(self.ols_params*self.todays_Xs,axis=1),axis=1)
        return s 
    
    def next_day_truth(self,phi:np.ndarray) -> np.ndarray:
        """ 
        Parameters
        ----------
        phi:np.ndarray
            ground truth VAR coeficcients. Shape = (N,N,Q)

        Returns
        -------
        s : np.ndarray 
            Next day values before noise is added. Shape = (N) 
        """
        s = np.sum(np.sum(phi*self.todays_Xs,axis=1),axis=1)
        return s 
    
class benchmarking():

    """
    Class to compute daily benchmarking statistics when doing backtesting.
    """
    def __init__(self,
                 predictions : np.ndarray,
                 market_excess_returns : np.ndarray,
                 yesterdays_predictions : np.ndarray,
                 ) -> None:
        """ 
        Parameters
        ----------
        predictions : np.ndarray
            predicted returns for each day. Shape = (N) 

        market_excess_returns : np.ndarray
            Excess returns on the prediction day. Equal to Raw returns minus SPY returns.
            Shape = (N)

        yesterdays_predictions : np.ndarray
            Predictions from the previous day. Shape = (N). Used to determine if a transaction occured.

        Returns
        -------
        None
        """
        self.predictions = predictions 
        self.market_excess_returns = market_excess_returns 
        self.yesterdays_predictions = yesterdays_predictions

    @property
    def n_stocks(self):
        n_stocks = self.predictions.shape[0] 
        return n_stocks
    
    def hit_ratio(self) -> float: 
        """ 
        Parameters
        ----------


        Returns 
        -------
        ratio : float 
            The fraction of predictions with the same sign as market excess returns
        """
        is_correct_sign = np.sign(self.predictions)*np.sign(self.market_excess_returns) 
        is_corr_ones = np.where(is_correct_sign==1,1,0)
        ratio = np.sum(is_corr_ones)/(self.n_stocks)
        return ratio
    
    def long_ratio(self) -> float:
        """ 
        Parameters
        ----------


        Returns 
        -------
        long_ratio : float 
            The fraction of predictions with sign +1
        """
        prediction_sign = np.sign(self.predictions) 
        is_corr_ones = np.where(prediction_sign==1,1,0)
        long_ratio = np.sum(is_corr_ones)/(self.n_stocks)
        return long_ratio
    
    def corr_SP(self) -> float: 
        """ 
        Parameters
        ----------


        Returns 
        -------
        corr_SP : float 
            The Spearman correlation between your predictions and the target market excess returns 
        """
        rho_sp , p = spearmanr(self.predictions,self.market_excess_returns) 
        return rho_sp 
    
    def PnL(self, quantile : float) -> float: 
        """ 
        Parameters
        ----------
        quantile : float
            The top x% largest (in magnitude) predictions where x \in [0,1].

        Returns
        -------
        PnL : float
            The quantile PnL where PnL is defined as \sum_{all_stocks} sign(predictions)*market_excess_returns
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.n_stocks*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(signed_predictions*self.market_excess_returns)
        return PnL 
    
    def transaction_indicator(self):
        """ 
        Parameters
        ----------


        Returns
        -------
        transaction_indicator : np.ndarray
            1 if a transaction occured, 0 otherwise. Shape = (N)
        """
        transaction_indicator = np.where(np.sign(self.predictions)-np.sign(self.yesterdays_predictions)==0,0,1)
        return transaction_indicator
    
    def weighted_PnL_transactions(self, weights : np.ndarray , quantile : float) -> float: 
        """ 
        Parameters
        ----------
        quantile : float
            The top x% largest (in magnitude) predictions where x \in [0,1].

        weights : np.ndarray
            The portfolio weightings for each stock. Shape = (N)

        Returns
        -------
        PnL : float
            The quantile PnL where PnL is defined as \sum_{all_stocks} sign(predictions)*market_excess_returns
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.n_stocks*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(weights*(signed_predictions*self.market_excess_returns - (0)*self.transaction_indicator())) #4bpts is the transaction cost 
        portfolio_size = np.sum(weights) 
        PnL = PnL/portfolio_size
        return PnL
    

