#!/usr/bin/env python3 
# USAGE: ./predict_model.py 

# Script to do NIRVAR predictions.

import numpy as np 
from scipy.stats import spearmanr
from scipy.stats import rankdata


def moving_average_predictors(X_test : np.ndarray, alpha : float) -> np.ndarray:
    """ 
    Utility function to compute the exponentially weighted average stock returns over the past l days.

    :param X_test: Last l days of stock values. Shape = (Q,N,l) 
    :type X_test: np.ndarray 

    :param alpha: Exponential parameter
    :type alpha: float

    :return weighted_average: Exponential smoothed predictors. Shape = (Q,N) 
    :rtype weighted_average: np.ndarray 
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

    :param ols_params: Phi coefficients. Should contain zeros for stocks not in neighbourhood. Shape = (N,N,Q)
    :type ols_params: np.ndarray 

    :param todays_Xs: The value of X_{i,q} for each feature of each stock today. Shape = (N,Q) 
    :type todays_Xs: np.ndarray 
    """

    def __init__(self,
                 ols_params  : np.ndarray,
                 todays_Xs :np.ndarray,
                 ) -> None:
        
        self.ols_params = ols_params 
        self.todays_Xs = todays_Xs

    def next_day_prediction(self) -> np.ndarray:
        """ 
        :return: s 
            Next day predictions. Shape = (N) 
        :rtype: np.ndarray 
        """
        s = np.sum(np.sum(self.ols_params*self.todays_Xs,axis=1),axis=1)
        return s 
    
    def next_day_truth(self,phi:np.ndarray) -> np.ndarray:
        """ 
        :param phi: Ground truth VAR coefficients. Shape = (N,N,Q)
        :type phi: np.ndarray 

        :return s: Next day values before noise is added. Shape = (N) 
        :rtype s: np.ndarray 
        """

        s = np.sum(np.sum(phi*self.todays_Xs,axis=1),axis=1)
        return s 
    
class benchmarking():

    """
    Class to compute daily benchmarking statistics when doing backtesting.

    :param predictions: Predicted returns for each day. Shape = (N) 
    :type predictions: np.ndarray 

    :param market_excess_returns: Excess returns on the prediction day. Equal to Raw returns minus SPY returns. Shape = (N) 
    :type market_excess_returns: np.ndarray 

    :param yesterdays_predictions: Predictions from the previous day. Shape = (N). Used to determine if a transaction occurred. 
    :type yesterdays_predictions: np.ndarray 

    :return: None
    :rtype: None
    """
    def __init__(self,
                 predictions : np.ndarray,
                 market_excess_returns : np.ndarray,
                 yesterdays_predictions : np.ndarray,
                 ) -> None:
        
        self.predictions = predictions 
        self.market_excess_returns = market_excess_returns 
        self.yesterdays_predictions = yesterdays_predictions

    @property
    def n_stocks(self):
        n_stocks = self.predictions.shape[0] 
        return n_stocks
    
    def hit_ratio(self) -> float: 
        """ 
        :return: ratio
            The fraction of predictions with the same sign as market excess returns
        :rtype: float
        """
        is_correct_sign = np.sign(self.predictions)*np.sign(self.market_excess_returns) 
        is_corr_ones = np.where(is_correct_sign==1,1,0)
        ratio = np.sum(is_corr_ones)/(self.n_stocks)
        return ratio
    
    def long_ratio(self) -> float:
        """ 
        :return: long_ratio 
            The fraction of predictions with sign +1
        :rtype: float
        """
        prediction_sign = np.sign(self.predictions) 
        is_corr_ones = np.where(prediction_sign==1,1,0)
        long_ratio = np.sum(is_corr_ones)/(self.n_stocks)
        return long_ratio
    
    def corr_SP(self) -> float: 
        """ 
        :return: corr_SP 
            The Spearman correlation between your predictions and the target market excess returns
        :rtype: float 
        """
        rho_sp , p = spearmanr(self.predictions,self.market_excess_returns) 
        return rho_sp 
    
    def PnL(self, quantile : float) -> float: 
        """ 
        :param quantile: The top x% largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float 

        :return PnL: The quantile PnL where PnL is defined as \sum_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float 
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.n_stocks*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(signed_predictions*self.market_excess_returns)
        return PnL 
    
    def transaction_indicator(self):
        """ 
        :return: transaction_indicator
            1 if a transaction occured, 0 otherwise. Shape = (N)
        :rtype: np.ndarray
        """
        transaction_indicator = np.where(np.sign(self.predictions)-np.sign(self.yesterdays_predictions)==0,0,1)
        return transaction_indicator
    
    def weighted_PnL_transactions(self, weights : np.ndarray , quantile : float) -> float: 
        """ 
        :param quantile: The top x% largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float 

        :param weights: The portfolio weightings for each stock. Shape = (N)
        :type weights: np.ndarray 

        :return PnL: The quantile PnL where PnL is defined as ∑_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float 
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.n_stocks*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(weights*(signed_predictions*self.market_excess_returns - (0)*self.transaction_indicator())) #4bpts is the transaction cost 
        portfolio_size = np.sum(weights) 
        PnL = PnL/portfolio_size
        return PnL
    

