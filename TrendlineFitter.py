import pandas as pd
import numpy as np

class TrendlineFitter:
    
    def __init__(self):
        pass

    def _ols_regression(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        
        n = len(x)
        if n < 2:
            raise ValueError
        
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
      
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y - m * sum_x) / n
        return m, b

    def fit_uptrend_line(self, prices: pd.Series, swing_low_indices: list) -> tuple[np.ndarray, float, float]:
       
        if len(swing_low_indices) < 2:
          
            avg_price = prices.mean()
            trendline = np.full(len(prices), avg_price)
            return trendline, 0.0, avg_price
        
        x = np.array(swing_low_indices)
        y = prices.iloc[swing_low_indices].values
        
        m, b = self._ols_regression(x, y)
       
        trendline = m * np.arange(len(prices)) + b
        return trendline, m, b

    def fit_downtrend_line(self, prices: pd.Series, swing_high_indices: list) -> tuple[np.ndarray, float, float]:
      
        if len(swing_high_indices) < 2:
           
            avg_price = prices.mean()
            trendline = np.full(len(prices), avg_price)
            return trendline, 0.0, avg_price
     
        x = np.array(swing_high_indices)
        y = prices.iloc[swing_high_indices].values
       
        m, b = self._ols_regression(x, y)
     
        trendline = m * np.arange(len(prices)) + b
        return trendline, m, b