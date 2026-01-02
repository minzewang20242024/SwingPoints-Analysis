import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class SwingPointDetector:
    
    def __init__(self, window_size: int = 5, min_distance: int = 3):
        self.window_size = window_size  
        self.min_distance = min_distance  

    def _filter_by_min_distance(self, swing_points: np.ndarray) -> list:
        
        if len(swing_points) == 0:
            return []
        
        filtered_points = [swing_points[0]]
        for point in swing_points[1:]:
            if point - filtered_points[-1] >= self.min_distance:
                filtered_points.append(point)
        return filtered_points

    def detect_swing_highs(self, prices: pd.Series) -> list:
       
       
        swing_high_indices = argrelextrema(prices.values, np.greater, order=self.window_size)[0]
        
        valid_indices = [idx for idx in swing_high_indices if idx >= self.window_size and idx <= len(prices)-self.window_size]
        filtered_highs = self._filter_by_min_distance(np.array(valid_indices))
        return filtered_highs

    def detect_swing_lows(self, prices: pd.Series) -> list:
        
        
        swing_low_indices = argrelextrema(prices.values, np.less, order=self.window_size)[0]
        
        valid_indices = [idx for idx in swing_low_indices if idx >= self.window_size and idx <= len(prices)-self.window_size]
        filtered_lows = self._filter_by_min_distance(np.array(valid_indices))
        return filtered_lows