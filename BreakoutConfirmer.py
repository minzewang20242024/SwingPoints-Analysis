import pandas as pd
import numpy as np

class BreakoutConfirmer:
   
    def __init__(self):
        pass

    def confirm_breakouts(self, prices: pd.Series, downtrend_line: np.ndarray) -> list:
       
        daily_breakout = prices > downtrend_line
       
        consecutive_breakout = daily_breakout & daily_breakout.shift(1, fill_value=False)
       
        valid_breakout_indices = consecutive_breakout[consecutive_breakout == True].index.tolist()
        return valid_breakout_indices

    def confirm_breakdowns(self, prices: pd.Series, uptrend_line: np.ndarray) -> list:
       
      
        daily_breakdown = prices < uptrend_line
       
        consecutive_breakdown = daily_breakdown & daily_breakdown.shift(1, fill_value=False)
      
        valid_breakdown_indices = consecutive_breakdown[consecutive_breakdown == True].index.tolist()
        return valid_breakdown_indices