import pandas as pd
import numpy as np
from .DataFetcher import DataFetcher
from .SwingPointDetector import SwingPointDetector
from .TrendlineFitter import TrendlineFitter
from .BreakoutConfirmer import BreakoutConfirmer

class StockAnalyzer:
   
    def __init__(self, window_size: int = 5, min_distance: int = 3, random_seed: int = 42):
       
        self.data_fetcher = DataFetcher(random_seed=random_seed)
        self.swing_detector = SwingPointDetector(window_size=window_size, min_distance=min_distance)
        self.trend_fitter = TrendlineFitter()
        self.breakout_confirmer = BreakoutConfirmer()

     
        self.analysis_result = {}

    def analyze(self, data_source: str, **kwargs) -> dict:
       
       
        if data_source == "yahoo":
            data = self.data_fetcher.fetch_yahoo_stock(
                ticker=kwargs.get("ticker"),
                start_date=kwargs.get("start_date", "2024-01-01"),
                end_date=kwargs.get("end_date", "2024-12-31")
            )
        elif data_source == "csv":
            data = self.data_fetcher.fetch_local_csv(csv_path=kwargs.get("csv_path"))
        elif data_source == "random":
            data = self.data_fetcher.fetch_random_walk(
                n_points=kwargs.get("n_points", 250),
                initial_price=kwargs.get("initial_price", 100.0),
                volatility=kwargs.get("volatility", 0.5)
            )
        else:
            raise ValueError("yahoo / csv / random")

      
        prices = data["Close"]
        swing_highs = self.swing_detector.detect_swing_highs(prices)
        swing_lows = self.swing_detector.detect_swing_lows(prices)

        
        uptrend_line, uptrend_m, uptrend_b = self.trend_fitter.fit_uptrend_line(prices, swing_lows)
        downtrend_line, downtrend_m, downtrend_b = self.trend_fitter.fit_downtrend_line(prices, swing_highs)

 
        breakouts = self.breakout_confirmer.confirm_breakouts(prices, downtrend_line)
        breakdowns = self.breakout_confirmer.confirm_breakdowns(prices, uptrend_line)


        self.analysis_result = {
            "data": data,
            "metrics": {
                "data_points": len(data),
                "swing_highs_count": len(swing_highs),
                "swing_lows_count": len(swing_lows),
                "breakouts_count": len(breakouts),
                "breakdowns_count": len(breakdowns)
            },
            "swing_points": {
                "swing_highs_indices": swing_highs,
                "swing_lows_indices": swing_lows
            },
            "trendlines": {
                "uptrend_line": uptrend_line,
                "uptrend_slope": uptrend_m,
                "uptrend_intercept": uptrend_b,
                "downtrend_line": downtrend_line,
                "downtrend_slope": downtrend_m,
                "downtrend_intercept": downtrend_b
            },
            "signals": {
                "breakouts_indices": breakouts,
                "breakdowns_indices": breakdowns
            }
        }

        return self.analysis_result

    def export_result_to_excel(self, excel_path: str) -> None:
      
        if not self.analysis_result:
            raise ValueError

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
           
            self.analysis_result["data"].to_excel(writer, sheet_name="Raw_Data", index=False)
          
            metrics_df = pd.DataFrame.from_dict(self.analysis_result["metrics"], orient="index", columns=["Value"])
            metrics_df.to_excel(writer, sheet_name="Core_Metrics")
          
            signals_df = pd.DataFrame({
                "Breakouts_Indices": self.analysis_result["signals"]["breakouts_indices"],
                "Breakdowns_Indices": self.analysis_result["signals"]["breakdowns_indices"]
            })
            signals_df.to_excel(writer, sheet_name="Signals", index=False)