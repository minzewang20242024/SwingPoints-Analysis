import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from swing_point_detector import SwingPointDetector

class StockAnalyzer:
    """
    Stock Trend Analysis Class (Optimized for Large-Cap Tech Stocks)
    Functionality: Download stock data, fit trend lines, detect candidate breakout/breakdown signals.
    Core data is initialized via __init__, and users can select data sources and calculation parameters.
    """

    def __init__(self, ticker: str = None, start_date: str = None, end_date: str = None):
        """
        Class Initialization: Initialize core data, parameters and swing point detector.
        :param ticker: Stock ticker symbol (e.g., "AAPL", default None)
        :param start_date: Analysis start date (format "YYYY-MM-DD", default None)
        :param end_date: Analysis end date (format "YYYY-MM-DD", default None)
        """
        # Core data initialization (passed via __init__ for better encapsulation)
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = pd.DataFrame()  # Store downloaded complete stock data
        self.prices = pd.Series()         # Store extracted closing price series
        self.volume = pd.Series()         # Store extracted trading volume series

        # Trend line and signal data initialization
        self.uptrend_line = None  # Uptrend line (fitted by swing lows)
        self.downtrend_line = None  # Downtrend line (fitted by swing highs)
        self.breakout_indices = []  # Candidate breakout signal indices
        self.breakdown_indices = []  # Candidate breakdown signal indices

        # Initialize swing point detector (associate with SwingPointDetector class)
        self.swing_detector = SwingPointDetector()

    def _download_yahoo_finance(self) -> None:
        """
        Helper Method: Download stock data from Yahoo Finance (default data source, supports large-cap tech stocks).
        Download results are stored in self.stock_data, with closing price and volume series extracted.
        """
        if not self.ticker or not self.start_date or not self.end_date:
            raise ValueError("Stock ticker, start date and end date cannot be empty. Please set them first.")

        # Download data from Yahoo Finance
        try:
            self.stock_data = yf.download(
                tickers=self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            # Extract closing price and volume series (reset index to ensure consistency in subsequent calculations)
            self.prices = self.stock_data["Close"].reset_index(drop=True)
            self.volume = self.stock_data["Volume"].reset_index(drop=True)
            print(f"Successfully downloaded {self.ticker} data from Yahoo Finance ({self.start_date} to {self.end_date})")
        except Exception as e:
            raise Exception(f"Failed to download data from Yahoo Finance: {str(e)}")

    def download_data(self, data_source: str = "yahoo") -> None:
        """
        Core Method: Users select data source to download stock data.
        :param data_source: Data source selection ("yahoo" = Yahoo Finance, default value, other sources can be extended later)
        :raise ValueError: Invalid data source
        """
        if data_source == "yahoo":
            self._download_yahoo_finance()
        else:
            raise ValueError(f"Invalid data source: {data_source}. Only 'yahoo' is supported currently.")

        # Update closing price series of swing point detector
        self.swing_detector.set_prices(self.prices)

    def fit_trend_lines(self) -> None:
        """
        Core Method: Fit uptrend/downtrend lines (linear regression, optimized for large-cap tech stocks).
        Uptrend line is fitted by swing lows, and downtrend line is fitted by swing highs.
        """
        if self.swing_detector.swing_highs == [] or self.swing_detector.swing_lows == []:
            raise ValueError("No swing points identified. Please call detect_swing_points() first.")

        # Extract swing high and swing low data
        sh_indices = np.array(self.swing_detector.swing_highs)
        sh_prices = np.array(self.swing_detector.get_swing_points()["swing_highs_prices"])
        sl_indices = np.array(self.swing_detector.swing_lows)
        sl_prices = np.array(self.swing_detector.get_swing_points()["swing_lows_prices"])

        n = len(self.prices)

        # Fit downtrend line (swing highs, linear regression)
        if len(sh_indices) >= 2:  # At least 2 points are required to fit a trend line
            slope_d, intercept_d, r_value_d, p_value_d, std_err_d = stats.linregress(sh_indices, sh_prices)
            self.downtrend_line = slope_d * np.arange(n) + intercept_d

        # Fit uptrend line (swing lows, linear regression)
        if len(sl_indices) >= 2:  # At least 2 points are required to fit a trend line
            slope_u, intercept_u, r_value_u, p_value_u, std_err_u = stats.linregress(sl_indices, sl_prices)
            self.uptrend_line = slope_u * np.arange(n) + intercept_u

        print(" Trend line fitting completed (uptrend line + downtrend line)")

    def detect_swing_points(self, swing_method: str = "fixed") -> None:
        """
        Core Method: Users select swing point detection method to identify swing points.
        :param swing_method: Swing point detection method ("fixed" = fixed window, "adaptive" = adaptive window, default "fixed")
        """
        if self.prices.empty:
            raise ValueError("Closing price series is empty. Please download valid stock data first.")

        # Call detect_swing_points method of swing point detector
        self.swing_detector.detect_swing_points(method=swing_method)
        print(f" Swing point detection completed (method: {swing_method}). Identified {len(self.swing_detector.swing_highs)} highs and {len(self.swing_detector.swing_lows)} lows.")

    def _detect_candidate_signals(self) -> None:
        """
        Helper Method: Detect candidate breakout/breakdown signals (closing price crosses trend line for 2 consecutive days).
        Results are stored in self.breakout_indices and self.breakdown_indices.
        """
        if self.downtrend_line is None or self.uptrend_line is None:
            raise ValueError("Trend lines are not fitted. Please call fit_trend_lines() first.")

        n = len(self.prices)
        breakout_temp = []
        breakdown_temp = []

        # Traverse closing price series to detect candidate signals
        for t in range(1, n):
            # Breakout signal: closing price is above downtrend line for 2 consecutive days
            if self.prices.iloc[t] > self.downtrend_line[t] and self.prices.iloc[t-1] > self.downtrend_line[t-1]:
                breakout_temp.append(t)

            # Breakdown signal: closing price is below uptrend line for 2 consecutive days
            if self.prices.iloc[t] < self.uptrend_line[t] and self.prices.iloc[t-1] < self.uptrend_line[t-1]:
                breakdown_temp.append(t)

        # Deduplicate and store final signal indices
        self.breakout_indices = list(set(breakout_temp))
        self.breakdown_indices = list(set(breakdown_temp))
        self.breakout_indices.sort()
        self.breakdown_indices.sort()

    def detect_signals(self) -> None:
        """
        Core Method: Detect candidate breakout/breakdown signals (unified public interface exposed externally).
        """
        self._detect_candidate_signals()
        print(f"Candidate signal detection completed. Identified {len(self.breakout_indices)} breakout signals and {len(self.breakdown_indices)} breakdown signals.")

    def get_analysis_result(self) -> dict:
        """
        Public Method: Return complete analysis results.
        :return: Dictionary containing data, swing points, trend lines and signals
        """
        return {
            "data": {
                "ticker": self.ticker,
                "prices": self.prices,
                "volume": self.volume
            },
            "swing_points": self.swing_detector.get_swing_points(),
            "trendlines": {
                "uptrend_line": self.uptrend_line,
                "downtrend_line": self.downtrend_line
            },
            "signals": {
                "breakout_indices": self.breakout_indices,
                "breakdown_indices": self.breakdown_indices
            }
        }