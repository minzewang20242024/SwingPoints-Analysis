import pandas as pd
import numpy as np

class SwingPointDetector:
    """
    Swing Point Detection Class (Optimized for Large-Cap Tech Stocks)
    Functionality: Identify swing highs and swing lows of stocks using either fixed window method 
    or volatility-adaptive window method.
    Core data is initialized via __init__, and users can select calculation methods.
    """

    def __init__(self, prices: pd.Series = None, base_window: int = 5, min_distance: int = 3, vol_window: int = 20):
        """
        Class Initialization: Initialize core data and parameters.
        :param prices: Stock closing price series (Pandas Series, index as dates, values as closing prices)
        :param base_window: Base window size (default 5, optimized value for large-cap tech stocks after justification)
        :param min_distance: Minimum distance between adjacent swing points (default 3, filter duplicate noise points)
        :param vol_window: Rolling volatility calculation window (default 20, calculate short-term volatility)
        """
        # Core data initialization (passed via __init__ for better encapsulation)
        self.prices = prices if prices is not None else pd.Series()
        self.swing_highs = []  # Store indices of finally identified swing highs
        self.swing_lows = []   # Store indices of finally identified swing lows

        # Core parameter initialization (default values justified for large-cap tech stocks)
        self.base_window = base_window
        self.min_distance = min_distance
        self.vol_window = vol_window
        self.adaptive_windows = None  # Store adaptive window sizes (calculated dynamically)

    def _filter_by_min_distance(self, swing_points: np.ndarray) -> list:
        """
        Helper Method: Filter swing points by minimum distance to remove overly close duplicate points.
        :param swing_points: Numpy array of preliminarily identified swing point indices
        :return: List of valid swing point indices after filtering
        """
        if len(swing_points) == 0:
            return []

        # Initialize filtered results and retain the first swing point
        filtered = [swing_points[0]]
        for point in swing_points[1:]:
            # Judge if the distance between current point and the last retained point meets the minimum distance requirement
            if point - filtered[-1] >= self.min_distance:
                filtered.append(point)

        return filtered

    def _calculate_adaptive_window(self) -> None:
        """
        Helper Method: Calculate volatility-adaptive window sizes (optimized for large-cap tech stocks).
        Results are stored in self.adaptive_windows, with window sizes constrained between 3 and 10.
        """
        if self.prices.empty:
            raise ValueError("Closing price series is empty. Please set valid data via set_prices() first.")

        # Calculate 20-day rolling volatility (sample standard deviation, suitable for small sample calculation)
        rolling_vol = self.prices.rolling(window=self.vol_window).std()
        avg_vol = rolling_vol.mean()  # Calculate global average volatility (as benchmark for window adjustment)

        # Dynamically calculate adaptive window size, constrained between 3 and 10
        adaptive_window = np.maximum(
            3,
            np.minimum(
                10,
                np.round(self.base_window * (rolling_vol / avg_vol))
            )
        ).astype(int)

        # Fill the first vol_window values (volatility not calculated, use base window)
        adaptive_window[:self.vol_window] = self.base_window
        self.adaptive_windows = adaptive_window

    def detect_swing_points(self, method: str = "fixed") -> None:
        """
        Core Method: Users select calculation method to identify swing highs and swing lows.
        :param method: Calculation method selection ("fixed" = fixed window method, "adaptive" = adaptive window method, default "fixed")
        :raise ValueError: Invalid calculation method or empty closing price series
        """
        if self.prices.empty:
            raise ValueError("Closing price series is empty. Please set valid data via set_prices() first.")

        n = len(self.prices)
        if n < 2 * self.base_window:
            raise ValueError("Closing price series is too short to meet window calculation requirements.")

        # Execute corresponding swing point detection according to user-selected method
        if method == "fixed":
            self._detect_fixed_window()
        elif method == "adaptive":
            self._detect_adaptive_window()
        else:
            raise ValueError(f"Invalid calculation method: {method}. Supported methods are 'fixed' and 'adaptive'.")

    def _detect_fixed_window(self) -> None:
        """
        Internal Method: Identify swing points using fixed window method (benchmark method, optimized for large-cap tech stocks).
        Window size is self.base_window (default 5), identifying local extremum points.
        """
        w = self.base_window
        n = len(self.prices)
        swing_highs_temp = []
        swing_lows_temp = []

        # Traverse closing price series, exclude edge data (cannot meet front and rear window requirements)
        for t in range(w, n - w):
            # Extract window data of w days before and after the current point
            window = self.prices.iloc[t - w: t + w + 1]
            current_price = self.prices.iloc[t]

            # Judge if it is a swing high (maximum value in the window)
            if current_price == window.max():
                swing_highs_temp.append(t)

            # Judge if it is a swing low (minimum value in the window)
            if current_price == window.min():
                swing_lows_temp.append(t)

        # Filter duplicate noise points and store final results
        self.swing_highs = self._filter_by_min_distance(np.array(swing_highs_temp))
        self.swing_lows = self._filter_by_min_distance(np.array(swing_lows_temp))

    def _detect_adaptive_window(self) -> None:
        """
        Internal Method: Identify swing points using volatility-adaptive window method (optimized method for high-volatility periods of large-cap tech stocks).
        Window size is adjusted dynamically, constrained between 3 and 10, to improve identification accuracy in high-volatility periods.
        """
        # First calculate adaptive window sizes
        self._calculate_adaptive_window()
        n = len(self.prices)
        swing_highs_temp = []
        swing_lows_temp = []

        # Traverse closing price series, exclude edge data (first vol_window and last base_window points)
        for t in range(self.vol_window, n - self.base_window):
            w = self.adaptive_windows[t]
            # Ensure the window does not exceed the series boundary
            if t - w < 0 or t + w >= n:
                continue

            # Extract window data of w days before and after the current point
            window = self.prices.iloc[t - w: t + w + 1]
            current_price = self.prices.iloc[t]

            # Judge if it is a swing high (maximum value in the window)
            if current_price == window.max():
                swing_highs_temp.append(t)

            # Judge if it is a swing low (minimum value in the window)
            if current_price == window.min():
                swing_lows_temp.append(t)

        # Filter duplicate noise points and store final results
        self.swing_highs = self._filter_by_min_distance(np.array(swing_highs_temp))
        self.swing_lows = self._filter_by_min_distance(np.array(swing_lows_temp))

    def set_prices(self, prices: pd.Series) -> None:
        """
        Public Method: Update closing price series (no need to reinitialize the class for subsequent re-detection).
        :param prices: New stock closing price series (Pandas Series, index as dates, values as closing prices)
        """
        self.prices = prices
        self.swing_highs = []
        self.swing_lows = []
        self.adaptive_windows = None

    def get_swing_points(self) -> dict:
        """
        Public Method: Return identified swing point results.
        :return: Dictionary containing swing high/low indices and corresponding closing prices
        """
        return {
            "swing_highs_indices": self.swing_highs,
            "swing_highs_prices": self.prices.iloc[self.swing_highs].tolist() if self.swing_highs else [],
            "swing_lows_indices": self.swing_lows,
            "swing_lows_prices": self.prices.iloc[self.swing_lows].tolist() if self.swing_lows else []
        }