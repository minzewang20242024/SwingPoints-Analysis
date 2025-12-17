# ============================================================
#   STOCK ANALYSIS PROJECT — Swing Points, Trendlines,
#   Breakout/Breakdown Detection, Full Visualization
#   Author: YOUR NAME
#   Last Updated: 2025
# ============================================================

# --------------------- Import Libraries ----------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# ---------------------- Global Parameters --------------------
# Stock-related parameters
TICKER = "AAPL"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

# Swing detection parameters
SWING_WINDOW = 5
MIN_CHANGE = 0.1
MIN_DISTANCE = 3
ORDER = 5

# Visualization parameters
FIGSIZE = (12, 8)
UP_TREND_COLOR = "green"
DOWN_TREND_COLOR = "red"
BREAKOUT_COLOR = "orange"
BREAKDOWN_COLOR = "purple"

# Breakout confirmation
CONSECUTIVE_DAYS = 2


# ============================================================
#               DATA ACQUISITION FUNCTION
# ============================================================
def get_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
    """
    Download historical stock data from Yahoo Finance.
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    return stock_data


# ============================================================
#               SWING POINT DETECTION METHODS
# ============================================================
def find_swing_points_sliding_window(prices, window=SWING_WINDOW):
    """
    Method 1: Sliding window swing point detection.
    Detects swing highs (local maxima) and swing lows (local minima).
    Fixes Series truth-value ambiguity by comparing float values.
    """
    swing_highs = []
    swing_lows = []

    for i in range(window, len(prices) - window):
        window_slice = prices.iloc[i - window : i + window + 1]

        # Convert both sides to float → prevents ambiguous truth value errors
        if float(prices.iloc[i]) == float(window_slice.max()):
            swing_highs.append(i)

        if float(prices.iloc[i]) == float(window_slice.min()):
            swing_lows.append(i)

    return swing_highs, swing_lows



def find_swing_points_slope_reversal(prices, min_change=MIN_CHANGE, min_distance=MIN_DISTANCE):
    """
    Method 2: Slope-Reversal (first derivative sign change).
    Works in real time (does not use future data).
    """
    diffs = prices.diff().dropna()
    swing_highs = []
    swing_lows = []
    last_high = -min_distance
    last_low = -min_distance

    for i in range(1, len(diffs)):
        # Swing High
        if diffs.iloc[i - 1] > min_change and diffs.iloc[i] <= 0:
            if i - last_high >= min_distance:
                swing_highs.append(i)
                last_high = i

        # Swing Low
        if diffs.iloc[i - 1] < -min_change and diffs.iloc[i] >= 0:
            if i - last_low >= min_distance:
                swing_lows.append(i)
                last_low = i

    return swing_highs, swing_lows


def find_swing_points_scipy(prices, order=ORDER):
    """
    Method 3: SciPy argrelextrema (local maxima/minima detection).
    """
    values = prices.values
    highs = argrelextrema(values, np.greater, order=order)[0]
    lows = argrelextrema(values, np.less, order=order)[0]
    return list(highs), list(lows)


# ============================================================
#               TRENDLINE FITTING FUNCTION
# ============================================================
def fit_trendline(dates, prices, swing_indices):
    """
    Fit trendline using linear regression on swing points.
    """
    swing_prices = prices.iloc[swing_indices]
    dates_numeric = np.arange(len(dates))
    swing_dates_numeric = dates_numeric[swing_indices]

    slope, intercept = np.polyfit(swing_dates_numeric, swing_prices, 1)
    trendline = slope * dates_numeric + intercept

    return slope, intercept, trendline


# ============================================================
#               BREAKOUT / BREAKDOWN DETECTION
# ============================================================
def detect_breakout_breakdown(close_prices, up_trendline, down_trendline):
    """
    Detect breakout (price > downtrend line) and breakdown (price < uptrend line).
    Converts comparison values to float to avoid ambiguous Series truth-value errors.
    """
    breakout = []
    breakdown = []

    for i in range(CONSECUTIVE_DAYS, len(close_prices)):

        # --- Breakout (price crosses above downtrend line)
        if all(
            float(close_prices.iloc[i - k]) > float(down_trendline[i - k])
            for k in range(CONSECUTIVE_DAYS)
        ):
            breakout.append(i)

        # --- Breakdown (price crosses below uptrend line)
        if all(
            float(close_prices.iloc[i - k]) < float(up_trendline[i - k])
            for k in range(CONSECUTIVE_DAYS)
        ):
            breakdown.append(i)

    return breakout, breakdown



# ============================================================
#               VISUALIZATION FUNCTION
# ============================================================
def plot_stock_data_with_trend_breakpoints(stock_data):
    """
    Plot swing points, trendlines, breakout/breakdown points.
    """
    dates = stock_data["Date"]
    close = stock_data["Close"]

    # Step 1: Swing points
    swing_highs, swing_lows = find_swing_points_sliding_window(close)

    # Step 2: Trendlines
    up_trendline = np.zeros(len(close))
    down_trendline = np.zeros(len(close))

    if len(swing_lows) >= 2:
        _, _, up_trendline = fit_trendline(dates, close, swing_lows)
    if len(swing_highs) >= 2:
        _, _, down_trendline = fit_trendline(dates, close, swing_highs)

    # Step 3: Breakout / Breakdown
    breakout, breakdown = detect_breakout_breakdown(close, up_trendline, down_trendline)

    # Step 4: Plot everything
    plt.figure(figsize=FIGSIZE)

    plt.plot(dates, close, label="Close Price", color="blue")
    plt.scatter(dates.iloc[swing_highs], close.iloc[swing_highs], color="red", label="Swing Highs")
    plt.scatter(dates.iloc[swing_lows], close.iloc[swing_lows], color="green", label="Swing Lows")

    plt.plot(dates, up_trendline, linestyle="--", color=UP_TREND_COLOR, label="Uptrend Line")
    plt.plot(dates, down_trendline, linestyle="--", color=DOWN_TREND_COLOR, label="Downtrend Line")

    plt.scatter(dates.iloc[breakout], close.iloc[breakout], marker="*", color=BREAKOUT_COLOR, s=200, label="Breakout Points")
    plt.scatter(dates.iloc[breakdown], close.iloc[breakdown], marker="v", color=BREAKDOWN_COLOR, s=150, label="Breakdown Points")

    plt.title(f"{TICKER} Trendlines and Breakout/Breakdown Points (2024)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("AAPL_trend_breakpoints.png")
    plt.show()

    print("\n===== BREAKOUT/BREAKDOWN SUMMARY =====")
    print("Breakouts:", len(breakout))
    print("Breakdown:", len(breakdown))


# ============================================================
#               MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("Downloading stock data...")
    stock_data = get_stock_data()

    print("Generating chart with trendlines and breakout/breakdown points...")
    plot_stock_data_with_trend_breakpoints(stock_data)

    print("Analysis completed successfully!")
