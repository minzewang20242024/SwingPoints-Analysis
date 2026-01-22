# ============================================================
#   STOCK ANALYSIS PROJECT — Swing Points, Trendlines,
#   Breakout/Breakdown Detection, Multi-Source Support
#   Author: Minze Wang
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

# Breakout confirmation days
CONSECUTIVE_DAYS = 2


# ============================================================
#               MULTIPLE DATA SOURCE FUNCTIONS
# ============================================================
def get_stock_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
    """Download stock data from Yahoo Finance."""
    stock = yf.download(ticker, start=start_date, end=end_date,
                        auto_adjust=False, progress=False)
    stock.reset_index(inplace=True)
    stock["Date"] = pd.to_datetime(stock["Date"])
    return stock


def get_data_csv(path):
    """Load stock price data from local CSV file."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_data_random(n=250):
    """Generate synthetic random-walk price data."""
    dates = pd.date_range(end=datetime.today(), periods=n)
    prices = np.cumsum(np.random.randn(n)) + 150
    df = pd.DataFrame({"Date": dates, "Close": prices})
    return df


# ============================================================
#               SWING POINT DETECTION METHODS
# ============================================================
def find_swing_points_sliding_window(prices, window=SWING_WINDOW):
    swing_highs = []
    swing_lows = []

    for i in range(window, len(prices) - window):
        window_slice = prices.iloc[i - window: i + window + 1]

        # Compare floats using .item() to avoid ambiguity warnings
        if prices.iloc[i].item() == window_slice.max().item():
            swing_highs.append(i)

        if prices.iloc[i].item() == window_slice.min().item():
            swing_lows.append(i)

    return swing_highs, swing_lows


def find_swing_points_slope_reversal(prices, min_change=MIN_CHANGE, min_distance=MIN_DISTANCE):
    diffs = prices.diff().dropna()
    swing_highs, swing_lows = [], []
    last_high, last_low = -min_distance, -min_distance

    for i in range(1, len(diffs)):
        if diffs.iloc[i - 1] > min_change and diffs.iloc[i] <= 0:
            if i - last_high >= min_distance:
                swing_highs.append(i)
                last_high = i

        if diffs.iloc[i - 1] < -min_change and diffs.iloc[i] >= 0:
            if i - last_low >= min_distance:
                swing_lows.append(i)
                last_low = i

    return swing_highs, swing_lows


def find_swing_points_scipy(prices, order=ORDER):
    values = prices.values
    highs = argrelextrema(values, np.greater, order=order)[0]
    lows = argrelextrema(values, np.less, order=order)[0]
    return list(highs), list(lows)


# ============================================================
#               TRENDLINE FITTING FUNCTION
# ============================================================
def fit_trendline(dates, prices, swing_indices):
    swing_prices = prices.iloc[swing_indices]
    numeric_dates = np.arange(len(dates))
    swing_numeric = numeric_dates[swing_indices]

    slope, intercept = np.polyfit(swing_numeric, swing_prices, 1)
    trendline = slope * numeric_dates + intercept
    return slope, intercept, trendline


# ============================================================
#               BREAKOUT / BREAKDOWN DETECTION
# ============================================================
def detect_breakout_breakdown(close_prices, up_line, down_line):
    breakout, breakdown = [], []

    for i in range(CONSECUTIVE_DAYS, len(close_prices)):

        # Breakout detection
        if all(
            close_prices.iloc[i - k].item() > float(down_line[i - k])
            for k in range(CONSECUTIVE_DAYS)
        ):
            breakout.append(i)

        # Breakdown detection
        if all(
            close_prices.iloc[i - k].item() < float(up_line[i - k])
            for k in range(CONSECUTIVE_DAYS)
        ):
            breakdown.append(i)

    return breakout, breakdown


# ============================================================
#               OPTIONAL PLOTTING FUNCTION
# ============================================================
def plot_stock_data_with_trend_breakpoints(stock_data, save_path="chart.png"):
    dates = stock_data["Date"]
    close = stock_data["Close"]

    highs, lows = find_swing_points_sliding_window(close)

    up_tr, down_tr = np.zeros(len(close)), np.zeros(len(close))

    if len(lows) >= 2:
        _, _, up_tr = fit_trendline(dates, close, lows)
    if len(highs) >= 2:
        _, _, down_tr = fit_trendline(dates, close, highs)

    breakout, breakdown = detect_breakout_breakdown(close, up_tr, down_tr)

    plt.figure(figsize=FIGSIZE)
    plt.plot(dates, close, label="Close Price", color="blue")
    plt.scatter(dates.iloc[highs], close.iloc[highs], color="red", label="Swing Highs")
    plt.scatter(dates.iloc[lows], close.iloc[lows], color="green", label="Swing Lows")
    plt.plot(dates, up_tr, "--", color=UP_TREND_COLOR, label="Uptrend")
    plt.plot(dates, down_tr, "--", color=DOWN_TREND_COLOR, label="Downtrend")

    plt.scatter(dates.iloc[breakout], close.iloc[breakout], marker="*", color="orange", s=200)
    plt.scatter(dates.iloc[breakdown], close.iloc[breakdown], marker="v", color="purple", s=150)

    plt.title("Trendlines & Breakout/Breakdown Signals")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ============================================================
#               MAIN EXECUTION — ANALYSIS PIPELINE
# ============================================================
if __name__ == "__main__":

    datasets = {
        "AAPL (Yahoo)": get_stock_data(),
        "Random Data": get_data_random(),
        # Add CSV example if needed:
        # "CSV Data": get_data_csv("yourfile.csv")
    }

    print("\n================ MULTI-DATASET ANALYSIS ================\n")

    for name, df in datasets.items():
        print(f"\n----- Analyzing {name} -----")

        close = df["Close"]
        dates = df["Date"]

        # ✦ Step 1: swing points
        highs, lows = find_swing_points_sliding_window(close)

        # ✦ Step 2: trendlines
        _, _, up_line = fit_trendline(dates, close, lows) if len(lows) >= 2 else (0, 0, np.zeros(len(close)))
        _, _, down_line = fit_trendline(dates, close, highs) if len(highs) >= 2 else (0, 0, np.zeros(len(close)))

        # ✦ Step 3: breakout/breakdown detection
        breakout, breakdown = detect_breakout_breakdown(close, up_line, down_line)

        # ✦ Print results
        print(f"Swing Highs: {len(highs)}")
        print(f"Swing Lows:  {len(lows)}")
        print(f"Breakouts:   {len(breakout)}")
        print(f"Breakdowns:  {len(breakdown)}")

    print("\n================ ANALYSIS COMPLETED ==================\n")
