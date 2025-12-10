import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# -------------------------- 1. 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# -------------------------- 2. 下载股票数据 --------------------------
def get_stock_data(ticker, start_date=None, end_date=None):
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    print(f"✅ 下载 {ticker}: {start_date} → {end_date}, 共 {len(df)} 条数据")
    return df

# -------------------------- 3. 找摆动点 --------------------------
def find_swing_points(prices, window=5):
    highs, lows = [], []
    for i in range(window, len(prices)-window):
        if prices[i] == max(prices[i-window:i+window+1]):
            highs.append(i)
        if prices[i] == min(prices[i-window:i+window+1]):
            lows.append(i)
    return highs, lows

# -------------------------- 4. 绘制趋势线 --------------------------
def fit_trendline(indices, prices):
    """拟合趋势线，基于真实价格 index 进行线性回归"""
    if len(indices) < 2:
        return None

    x = np.array(indices)
    y = prices[indices]

    # 拟合线性方程 y = ax + b
    a, b = np.polyfit(x, y, 1)

    # 生成整个区间的趋势线
    full_x = np.arange(len(prices))
    full_y = a * full_x + b
    return full_y

def plot_trend_lines(stock_data, ticker):
    dates = stock_data['Date']
    prices = stock_data['Close'].values

    # 找摆动点
    swing_highs, swing_lows = find_swing_points(prices, window=5)

    # 画价格
    plt.figure()
    plt.plot(dates, prices, label='Close Price', color='blue')

    # 摆动点
    plt.scatter(dates[swing_highs], prices[swing_highs], color='red', s=50, label='Swing Highs')
    plt.scatter(dates[swing_lows], prices[swing_lows], color='green', s=50, label='Swing Lows')

    # 上升趋势线（低点）
    up_trend = fit_trendline(swing_lows, prices)
    if up_trend is not None:
        plt.plot(dates, up_trend, color='green', linestyle='--', linewidth=2, label='Uptrend Line')

    # 下降趋势线（高点）
    down_trend = fit_trendline(swing_highs, prices)
    if down_trend is not None:
        plt.plot(dates, down_trend, color='red', linestyle='--', linewidth=2, label='Downtrend Line')

    # 图形设置
    plt.title(f"{ticker} Stock with Trendlines (Last Year)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{ticker}_trendlines.png", dpi=300)
    print(f"📁 已保存图像: {ticker}_trendlines.png")

    plt.show()

# -------------------------- 5. 主函数 --------------------------
def main():
    ticker = "AAPL"
    df = get_stock_data(ticker, "2024-01-01", "2024-12-31")
    plot_trend_lines(df, ticker)

# -------------------------- 6. 运行 --------------------------
if __name__ == "__main__":
    main()
