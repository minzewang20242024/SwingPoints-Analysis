# Task1: AAPL股票摆动点检测代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 修正数据转换：通过.values获取数值数组后转列表
df = yf.download("AAPL", start="2025-01-01", end="2025-06-01")
close_price = df["Close"].values.tolist()  # 修正转换写法
price_dates = df.index.tolist()

def detect_swing_points(price, window=2):
    swing_highs = []
    swing_lows = []
    for i in range(window, len(price)-window):
        prev_prices = [price[i-j] for j in range(1, window+1)]
        next_prices = [price[i+j] for j in range(1, window+1)]
        is_swing_high = (price[i] > max(prev_prices)) and (price[i] > max(next_prices))
        is_swing_low = (price[i] < min(prev_prices)) and (price[i] < min(next_prices))
        if is_swing_high:
            swing_highs.append(i)
        if is_swing_low:
            swing_lows.append(i)
    return swing_highs, swing_lows

sh_indices, sl_indices = detect_swing_points(close_price)

# 提取摆动点的价格和日期
sh_prices = [close_price[i] for i in sh_indices]
sh_dates = [price_dates[i] for i in sh_indices]
sl_prices = [close_price[i] for i in sl_indices]
sl_dates = [price_dates[i] for i in sl_indices]

# 绘图
plt.rcParams['font.sans-serif'] = ['Arial']
plt.figure(figsize=(10, 6))
plt.plot(price_dates, close_price, color="blue", label="AAPL Close Price")
plt.scatter(sh_dates, sh_prices, color="red", s=50, label="Swing High (SH)")
plt.scatter(sl_dates, sl_prices, color="green", s=50, label="Swing Low (SL)")
plt.title("AAPL Swing Points Detection (Jan-Jun 2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)

plt.savefig("/Users/xavier/Desktop/swing_points.png", dpi=150, bbox_inches="tight")
plt.show()

print("=== 摆动点检测结果 ===")
print(f"摆动高点（SH）数量：{len(sh_indices)}")
print(f"摆动低点（SL）数量：{len(sl_indices)}")