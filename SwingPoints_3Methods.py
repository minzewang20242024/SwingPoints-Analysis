# 第一步：导入需要的工具包（小白理解：像借工具干活）
import pandas as pd  # 处理数据
import numpy as np   # 计算数值
import matplotlib.pyplot as plt  # 画图
import yfinance as yf  # 下载股票数据
from scipy.signal import argrelextrema  # 找高低点的工具包

# 第二步：下载股票数据（AAPL 2025年1-6月收盘价）
# 不用改，直接运行就能下载
df = yf.download("AAPL", start="2025-01-01", end="2025-06-01")
prices = df["Close"].values  # 提取价格数值
dates = df.index  # 提取日期（用于画图）
window = 2  # 对比前后2个价格，不用改

# 第三步：定义3种检测方法（核心功能）
# 方法1：窗口对比法（新手基础版）
def detect_method1(prices, window=2):
    sh_list = []  # 存摆动高点的位置
    sl_list = []  # 存摆动低点的位置
    # 遍历每个价格（跳过前后2个，避免越界）
    for i in range(window, len(prices)-window):
        # 判断摆动高点：当前价 > 前2个价的最大值 且 > 后2个价的最大值
        if prices[i] > np.max(prices[i-window:i]) and prices[i] > np.max(prices[i+1:i+window+1]):
            sh_list.append(i)
        # 判断摆动低点：当前价 < 前2个价的最小值 且 < 后2个价的最小值
        if prices[i] < np.min(prices[i-window:i]) and prices[i] < np.min(prices[i+1:i+window+1]):
            sl_list.append(i)
    return sh_list, sl_list

# 方法2：价格斜率法（看涨跌反转）
def detect_method2(prices, min_change=0.0, min_distance=1):
    """
    Detect swing highs and lows using slope reversal.
    This version is SAFE against NumPy array truth-value errors.
    """
    prices = np.asarray(prices, dtype=float).flatten()  # 强制转为一维 float 数组

    diff = np.diff(prices)  # 一维数组
    sh_list = []
    sl_list = []

    if len(diff) < 2:
        return sh_list, sl_list

    last_sh = -9999
    last_sl = -9999

    for i in range(1, len(diff)):
        prev_slope = float(diff[i - 1])  # 强制转换为 Python 标量
        curr_slope = float(diff[i])

        price_index = i

        # Swing High
        if prev_slope > 0 and curr_slope <= 0:
            if abs(prices[price_index] - prices[price_index - 1]) >= min_change:
                if price_index - last_sh >= min_distance:
                    sh_list.append(price_index)
                    last_sh = price_index

        # Swing Low
        if prev_slope < 0 and curr_slope >= 0:
            if abs(prices[price_index] - prices[price_index - 1]) >= min_change:
                if price_index - last_sl >= min_distance:
                    sl_list.append(price_index)
                    last_sl = price_index

    return sh_list, sl_list


# 方法3：Scipy工具法（便捷版，和方法1逻辑一样）
def detect_method3(prices, order=2):
    # 找局部高点（SH）和低点（SL）
    sh_list = argrelextrema(prices, np.greater, order=order)[0].tolist()
    sl_list = argrelextrema(prices, np.less, order=order)[0].tolist()
    return sh_list, sl_list

# 第四步：运行3种方法，得到结果
sh1, sl1 = detect_method1(prices, window)  # 方法1结果
sh2, sl2 = detect_method2(prices)          # 方法2结果
sh3, sl3 = detect_method3(prices, window)  # 方法3结果

# 第五步：画图（3个子图，对应3种方法）
# 使用Mac系统支持中文的字体
plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示异常
# 创建3行1列的图，大小适配屏幕
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
# 总标题
fig.suptitle("AAPL摆动点检测（3种方法）| 2025.1-6", fontsize=12, y=0.95)

# 画方法1的图
axes[0].plot(dates, prices, color="blue", label="收盘价")
axes[0].scatter(dates[sh1], prices[sh1], color="red", s=50, label="摆动高点")
axes[0].scatter(dates[sl1], prices[sl1], color="green", s=50, label="摆动低点")
axes[0].set_title("方法1：窗口对比法")
axes[0].set_ylabel("价格（美元）")
axes[0].legend()
axes[0].grid(alpha=0.3)  # 浅灰色网格，方便看

# 画方法2的图
axes[1].plot(dates, prices, color="blue", label="收盘价")
axes[1].scatter(dates[sh2], prices[sh2], color="red", s=50, label="摆动高点")
axes[1].scatter(dates[sl2], prices[sl2], color="green", s=50, label="摆动低点")
axes[1].set_title("方法2：价格斜率法")
axes[1].set_ylabel("价格（美元）")
axes[1].legend()
axes[1].grid(alpha=0.3)

# 画方法3的图
axes[2].plot(dates, prices, color="blue", label="收盘价")
axes[2].scatter(dates[sh3], prices[sh3], color="red", s=50, label="摆动高点")
axes[2].scatter(dates[sl3], prices[sl3], color="green", s=50, label="摆动低点")
axes[2].set_title("方法3：Scipy工具法")
axes[2].set_xlabel("日期")
axes[2].set_ylabel("价格（美元）")
axes[2].legend()
axes[2].grid(alpha=0.3)

# 旋转日期标签，避免重叠
plt.xticks(rotation=45)
# 保存图片到桌面（改这里：把xavier换成你的Mac用户名，比如你的用户名是xiaoming就改/Users/xiaoming/Desktop/）
plt.savefig("/Users/xavier/Desktop/SwingPoints_3Methods.png", dpi=150, bbox_inches="tight")
plt.show()

# 第六步：打印结果（方便填报告）
print("=== 3种方法检测结果 ===")
print(f"方法1（窗口对比）：摆动高点={len(sh1)}，摆动低点={len(sl1)}")
print(f"方法2（价格斜率）：摆动高点={len(sh2)}，摆动低点={len(sl2)}")
print(f"方法3（Scipy工具）：摆动高点={len(sh3)}，摆动低点={len(sl3)}")