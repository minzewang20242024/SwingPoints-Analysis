# SwingPoints-Analysis

# SwingPoints-Analysis

This repository contains **code, reports, and visualisations** for identifying **swing points**, fitting **trendlines**, and detecting **breakout / breakdown signals** in financial time series (e.g., AAPL, MSFT, GOOGL from Yahoo Finance).

The project focuses on:
1. Identifying swing highs and swing lows  
2. Constructing uptrend and downtrend lines from swing points  
3. Detecting breakout and breakdown signals  
4. Evaluating **true / false breakout and breakdown signals** using quantitative rules  

This repository is designed for **research, reproducibility, and extension**, rather than short-term trading automation.

---

## 1. Project Objectives

The main objectives of this project are:

- To study and compare **different swing point identification methods**
- To construct **trendlines only from swing points**, not from all data points
- To detect **candidate breakout and breakdown signals**
- To **classify signals as true or false** using outcome-based verification
- To evaluate signal quality using **classification metrics** such as Precision, Recall, and F1-score

---

## 2. Methods Overview

### 2.1 Swing Point Identification

Swing points are the foundation of trendline construction.

A **Swing High** is a local maximum price point.  
A **Swing Low** is a local minimum price point.

Implemented (or discussed) approaches include:

#### (1) Fixed Sliding Window Method (Baseline)
- Use a symmetric window of size `w` (e.g., 5 days)
- A point is a swing high if it is the maximum within `[t-w, …, t+w]`
- A point is a swing low if it is the minimum within `[t-w, …, t+w]`
- Optional filter: minimum distance between adjacent swing points

**Advantages:** simple, efficient, reproducible  
**Disadvantages:** poor adaptability to very high / low volatility assets

---

#### (2) Volatility-Adaptive Window Method
- Compute rolling volatility (e.g., 20-day standard deviation)
- Dynamically adjust window size:
  - High volatility → larger window (reduce noise)
  - Low volatility → smaller window (capture weak trends)

**Advantages:** adaptive to different assets  
**Disadvantages:** slightly higher computational complexity

---

#### (3) Slope Reversal Method (Optional)
- Based on sign changes of price differences
- Swing High: slope changes from positive to negative
- Swing Low: slope changes from negative to positive
- Requires amplitude threshold to avoid noise

---

### 2.2 Trendline Construction

- **Uptrend line**: fitted using swing lows  
- **Downtrend line**: fitted using swing highs  
- Linear regression (OLS) is applied to swing points only
- Trendlines are not fitted when insufficient swing points exist

---

### 2.3 Breakout / Breakdown Detection

Candidate signals are defined as:

- **Breakout**: closing price above the downtrend line for **2 consecutive days**
- **Breakdown**: closing price below the uptrend line for **2 consecutive days**

The number of confirmation days is configurable.

---

### 2.4 True / False Signal Classification

Candidate signals are evaluated using **future price outcomes**, not visual judgment.

Core parameters:
- **Holding Period (HP)**: e.g., 10 trading days
- **Minimum Threshold (MT)**: e.g., 3% price move

Optional enhancement:
- **Volume confirmation**:
  - Signal-day volume ≥ 1.5 × 20-day average volume

Signals are classified into:
- True Positive (TP)
- False Positive (FP)
- False Negative (FN)
- True Negative (TN)

---

## 3. Repository Structure

Typical structure:

main.py

stock_analysis/
  init.py
  analyzer.py # main analysis class
  data_sources.py # Yahoo / CSV / synthetic data loaders
  swing_points.py # swing point detection methods
  trendlines.py # trendline fitting (OLS)
  signals.py # breakout / breakdown detection
  evaluator.py # true/false signal evaluation
  plotting.py # visualisation utilities

reports/
  report.tex / report.pdf

data/
  example.csv

outputs/

figures/

results.xlsx



(Adjust names if your actual repo differs.)

---

## 4. Installation

### 4.1 Python Version
Recommended: **Python 3.10+**

### 4.2 Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```
### 4.3 Install Dependencies

pip install -r requirements.txt

### 5. Running the Project

python main.py --ticker AAPL --start 2024-01-01 --end 2024-12-31 --method adaptive --plot

Typical configurable parameters:

- ticker
- date range
- data source (Yahoo / CSV / synthetic)
- swing point method
- window size
- confirmation days
- holding period
- threshold
- plotting / export options

### 6. Outputs

- Swing point and trendline charts
- Breakout / breakdown markers
- Excel / CSV result tables
- Console summaries

### 7. Reproducibility Notes
- Synthetic data uses fixed random seeds
- Yahoo Finance data may change slightly over time
- All parameters are configurable for fair comparison