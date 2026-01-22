import pandas as pd
import numpy as np
import yfinance as yf

class DataFetcher:
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)  

    def fetch_yahoo_stock(self, ticker: str, start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
        
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        clean_data = pd.DataFrame({
            "Date": stock_data.index,
            "Close": stock_data["Close"]
        }).dropna().reset_index(drop=True)
        return clean_data

    def fetch_local_csv(self, csv_path: str) -> pd.DataFrame:
        
        csv_data = pd.read_csv(csv_path)
        
        clean_data = csv_data[["Date", "Close"]].dropna().reset_index(drop=True)
        clean_data["Date"] = pd.to_datetime(clean_data["Date"])  
        return clean_data

    def fetch_random_walk(self, n_points: int = 250, initial_price: float = 100.0, volatility: float = 0.5) -> pd.DataFrame:
        
        returns = np.random.normal(0, volatility, n_points)
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] + ret)
        
        random_data = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=n_points),
            "Close": prices
        })
        return random_data