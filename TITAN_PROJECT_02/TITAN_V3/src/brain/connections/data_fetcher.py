import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

# Add project root to path
# Path: achilles_trading_bot/src/brain/connections/data_fetcher.py
# Root is 4 directories up
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import settings

class DataFetcher:
    def __init__(self, symbol=settings.SYMBOL):
        self.symbol = symbol
        self.data_dir = settings.RAW_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_yahoo_data(self, period="1y", interval="1h"):
        """
        Fetches data from Yahoo Finance as a baseline.
        Note: Yahoo Finance symbols for Gold is 'GC=F' or similar, XAUUSD=X often used.
        """
        yf_symbol = "GC=F" # Future Gold
        if self.symbol == "XAUUSD":
             yf_symbol = "GC=F" # Fallback mapping
        
        print(f"Fetching {period} of {interval} data for {yf_symbol}...")
        try:
            df = yf.download(tickers=yf_symbol, period=period, interval=interval)
            
            if df.empty:
                print("Warning: Downloaded data is empty.")
                return None
                
            # Save to CSV
            filename = f"{self.symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.fetch_yahoo_data(period="1mo", interval="1h")
