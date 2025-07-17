import yfinance as yf
import os

def fetch_spy(
    ticker="SPY", 
    start="2010-01-01", 
    end=None, 
    interval="1d", 
    save_path="data/raw/spy.csv"
):
    print(f"Fetching {ticker} from {start} to {end or 'today'}...")
    df = yf.download(ticker, start=start, end=end, interval=interval)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"✅ Saved to {save_path}")
    return df


if __name__ == "__main__":
    fetch_spy()
