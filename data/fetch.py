import yfinance as yf
import os
from datetime import date


def fetch_yfinance(
    ticker="SPY",
    start="2010-01-01",
    end=None,
    interval="1d",
    save_path="data/raw/spy.csv",
    overwrite=False
):
    end = end or str(date.today())

    if os.path.exists(save_path) and not overwrite:
        print(f"📄 {save_path} already exists. Use overwrite=True to refetch.")
        return

    print(f"⬇️ Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, interval=interval, multi_level_index = False)

    if df.empty:
        raise ValueError(f"❌ No data returned for {ticker} in given date range.")

    df = df.reset_index()

    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[expected_cols]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {len(df)} rows to {save_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    fetch_yfinance()
