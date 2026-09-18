"""
Download QQQ adjusted price data via yfinance (auto_adjust=True).
Saves as QQQ.csv in the same directory, replacing the stooq version.

Run: python fetch_qqq_yfinance.py
"""

import yfinance as yf
import pandas as pd

START_DATE = "2014-01-01"
END_DATE   = "2023-12-31"

print("Downloading QQQ (dividend-adjusted) via yfinance ...")
raw = yf.download("QQQ", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

if raw.empty:
    raise RuntimeError("Download failed — still rate limited. Wait a few minutes and retry.")

# Flatten MultiIndex columns (yfinance >= 0.2 may produce these)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

# Reshape to match the format expected by qqq_investment_analysis.py:
# columns: symbol, date, open, close, high, low, volume
df = raw.copy()
df.index.name = "date"
df.columns = [c.lower() for c in df.columns]          # Open→open, Close→close …
df.insert(0, "symbol", "QQQ")
df = df[["symbol", "open", "close", "high", "low", "volume"]]
df = df.sort_index()

out_path = "QQQ.csv"
df.to_csv(out_path)

print(f"Saved {len(df)} rows → {out_path}")
print(f"Start close : {df['close'].iloc[0]:.4f}")
print(f"End close   : {df['close'].iloc[-1]:.4f}")
print("\nFirst 3 rows:")
print(df.head(3))
