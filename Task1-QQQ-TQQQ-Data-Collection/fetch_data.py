import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import time

start_date = '2014-01-01'
end_date = '2024-01-01'

APIKEY = 'ADiIv4cTWsaetYMEz5KwUnNZl2JG0kx3'

def fetch_stooq(symbol, start, end, retries=3):
    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={symbol.lower()}.us"
        f"&d1={start.replace('-', '')}"
        f"&d2={end.replace('-', '')}"
        f"&i=d"
        f"&apikey={APIKEY}"
    )
    headers = {'User-Agent': 'Mozilla/5.0'}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, verify=False, timeout=30)
            if 'Date' not in response.text:
                print(f"Warning: Unexpected response for {symbol}: {response.text[:300]}")
                return None
            df = pd.read_csv(StringIO(response.text))
            df.columns = [c.lower() for c in df.columns]
            df.insert(0, 'symbol', symbol)
            df = df[['symbol', 'date', 'open', 'close', 'high', 'low', 'volume']]
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            time.sleep(3)

    print(f"All {retries} attempts failed for {symbol}.")
    return None

# Fetch daily adjusted stock prices for QQQ and TQQQ from Stooq
qqq_data = fetch_stooq('QQQ', start_date, end_date)
tqqq_data = fetch_stooq('TQQQ', start_date, end_date)

# Save to CSV files
qqq_data.to_csv('QQQ.csv', index=False)
tqqq_data.to_csv('TQQQ.csv', index=False)

print("QQQ - first 5 rows:")
print(qqq_data.head())
print("\nTQQQ - first 5 rows:")
print(tqqq_data.head())
print(f"\nQQQ: {len(qqq_data)} records, TQQQ: {len(tqqq_data)} records")
print("QQQ.csv and TQQQ.csv have been saved.")