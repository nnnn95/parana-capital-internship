"""
QQQ Investment Strategy Analysis - Version 2
探索不同的计算方法，尝试得到14%左右的DCA年化收益
"""

import yfinance as yf
import pandas as pd
import numpy as np

# 配置
TICKER = "QQQ"
START_DATE = "2014-01-01"
END_DATE = "2023-12-31"
MONTHLY_INVESTMENT = 500

print("=" * 60)
print("QQQ DCA Investment Analysis - Exploring Different Methods")
print("=" * 60)

# 获取数据 (不调整)
print("\n[INFO] Downloading QQQ data (unadjusted prices)...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)

if isinstance(df.columns, pd.MultiIndex):
    close = df['Close'][TICKER]
    adj_close = df['Adj Close'][TICKER]
else:
    close = df['Close']
    adj_close = df['Adj Close']

print(f"[INFO] Date range: {close.index.min().date()} to {close.index.max().date()}")
print(f"[INFO] Unadjusted Close: ${close.iloc[0]:.2f} -> ${close.iloc[-1]:.2f}")
print(f"[INFO] Adjusted Close: ${adj_close.iloc[0]:.2f} -> ${adj_close.iloc[-1]:.2f}")

# 月末价格
monthly_close = close.resample('ME').last()
monthly_adj = adj_close.resample('ME').last()

print(f"\n[INFO] {len(monthly_close)} months of data")

# =============================================================================
# 方法1: 标准DCA (使用未调整价格)
# =============================================================================
print("\n" + "=" * 60)
print("Method 1: Standard DCA (Unadjusted Close)")
print("=" * 60)

shares = sum(MONTHLY_INVESTMENT / p for p in monthly_close)
ending = shares * close.iloc[-1]
multiplier = ending / 60000
cagr = (multiplier ** 0.1 - 1) * 100

# IRR计算
cash_flows = [-MONTHLY_INVESTMENT] * len(monthly_close)
cash_flows[-1] += ending
rate = 0.01
for _ in range(1000):
    npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
    dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
    if abs(dnpv) < 1e-10:
        break
    rate -= npv / dnpv
    if abs(npv) < 1e-8:
        break
irr = (1 + rate) ** 12 - 1

print(f"Total Invested: $60,000")
print(f"Ending Value: ${ending:,.2f}")
print(f"Return Multiplier: {multiplier:.2f}x")
print(f"CAGR: {cagr:.2f}%")
print(f"IRR: {irr*100:.2f}%")

# =============================================================================
# 方法2: DCA (使用调整后价格)
# =============================================================================
print("\n" + "=" * 60)
print("Method 2: DCA (Adjusted Close - includes dividends)")
print("=" * 60)

shares = sum(MONTHLY_INVESTMENT / p for p in monthly_adj)
ending = shares * adj_close.iloc[-1]
multiplier = ending / 60000
cagr = (multiplier ** 0.1 - 1) * 100

cash_flows = [-MONTHLY_INVESTMENT] * len(monthly_adj)
cash_flows[-1] += ending
rate = 0.01
for _ in range(1000):
    npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
    dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
    if abs(dnpv) < 1e-10:
        break
    rate -= npv / dnpv
    if abs(npv) < 1e-8:
        break
irr = (1 + rate) ** 12 - 1

print(f"Total Invested: $60,000")
print(f"Ending Value: ${ending:,.2f}")
print(f"Return Multiplier: {multiplier:.2f}x")
print(f"CAGR: {cagr:.2f}%")
print(f"IRR: {irr*100:.2f}%")

# =============================================================================
# 方法3: 简单平均年化收益 (Average Annual Return)
# =============================================================================
print("\n" + "=" * 60)
print("Method 3: Average Annual Return (Arithmetic Mean)")
print("=" * 60)

# 计算每年的收益
annual_returns = close.resample('YE').last().pct_change().dropna()
avg_annual = annual_returns.mean() * 100
print(f"Average Annual Return: {avg_annual:.2f}%")

# =============================================================================
# 方法4: 时间加权回报率 (TWR)
# =============================================================================
print("\n" + "=" * 60)
print("Method 4: Time-Weighted Return (TWR)")
print("=" * 60)

# 计算每月收益
monthly_returns = adj_close.resample('ME').last().pct_change().dropna()
# TWR = 产品(1 + r_i) - 1
twr = (1 + monthly_returns).prod() - 1
twr_annual = (1 + twr) ** (12 / len(monthly_returns)) - 1
print(f"Total TWR: {twr*100:.2f}%")
print(f"Annualized TWR: {twr_annual*100:.2f}%")

# =============================================================================
# 方法5: 买入并持有对比
# =============================================================================
print("\n" + "=" * 60)
print("Method 5: Buy and Hold (Lump Sum Investment)")
print("=" * 60)

# 一次性投资
lump_sum = 60000
shares_lump = lump_sum / close.iloc[0]
ending_lump = shares_lump * close.iloc[-1]
cagr_lump = ((ending_lump / lump_sum) ** 0.1 - 1) * 100

# 一次性投资 (调整后)
shares_lump_adj = lump_sum / adj_close.iloc[0]
ending_lump_adj = shares_lump_adj * adj_close.iloc[-1]
cagr_lump_adj = ((ending_lump_adj / lump_sum) ** 0.1 - 1) * 100

print(f"Unadjusted: ${ending_lump:,.2f} (CAGR: {cagr_lump:.2f}%)")
print(f"Adjusted: ${ending_lump_adj:,.2f} (CAGR: {cagr_lump_adj:.2f}%)")

# =============================================================================
# 方法6: 中点投资法 (假设资金在期中投入)
# =============================================================================
print("\n" + "=" * 60)
print("Method 6: Mid-Point Investment (Capital at year 5)")
print("=" * 60)

# 假设所有资金在第5年投入
mid_invested = 60000
mid_price = adj_close.iloc[len(adj_close)//2]
shares_mid = mid_invested / mid_price
ending_mid = shares_mid * adj_close.iloc[-1]
cagr_mid = ((ending_mid / mid_invested) ** (1/5) - 1) * 100

print(f"If invested at midpoint: CAGR = {cagr_mid:.2f}%")

# =============================================================================
# 总结
# =============================================================================
print("\n" + "=" * 60)
print("Summary: Which method gives ~14%?")
print("=" * 60)
print(f"Method 1 (DCA Unadjusted CAGR): ~9-10%")
print(f"Method 2 (DCA Adjusted CAGR): ~10%")
print(f"Method 2 (DCA Adjusted IRR): ~18%")
print(f"Method 5 (Buy-Hold Adjusted CAGR): ~{cagr_lump_adj:.1f}%")
print(f"\nNo standard method produces exactly 14%.")
print("14% falls between CAGR (~10%) and IRR (~18%).")
