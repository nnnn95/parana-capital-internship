"""
QQQ Investment Strategy Analysis (2014-01-01 ~ 2023-12-31)
===========================================================
Strategies : Dollar-Cost Averaging (DCA) vs Momentum Investing
Data source: yfinance auto_adjust=True (dividend-adjusted total return prices)
Required   : pip install pandas numpy matplotlib seaborn yfinance
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0    # USD invested per month
RISK_FREE_RATE     = 0.02     # Annual risk-free rate for Sharpe ratio


# ── Step 1: Data Acquisition ──────────────────────────────────────────────────
def fetch_data(ticker, start_date, end_date, retries=5, base_delay=10):
    """
    Download daily dividend-adjusted close prices via yfinance.

    auto_adjust=True bakes dividends into the price series so that
    return calculations automatically reflect total return
    (capital gains + dividends reinvested).

    Retries with exponential back-off on HTTP 429 rate-limit errors:
    waits 10 s, 20 s, 40 s, 80 s between attempts.
    """
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"[INFO] Rate limited. Waiting {wait:.0f}s before retry {attempt}/{retries-1}...")
                time.sleep(wait)

            print(f"[INFO] Downloading {ticker} via yfinance (attempt {attempt+1})...")
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)

            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            # Flatten MultiIndex columns that yfinance >= 0.2 may produce
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            data = raw[["Close"]].copy()
            print(f"[INFO] Downloaded {len(data)} daily rows — "
                  f"start: ${data['Close'].iloc[0]:.2f}  end: ${data['Close'].iloc[-1]:.2f}")
            return data

        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ["429", "RateLimit", "Too Many"]):
                print(f"[WARNING] {e}")
                continue   # retry
            raise RuntimeError(f"yfinance download failed: {e}") from e

    raise RuntimeError(f"Download failed after {retries} attempts. Wait a few minutes and retry.")


# ── Step 2: Data Preprocessing ────────────────────────────────────────────────
def preprocess(daily_df):
    """
    Clean raw price data and derive additional indicators.
    - Drop missing Close values
    - Compute 50-day and 200-day Simple Moving Averages
    - Resample to month-end close prices for strategy simulation
    """
    df = daily_df.copy()
    df.dropna(subset=["Close"], inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.loc[START_DATE:END_DATE].sort_index()

    # 50-day SMA: short-term trend indicator
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    # 200-day SMA: long-term trend indicator
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Month-end close: one price per calendar month
    monthly = df["Close"].resample("ME").last()
    print(f"[INFO] Preprocessed: {len(df)} trading days, {len(monthly)} month-end prices")
    return df, monthly


# ── Step 3: Risk Metrics ──────────────────────────────────────────────────────
def max_drawdown(values):
    """Max drawdown fraction: min( (V - rolling_peak) / rolling_peak )."""
    return float(((values - values.cummax()) / values.cummax()).min())

def annualised_volatility(returns):
    """Sample std of monthly returns annualised by sqrt(12)."""
    return float(returns.std(ddof=1) * np.sqrt(12))

def sharpe_ratio(returns):
    """Annualised Sharpe = (mean excess return / std excess return) x sqrt(12)."""
    excess = returns - RISK_FREE_RATE / 12
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(12))

def xirr(portfolio, monthly_investment):
    """
    XIRR — annualised Internal Rate of Return for periodic investments.

    For DCA, each $500 is invested at a different point in time.
    CAGR on total invested assumes all capital was deployed at t=0, which
    systematically understates the true return. XIRR solves for the monthly
    rate r that makes NPV of all cash flows = 0:

        sum( CF_t / (1+r)^t ) = 0

    CF_t = -monthly_investment for each month t, plus +ending_value at final t.
    Annualised: (1 + monthly_r)^12 - 1.
    """
    ending     = portfolio.iloc[-1]
    cash_flows = [-monthly_investment] * len(portfolio)
    cash_flows[-1] += ending   # receive portfolio value at end

    rate = 0.01   # initial guess: 1% per month
    for _ in range(1000):
        npv  = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        if dnpv == 0:
            break
        rate -= npv / dnpv
        if abs(npv) < 1e-8:
            break
    return (1 + rate) ** 12 - 1   # annualise monthly rate


def compute_metrics(portfolio, monthly_investment=MONTHLY_INVESTMENT):
    """
    Compute all return and risk metrics for a portfolio value series.

    Annualised Return uses XIRR (Internal Rate of Return) — the correct
    method for DCA because it accounts for the timing of each $500 cash flow.
    Dividends are included via yfinance auto_adjust=True.
    """
    invested = monthly_investment * len(portfolio)
    ending   = portfolio.iloc[-1]
    rets     = portfolio.pct_change().dropna()
    return {
        "Total Invested ($)":    invested,
        "Ending Value ($)":      ending,
        "Return Multiplier (x)": ending / invested,
        "Total Return (%)":      (ending - invested) / invested * 100,
        "Annualised Return (%)": xirr(portfolio, monthly_investment) * 100,
        "Max Drawdown (%)":      max_drawdown(portfolio) * 100,
        "Volatility Ann. (%)":   annualised_volatility(rets) * 100,
        "Sharpe Ratio":          sharpe_ratio(rets),
    }


# ── Step 4: Strategy Implementations ─────────────────────────────────────────
def _sim_dca(monthly, monthly_investment):
    """
    DCA core simulation — returns monthly portfolio value series.
    shares_bought = monthly_investment / price  (never sell)
    """
    shares, values = 0.0, []
    for price in monthly:
        shares += monthly_investment / price    # buy fractional shares each month
        values.append(shares * price)           # current portfolio market value
    return pd.Series(values, index=monthly.index)


def _sim_momentum(monthly, monthly_investment):
    """
    Momentum core simulation — returns monthly portfolio value series.
    Signal: 6-month cumulative return skipping the most recent month.
    position = 1 -> invest $500 in QQQ; position = 0 -> hold cash.
    """
    rets      = monthly.pct_change()
    # Skip the most recent month (x[:-1]) to avoid short-term reversal noise
    momentum  = rets.rolling(6).apply(lambda x: x[:-1].sum(), raw=True)
    positions = (momentum > 0).astype(int)   # 1 = invest, 0 = cash

    shares, cash, values = 0.0, 0.0, []
    for date, price in monthly.items():
        if positions[date] == 1:
            shares += monthly_investment / price   # positive momentum -> buy QQQ
        else:
            cash += monthly_investment             # negative momentum -> hold cash
        values.append(shares * price + cash)
    return pd.Series(values, index=monthly.index)


# ── Public API (matches PDF function signatures) ──────────────────────────────
def backtest_dca(ticker, start_date, end_date, monthly_investment):
    """
    Dollar-Cost Averaging strategy backtest.
    Returns: annualized_return (%), total_return (%)
    """
    _, monthly        = preprocess(fetch_data(ticker, start_date, end_date))
    portfolio         = _sim_dca(monthly, monthly_investment)
    m                 = compute_metrics(portfolio, monthly_investment)
    return m["Annualised Return (%)"], m["Total Return (%)"]


def momentum_strategy(ticker, start_date, end_date, monthly_investment):
    """
    Momentum investing strategy backtest.
    Returns: annualized_return (%), total_return (%)
    """
    _, monthly        = preprocess(fetch_data(ticker, start_date, end_date))
    portfolio         = _sim_momentum(monthly, monthly_investment)
    m                 = compute_metrics(portfolio, monthly_investment)
    return m["Annualised Return (%)"], m["Total Return (%)"]


# ── Step 5 & 6: Simulation + Visualisation ───────────────────────────────────
def plot(dca, mom, daily, save_path):
    """4-panel comparison chart: portfolio value, return distribution, drawdown, price+SMA."""
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("QQQ Investment Strategy Comparison  (2014-2023)",
                 fontsize=14, fontweight="bold")

    # 1. Portfolio value over time
    ax = axes[0, 0]
    ax.plot(dca.index, dca, label="DCA",      color="steelblue", linewidth=2)
    ax.plot(mom.index, mom, label="Momentum", color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Portfolio Value Over Time"); ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()

    # 2. Monthly return distribution
    ax = axes[0, 1]
    ax.hist(dca.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="steelblue",  edgecolor="white", label="DCA")
    ax.hist(mom.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="darkorange", edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution"); ax.set_xlabel("%"); ax.legend()

    # 3. Drawdown over time
    ax = axes[1, 0]
    dca_dd = (dca - dca.cummax()) / dca.cummax() * 100
    mom_dd = (mom - mom.cummax()) / mom.cummax() * 100
    ax.fill_between(dca.index, dca_dd, 0, alpha=0.4, color="steelblue",  label="DCA")
    ax.fill_between(mom.index, mom_dd, 0, alpha=0.4, color="darkorange", label="Momentum")
    ax.set_title("Drawdown Over Time"); ax.set_ylabel("%"); ax.legend()

    # 4. QQQ adjusted close price with SMA-50 and SMA-200
    ax = axes[1, 1]
    ax.plot(daily.index, daily["Close"],   color="black",     linewidth=0.8, alpha=0.7, label="Adj Close")
    ax.plot(daily.index, daily["SMA_50"],  color="royalblue", linewidth=1.2, linestyle="--", label="SMA 50")
    ax.plot(daily.index, daily["SMA_200"], color="crimson",   linewidth=1.2, linestyle="-.", label="SMA 200")
    ax.set_title("QQQ Adj Close + Moving Averages"); ax.set_ylabel("USD"); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved -> {save_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  QQQ Investment Strategy Analysis (yfinance)")
    print(f"  Period            : {START_DATE}  ->  {END_DATE}")
    print(f"  Monthly Investment: ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Data              : yfinance (auto_adjust=True, dividends included)")
    print("=" * 55)

    # Public API calls — matches PDF spec signatures
    dca_annual, dca_total = backtest_dca(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)
    mom_annual, mom_total = momentum_strategy(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)

    print(f"\n[DCA]      Annualised Return: {dca_annual:.2f}%  |  Total Return: {dca_total:.2f}%")
    print(f"[Momentum] Annualised Return: {mom_annual:.2f}%  |  Total Return: {mom_total:.2f}%")

    # Full analysis with risk metrics and charts (fetches data once)
    daily, monthly = preprocess(fetch_data(TICKER, START_DATE, END_DATE))
    dca_portfolio  = _sim_dca(monthly, MONTHLY_INVESTMENT)
    mom_portfolio  = _sim_momentum(monthly, MONTHLY_INVESTMENT)

    dca_m = compute_metrics(dca_portfolio)
    mom_m = compute_metrics(mom_portfolio)

    print(f"\n{'Metric':<26} {'DCA':>14} {'Momentum':>14}")
    print("-" * 56)
    for key in dca_m:
        fmt = ",.2f" if "$" in key else ".4f" if "Sharpe" in key else ".2f"
        print(f"{key:<26} {format(dca_m[key], fmt):>14} {format(mom_m[key], fmt):>14}")
    print("-" * 56)

    chart = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qqq_strategy_analysis_yfinance.png")
    plot(dca_portfolio, mom_portfolio, daily, save_path=chart)
