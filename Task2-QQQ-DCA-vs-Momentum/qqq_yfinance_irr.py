"""
QQQ Investment Strategy Analysis (2014-01-01 ~ 2023-12-31)
===========================================================
Strategies : Dollar-Cost Averaging (DCA) vs Momentum Investing
Data source: yfinance auto_adjust=True  (dividend-adjusted total return prices)
Return calc: XIRR / IRR  (accounts for timing of each monthly $500 cash flow)
Required   : pip install pandas numpy matplotlib seaborn yfinance
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0    # USD invested per month
RISK_FREE_RATE     = 0.02     # Annual risk-free rate (for Sharpe ratio)


# ── Step 1: Fetch dividend-adjusted price data via yfinance ────────────────────
def fetch_data(ticker, start_date, end_date, retries=5, base_delay=10):
    """
    Download dividend-adjusted daily close prices using yfinance.

    auto_adjust=True applies backward dividend adjustment to all historical
    prices, so the price series reflects total return (price + reinvested
    dividends). This is the correct input for DCA return calculations.

    Exponential back-off is used to handle HTTP 429 rate-limit errors.
    """
    for attempt in range(retries):
        try:
            if attempt > 0:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"[INFO] Rate limited — waiting {wait:.0f}s "
                      f"(retry {attempt}/{retries - 1}) ...")
                time.sleep(wait)

            print(f"[INFO] Downloading {ticker} (attempt {attempt + 1}) ...")
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)

            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Close"]].copy()
            df.dropna(inplace=True)
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            print(f"[INFO] {len(df)} daily rows downloaded  "
                  f"| start price: ${float(df['Close'].iloc[0]):.2f}  "
                  f"| end price:   ${float(df['Close'].iloc[-1]):.2f}")
            return df

        except Exception as exc:
            msg = str(exc)
            if any(k in msg for k in ["429", "RateLimit", "Too Many"]):
                print(f"[WARNING] Rate limit hit: {exc}")
                continue
            raise RuntimeError(f"yfinance error: {exc}") from exc

    raise RuntimeError(
        f"Download failed after {retries} attempts — wait a few minutes and retry."
    )


# ── Step 2: Preprocessing ──────────────────────────────────────────────────────
def preprocess(daily_df):
    """
    Trim to analysis window, compute SMAs, resample to month-end prices.
    Returns (daily_df_with_sma, monthly_close_series).
    """
    df = daily_df.loc[START_DATE:END_DATE].sort_index().copy()
    df["SMA_50"]  = df["Close"].rolling(50).mean()   # short-term trend
    df["SMA_200"] = df["Close"].rolling(200).mean()  # long-term trend

    monthly = df["Close"].resample("ME").last()
    print(f"[INFO] Analysis window: {len(df)} trading days, "
          f"{len(monthly)} month-end prices")
    return df, monthly


# ── Step 3: IRR / XIRR ────────────────────────────────────────────────────────
def compute_irr(portfolio, monthly_investment):
    """
    Annualised Internal Rate of Return (XIRR) for a DCA portfolio.

    Why IRR instead of simple CAGR?
    --------------------------------
    CAGR = (Ending / Total_Invested)^(1/n) - 1 assumes all capital was deployed
    at t = 0. For DCA, the first $500 has compounded for ~10 years while the
    last $500 has compounded for only 1 month. Treating $60,000 as a day-one
    lump sum seriously understates the true return.

    IRR solves for the monthly discount rate r that makes the Net Present Value
    of all cash flows equal to zero:

        sum_t [ CF_t / (1+r)^t ] = 0

    where CF_t = -500 for every month t, plus +ending_value at the final month.
    The annual rate = (1 + monthly_r)^12 - 1.

    Newton-Raphson iteration converges quickly (< 50 iterations typical).
    """
    ending     = float(portfolio.iloc[-1])
    cash_flows = [-monthly_investment] * len(portfolio)
    cash_flows[-1] += ending          # inflow: liquidate portfolio at end

    r = 0.01                          # initial guess: 1% per month
    for _ in range(1000):
        npv  = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows))
        if abs(dnpv) < 1e-12:
            break
        r -= npv / dnpv
        if abs(npv) < 1e-8:
            break

    return (1 + r) ** 12 - 1          # annualise


# ── Step 4: Risk metrics ───────────────────────────────────────────────────────
def max_drawdown(values):
    """Peak-to-trough drawdown as a fraction (negative number)."""
    return float(((values - values.cummax()) / values.cummax()).min())

def annualised_volatility(returns):
    """Sample standard deviation of monthly returns, scaled by sqrt(12)."""
    return float(returns.std(ddof=1) * np.sqrt(12))

def sharpe_ratio(returns):
    """Annualised Sharpe ratio using the configured risk-free rate."""
    excess = returns - RISK_FREE_RATE / 12
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(12))

def compute_metrics(portfolio, monthly_investment=MONTHLY_INVESTMENT):
    """Full metric dict for a portfolio value series."""
    invested = monthly_investment * len(portfolio)
    ending   = float(portfolio.iloc[-1])
    rets     = portfolio.pct_change().dropna()
    return {
        "Total Invested ($)":    invested,
        "Ending Value ($)":      ending,
        "Return Multiplier (x)": ending / invested,
        "Total Return (%)":      (ending - invested) / invested * 100,
        "Annualised Return (%)": compute_irr(portfolio, monthly_investment) * 100,
        "Max Drawdown (%)":      max_drawdown(portfolio) * 100,
        "Volatility Ann. (%)":   annualised_volatility(rets) * 100,
        "Sharpe Ratio":          sharpe_ratio(rets),
    }


# ── Step 5: Strategy simulations ──────────────────────────────────────────────
def _sim_dca(monthly, monthly_investment):
    """
    DCA simulation.
    Each month: buy (monthly_investment / price) fractional shares.
    Never sell. Returns monthly portfolio value series.
    """
    shares, values = 0.0, []
    for price in monthly:
        shares += monthly_investment / float(price)
        values.append(shares * float(price))
    return pd.Series(values, index=monthly.index)


def _sim_momentum(monthly, monthly_investment):
    """
    Momentum simulation.
    Signal: 6-month cumulative return excluding the most recent month
    (skipping t-1 avoids short-term reversal noise).
    position = 1  → invest $500 in QQQ this month
    position = 0  → park $500 as cash this month
    Returns monthly portfolio value series.
    """
    rets     = monthly.pct_change()
    momentum = rets.rolling(6).apply(lambda x: x[:-1].sum(), raw=True)
    pos      = (momentum > 0).astype(int)

    shares, cash, values = 0.0, 0.0, []
    for date, price in monthly.items():
        if pos[date] == 1:
            shares += monthly_investment / float(price)
        else:
            cash += monthly_investment
        values.append(shares * float(price) + cash)
    return pd.Series(values, index=monthly.index)


# ── Public API (matches PDF spec function signatures) ─────────────────────────
def backtest_dca(ticker, start_date, end_date, monthly_investment):
    """
    DCA strategy backtest.
    Returns: (annualized_return_pct, total_return_pct)
    """
    daily      = fetch_data(ticker, start_date, end_date)
    _, monthly = preprocess(daily)
    portfolio  = _sim_dca(monthly, monthly_investment)
    m          = compute_metrics(portfolio, monthly_investment)
    return m["Annualised Return (%)"], m["Total Return (%)"]


def momentum_strategy(ticker, start_date, end_date, monthly_investment):
    """
    Momentum strategy backtest.
    Returns: (annualized_return_pct, total_return_pct)
    """
    daily      = fetch_data(ticker, start_date, end_date)
    _, monthly = preprocess(daily)
    portfolio  = _sim_momentum(monthly, monthly_investment)
    m          = compute_metrics(portfolio, monthly_investment)
    return m["Annualised Return (%)"], m["Total Return (%)"]


# ── Step 6: Visualisation ──────────────────────────────────────────────────────
def plot(dca, mom, daily, save_path):
    """4-panel comparison chart."""
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("QQQ Investment Strategy Comparison  (2014–2023)",
                 fontsize=14, fontweight="bold")

    # Panel 1 — Portfolio value
    ax = axes[0, 0]
    ax.plot(dca.index, dca, label="DCA",      color="steelblue",  linewidth=2)
    ax.plot(mom.index, mom, label="Momentum", color="darkorange", linewidth=2,
            linestyle="--")
    ax.set_title("Portfolio Value Over Time")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()

    # Panel 2 — Monthly return distribution
    ax = axes[0, 1]
    ax.hist(dca.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="steelblue",  edgecolor="white", label="DCA")
    ax.hist(mom.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="darkorange", edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("%")
    ax.legend()

    # Panel 3 — Drawdown
    ax = axes[1, 0]
    dca_dd = (dca - dca.cummax()) / dca.cummax() * 100
    mom_dd = (mom - mom.cummax()) / mom.cummax() * 100
    ax.fill_between(dca.index, dca_dd, 0, alpha=0.4,
                    color="steelblue",  label="DCA")
    ax.fill_between(mom.index, mom_dd, 0, alpha=0.4,
                    color="darkorange", label="Momentum")
    ax.set_title("Drawdown Over Time")
    ax.set_ylabel("%")
    ax.legend()

    # Panel 4 — QQQ price + SMAs
    ax = axes[1, 1]
    ax.plot(daily.index, daily["Close"],   color="black",     linewidth=0.8,
            alpha=0.7, label="Adj Close")
    ax.plot(daily.index, daily["SMA_50"],  color="royalblue", linewidth=1.2,
            linestyle="--", label="SMA 50")
    ax.plot(daily.index, daily["SMA_200"], color="crimson",   linewidth=1.2,
            linestyle="-.", label="SMA 200")
    ax.set_title("QQQ Adj Close + Moving Averages")
    ax.set_ylabel("USD")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  QQQ Investment Strategy Analysis")
    print(f"  Period            : {START_DATE}  →  {END_DATE}")
    print(f"  Monthly Investment: ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Data source       : yfinance  (auto_adjust=True)")
    print(f"  Return method     : XIRR / IRR  (cash-flow timing aware)")
    print("=" * 60)

    # ── Quick summary via public API ──────────────────────────────────────────
    dca_ann, dca_tot = backtest_dca(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)
    mom_ann, mom_tot = momentum_strategy(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)

    print(f"\n[DCA]      Annualised Return (IRR): {dca_ann:.2f}%  "
          f"| Total Return: {dca_tot:.2f}%")
    print(f"[Momentum] Annualised Return (IRR): {mom_ann:.2f}%  "
          f"| Total Return: {mom_tot:.2f}%")

    # ── Full metrics + chart (fetch data once) ────────────────────────────────
    daily, monthly = preprocess(fetch_data(TICKER, START_DATE, END_DATE))
    dca_port = _sim_dca(monthly, MONTHLY_INVESTMENT)
    mom_port = _sim_momentum(monthly, MONTHLY_INVESTMENT)

    dca_m = compute_metrics(dca_port)
    mom_m = compute_metrics(mom_port)

    print(f"\n{'Metric':<26} {'DCA':>14} {'Momentum':>14}")
    print("─" * 56)
    for key in dca_m:
        fmt = ",.2f" if "$" in key else ".4f" if "Sharpe" in key else ".2f"
        print(f"{key:<26} {format(dca_m[key], fmt):>14} {format(mom_m[key], fmt):>14}")
    print("─" * 56)

    chart = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "qqq_strategy_yfinance_irr.png")
    plot(dca_port, mom_port, daily, save_path=chart)
