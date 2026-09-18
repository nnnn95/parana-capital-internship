"""
QQQ Investment Strategy Analysis
=================================
Analyzes historical data of QQQ ETF to simulate and compare two investment
strategies over the period 2014-01-01 to 2023-12-31:

  1. Dollar-Cost Averaging (DCA)  – invest $500 every month, never sell.
  2. Momentum Investing           – invest $500 only when 6-month momentum > 0.

Outputs
-------
  - Console summary table (total return, annualised return, risk metrics)
  - 4-panel chart saved to the same directory as this script

Required libraries
------------------
  pip install pandas numpy matplotlib seaborn yfinance

Reference formulas (from project spec)
---------------------------------------
  Total Return       = (Ending - Invested) / Invested × 100 %
  Annualised Return  = (Ending / Invested)^(1/n) − 1   [n = years]
  Volatility         = std(monthly returns) × sqrt(12)
  Sharpe Ratio       = (mean excess return / std excess return) × sqrt(12)
  Max Drawdown       = min( (value − rolling_max) / rolling_max )
"""

import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Global configuration
# ──────────────────────────────────────────────
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0     # Fixed USD invested each month
RISK_FREE_RATE     = 0.02      # Annual risk-free rate assumed at 2 %


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for *ticker* using yfinance.

    Automatically retries on HTTP 429 / YFRateLimitError using exponential
    back-off: waits base_delay × 2^attempt seconds between attempts.

    Parameters
    ----------
    ticker       : Ticker symbol, e.g. 'QQQ'.
    start_date   : Start of the date range, 'YYYY-MM-DD'.
    end_date     : End of the date range, 'YYYY-MM-DD'.
    max_retries  : Maximum number of download attempts (default 5).
    base_delay   : Initial wait time in seconds before the first retry (default 10 s).

    Returns
    -------
    pd.DataFrame with a single column 'Close', indexed by date (ascending).

    Raises
    ------
    RuntimeError if all retry attempts are exhausted or data is still empty.
    """
    import time
    import yfinance as yf

    last_exc = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential back-off: 10 s, 20 s, 40 s, 80 s …
                wait = base_delay * (2 ** (attempt - 1))
                print(f"[INFO] Waiting {wait:.0f} s before retry {attempt}/{max_retries - 1} …")
                time.sleep(wait)

            print(f"[INFO] Downloading {ticker} via yfinance (attempt {attempt + 1}) …")
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,   # Adjust prices for splits and dividends
                progress=False,
            )

            if raw.empty:
                # Empty result without an exception — treat as a soft failure
                raise ValueError("yfinance returned an empty DataFrame.")

            # Keep only the adjusted close price
            data = raw[["Close"]].copy()

            # Flatten MultiIndex columns that yfinance ≥ 0.2 may produce
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            print(f"[INFO] yfinance: {len(data)} daily rows received.")
            return data

        except Exception as exc:
            last_exc = exc
            exc_msg  = str(exc)

            # Detect rate-limit errors specifically to decide whether to retry
            is_rate_limit = (
                "429"          in exc_msg
                or "RateLimit" in exc_msg
                or "Too Many"  in exc_msg
            )

            if is_rate_limit:
                print(f"[WARNING] Rate limited by yfinance: {exc}")
                # Continue to the next retry iteration
            else:
                # Non-rate-limit error — fail immediately
                raise RuntimeError(
                    f"yfinance download failed for '{ticker}': {exc}"
                ) from exc

    # All retries exhausted
    raise RuntimeError(
        f"yfinance download failed for '{ticker}' after {max_retries} attempts "
        f"(last error: {last_exc}). "
        "The API is still rate-limiting. Wait a few minutes and try again."
    ) from last_exc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_data(
    daily_data: pd.DataFrame,
) -> tuple:
    """
    Clean raw price data and derive additional indicators.

    Steps performed
    ---------------
    1. Drop rows with missing 'Close' values.
    2. Remove duplicate dates (keep first occurrence).
    3. Ensure a proper DatetimeIndex, then slice to [START_DATE, END_DATE].
    4. Compute 50-day and 200-day Simple Moving Averages (SMA).
    5. Resample to month-end ('ME') to obtain one price per calendar month.

    Parameters
    ----------
    daily_data : Raw DataFrame with at least a 'Close' column.

    Returns
    -------
    (daily_df, monthly_df)
        daily_df   – cleaned daily prices with SMA_50 and SMA_200 columns.
        monthly_df – month-end 'Close' prices, one row per month.
    """
    df = daily_data.copy()

    # Remove rows where the close price is missing
    before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Removed {dropped} rows with missing Close prices.")

    # Remove duplicate index entries
    df = df[~df.index.duplicated(keep="first")]

    # Guarantee the index is a proper DatetimeIndex
    df.index = pd.to_datetime(df.index)

    # Restrict to the analysis window
    df = df.loc[START_DATE:END_DATE].copy()

    # ── Additional indicators ─────────────────────────────────────────────────
    # 50-day Simple Moving Average – short-term trend indicator
    df["SMA_50"]  = df["Close"].rolling(window=50).mean()
    # 200-day Simple Moving Average – long-term trend indicator
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    print(
        f"[INFO] Daily data ready: {df.index[0].date()} → "
        f"{df.index[-1].date()}  ({len(df)} trading days)"
    )

    # Resample to month-end: use the last available closing price each month
    monthly_df = df["Close"].resample("ME").last().to_frame()
    monthly_df.columns = ["Close"]
    print(f"[INFO] Monthly data: {len(monthly_df)} months")

    return df, monthly_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – RISK METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Compute the Maximum Drawdown (MDD) of a portfolio value series.

    Formula
    -------
    MDD = min{ (V_t − max(V_0 … V_t)) / max(V_0 … V_t) }

    A value of -0.30 means the portfolio fell 30 % from its prior peak at
    its worst point.

    Parameters
    ----------
    portfolio_values : Time series of portfolio values (non-negative).

    Returns
    -------
    float – maximum drawdown as a fraction (negative; e.g. -0.30 = −30 %).
    """
    # Running maximum up to each point in time (the 'high-water mark')
    running_max = portfolio_values.cummax()
    # Drawdown at each period relative to the preceding peak
    drawdown    = (portfolio_values - running_max) / running_max
    return float(drawdown.min())


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Compute volatility as the sample standard deviation of periodic returns.

    Formula (from spec)
    -------------------
    Volatility = sqrt( (1 / (N-1)) * Σ (R_i − R̄)² )

    Parameters
    ----------
    returns   : Series of periodic (e.g. monthly) returns.
    annualize : If True, multiply by sqrt(12) to convert monthly → annual.

    Returns
    -------
    float – (annualised) volatility.
    """
    # ddof=1 → sample standard deviation (N-1 denominator)
    vol = returns.std(ddof=1)
    if annualize:
        vol *= np.sqrt(12)   # Annualise from monthly frequency
    return float(vol)


def calculate_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Compute the annualised Sharpe Ratio.

    Formula (from spec)
    -------------------
    Sharpe = (Portfolio Return − Risk-Free Rate) / Volatility of Portfolio

    Uses monthly returns; both numerator and denominator are annualised.

    Parameters
    ----------
    portfolio_returns : Monthly portfolio return series.
    risk_free_rate    : Annual risk-free rate (default = 2 %).

    Returns
    -------
    float – annualised Sharpe Ratio.
    """
    # Convert the annual risk-free rate to a monthly equivalent
    monthly_rf = risk_free_rate / 12

    # Excess return over the risk-free rate each month
    excess = portfolio_returns - monthly_rf

    # Annualise: mean / std × sqrt(12)
    sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(12)
    return float(sharpe)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 & 5 – STRATEGY IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def backtest_dca(
    ticker: str,
    start_date: str,
    end_date: str,
    monthly_investment: float = MONTHLY_INVESTMENT,
) -> tuple:
    """
    Backtest the Dollar-Cost Averaging (DCA) strategy.

    Strategy rules
    --------------
    • Invest exactly *monthly_investment* USD on the last trading day of
      every calendar month.
    • Never sell any shares – accumulate throughout the entire period.
    • Portfolio value = total_shares_held × current_month_end_price.

    Shares purchased each month (from spec):
        shares_bought = monthly_investment / monthly_price

    Parameters
    ----------
    ticker             : ETF ticker symbol.
    start_date         : Analysis start date ('YYYY-MM-DD').
    end_date           : Analysis end date ('YYYY-MM-DD').
    monthly_investment : Fixed USD amount invested each month (default $500).

    Returns
    -------
    annualized_return  : float      – CAGR over the period (%).
    total_return       : float      – Overall return vs. total invested (%).
    portfolio_series   : pd.Series  – Monthly portfolio value time series.
    metrics            : dict       – Summary metrics dictionary.
    """
    # ── Fetch & preprocess ───────────────────────────────────────────────────
    raw                = fetch_data(ticker, start_date, end_date)
    _, monthly_data    = preprocess_data(raw)
    prices             = monthly_data["Close"]

    # ── Simulation ───────────────────────────────────────────────────────────
    total_shares   = 0.0   # Cumulative QQQ shares held
    total_invested = 0.0   # Cumulative USD committed to the market
    portfolio_vals = []

    for price in prices:
        # Buy a fractional number of shares at the month-end price
        shares_bought   = monthly_investment / price
        total_shares   += shares_bought
        total_invested += monthly_investment

        # Current portfolio value = shares held × today's price
        portfolio_vals.append(total_shares * price)

    portfolio_series = pd.Series(portfolio_vals, index=prices.index)

    # ── Return calculations (from spec) ──────────────────────────────────────
    ending_value = portfolio_series.iloc[-1]

    # Total Return = (Ending − Invested) / Invested × 100 %
    total_return = (ending_value - total_invested) / total_invested * 100.0

    # Annualised Return (CAGR): (Ending / Invested)^(1/n) − 1
    n_years           = (prices.index[-1] - prices.index[0]).days / 365.25
    annualized_return = ((ending_value / total_invested) ** (1.0 / n_years) - 1.0) * 100.0

    # ── Risk assessment ──────────────────────────────────────────────────────
    monthly_rets = portfolio_series.pct_change().dropna()
    max_dd       = calculate_max_drawdown(portfolio_series)
    volatility   = calculate_volatility(monthly_rets, annualize=True)
    sharpe       = calculate_sharpe_ratio(monthly_rets)

    metrics = {
        "total_invested":    total_invested,
        "ending_value":      ending_value,
        "total_return_pct":  total_return,
        "annualized_return": annualized_return,
        "max_drawdown_pct":  max_dd    * 100.0,
        "volatility_pct":    volatility * 100.0,
        "sharpe_ratio":      sharpe,
    }

    # ── Console output ───────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  DCA Strategy Results")
    print("─" * 45)
    print(f"  Total Invested     : ${total_invested:>12,.2f}")
    print(f"  Ending Value       : ${ending_value:>12,.2f}")
    print(f"  Return Multiplier  : {ending_value / total_invested:>10.2f}×  (expected ~2.7×)")
    print(f"  Total Return       : {total_return:>10.2f} %")
    print(f"  Annualised Return  : {annualized_return:>10.2f} %")
    print(f"  Max Drawdown       : {max_dd * 100:>10.2f} %")
    print(f"  Volatility (Ann.)  : {volatility * 100:>10.2f} %")
    print(f"  Sharpe Ratio       : {sharpe:>10.4f}")

    return annualized_return, total_return, portfolio_series, metrics


def momentum_strategy(
    ticker: str,
    start_date: str,
    end_date: str,
    monthly_investment: float = MONTHLY_INVESTMENT,
) -> tuple:
    """
    Backtest the Momentum Investing strategy.

    Strategy rules
    --------------
    • Each month compute a 6-month cumulative return, **skipping the most
      recent month** (i.e. using months t-6 to t-2) to avoid short-term
      reversal bias.
    • Signal = 1  →  positive momentum: invest *monthly_investment* in QQQ.
    • Signal = 0  →  negative momentum: hold cash, do NOT invest in QQQ.
    • Shares already held are never sold.
    • Portfolio value = shares_held × current_price + cash_balance.

    From spec:
        momentum  = monthly_returns.rolling(window=6).apply(lambda x: x[:-1].sum())
        positions = (momentum > 0).astype(int)

    Parameters
    ----------
    ticker             : ETF ticker symbol.
    start_date         : Analysis start date ('YYYY-MM-DD').
    end_date           : Analysis end date ('YYYY-MM-DD').
    monthly_investment : Fixed USD amount invested when signal = 1 (default $500).

    Returns
    -------
    annualized_return  : float      – CAGR over the period (%).
    total_return       : float      – Overall return vs. total invested (%).
    portfolio_series   : pd.Series  – Monthly portfolio value time series.
    metrics            : dict       – Summary metrics dictionary.
    """
    # ── Fetch & preprocess ───────────────────────────────────────────────────
    raw             = fetch_data(ticker, start_date, end_date)
    _, monthly_data = preprocess_data(raw)
    prices          = monthly_data["Close"]

    # ── Compute momentum signal ──────────────────────────────────────────────
    # Month-over-month percentage returns
    monthly_returns = prices.pct_change()

    # 6-month rolling cumulative return, skipping the most recent month:
    # x[:-1] excludes the last element of the rolling window, so we sum
    # months t-6 through t-2 (avoiding short-term reversal).
    momentum = monthly_returns.rolling(window=6).apply(
        lambda x: x[:-1].sum(), raw=True
    )

    # Position: 1 = invest in QQQ this month, 0 = stay in cash
    positions = (momentum > 0).astype(int)

    # ── Simulation ───────────────────────────────────────────────────────────
    total_shares   = 0.0   # Cumulative QQQ shares held
    total_invested = 0.0   # Cumulative USD committed (to market OR cash)
    cash_balance   = 0.0   # USD sitting in cash (signal = 0 months)
    portfolio_vals = []

    for date, price in prices.items():
        signal = int(positions.loc[date])

        if signal == 1:
            # Positive momentum → buy QQQ at this month's close price
            shares_bought   = monthly_investment / price
            total_shares   += shares_bought
            total_invested += monthly_investment
        else:
            # No signal → keep the $500 as cash (opportunity cost tracked)
            cash_balance   += monthly_investment
            total_invested += monthly_investment

        # Portfolio value = market value of shares + uninvested cash
        portfolio_vals.append(total_shares * price + cash_balance)

    portfolio_series = pd.Series(portfolio_vals, index=prices.index)

    # ── Return calculations (from spec) ──────────────────────────────────────
    ending_value = portfolio_series.iloc[-1]

    # Total Return relative to all capital committed
    total_return = (ending_value - total_invested) / total_invested * 100.0

    # Annualised Return (CAGR)
    n_years           = (prices.index[-1] - prices.index[0]).days / 365.25
    annualized_return = ((ending_value / total_invested) ** (1.0 / n_years) - 1.0) * 100.0

    # ── Risk assessment ──────────────────────────────────────────────────────
    monthly_rets = portfolio_series.pct_change().dropna()
    max_dd       = calculate_max_drawdown(portfolio_series)
    volatility   = calculate_volatility(monthly_rets, annualize=True)
    sharpe       = calculate_sharpe_ratio(monthly_rets)

    metrics = {
        "total_invested":    total_invested,
        "ending_value":      ending_value,
        "total_return_pct":  total_return,
        "annualized_return": annualized_return,
        "max_drawdown_pct":  max_dd    * 100.0,
        "volatility_pct":    volatility * 100.0,
        "sharpe_ratio":      sharpe,
    }

    # ── Console output ───────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Momentum Strategy Results")
    print("─" * 45)
    print(f"  Total Invested     : ${total_invested:>12,.2f}")
    print(f"  Ending Value       : ${ending_value:>12,.2f}")
    print(f"  Return Multiplier  : {ending_value / total_invested:>10.2f}×  (expected ~2.7×)")
    print(f"  Total Return       : {total_return:>10.2f} %")
    print(f"  Annualised Return  : {annualized_return:>10.2f} %")
    print(f"  Max Drawdown       : {max_dd * 100:>10.2f} %")
    print(f"  Volatility (Ann.)  : {volatility * 100:>10.2f} %")
    print(f"  Sharpe Ratio       : {sharpe:>10.4f}")

    return annualized_return, total_return, portfolio_series, metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    dca_series:  pd.Series,
    mom_series:  pd.Series,
    daily_data:  pd.DataFrame,
    save_path:   str = None,
) -> None:
    """
    Generate a 4-panel comparison chart:

    Panel 1 (top-left)  – Portfolio value growth over time.
    Panel 2 (top-right) – Monthly return distribution (histogram).
    Panel 3 (bot-left)  – Drawdown curves over time.
    Panel 4 (bot-right) – QQQ daily price with SMA-50 and SMA-200.

    Parameters
    ----------
    dca_series  : DCA portfolio value time series (monthly).
    mom_series  : Momentum portfolio value time series (monthly).
    daily_data  : Cleaned daily DataFrame with 'Close', 'SMA_50', 'SMA_200'.
    save_path   : Optional file path (.png) to save the figure.
    """
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"QQQ Investment Strategy Comparison  ({START_DATE} – {END_DATE})",
        fontsize=15,
        fontweight="bold",
    )

    # ── Panel 1: Portfolio value growth ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(dca_series.index, dca_series.values,
            label="DCA",      color="steelblue",  linewidth=2)
    ax.plot(mom_series.index, mom_series.values,
            label="Momentum", color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax.legend()

    # ── Panel 2: Monthly return distribution ─────────────────────────────────
    ax = axes[0, 1]
    dca_rets = dca_series.pct_change().dropna() * 100
    mom_rets = mom_series.pct_change().dropna() * 100
    ax.hist(dca_rets, bins=30, alpha=0.6, color="steelblue",
            edgecolor="white", label="DCA")
    ax.hist(mom_rets, bins=30, alpha=0.6, color="darkorange",
            edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("Monthly Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # ── Panel 3: Drawdown curves ─────────────────────────────────────────────
    ax = axes[1, 0]
    dca_dd = (dca_series - dca_series.cummax()) / dca_series.cummax() * 100
    mom_dd = (mom_series - mom_series.cummax()) / mom_series.cummax() * 100
    ax.fill_between(dca_series.index, dca_dd, 0,
                    alpha=0.4, color="steelblue",  label="DCA Drawdown")
    ax.fill_between(mom_series.index, mom_dd, 0,
                    alpha=0.4, color="darkorange", label="Momentum Drawdown")
    ax.set_title("Portfolio Drawdown Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()

    # ── Panel 4: QQQ price with moving averages ───────────────────────────────
    ax = axes[1, 1]
    ax.plot(daily_data.index, daily_data["Close"],
            label="QQQ Close", color="black",   linewidth=1,   alpha=0.75)
    ax.plot(daily_data.index, daily_data["SMA_50"],
            label="SMA 50",    color="royalblue", linewidth=1.5, linestyle="--")
    ax.plot(daily_data.index, daily_data["SMA_200"],
            label="SMA 200",   color="crimson",  linewidth=1.5, linestyle="-.")
    ax.set_title("QQQ Daily Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Chart saved → {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(dca_m: dict, mom_m: dict) -> None:
    """
    Print a formatted side-by-side comparison of both strategy metrics.

    Parameters
    ----------
    dca_m : Metrics dict returned by backtest_dca().
    mom_m : Metrics dict returned by momentum_strategy().
    """
    print("\n" + "═" * 62)
    print(f"  {'Metric':<26} {'DCA':>15} {'Momentum':>15}")
    print("═" * 62)

    rows = [
        ("Total Invested ($)",     dca_m["total_invested"],    mom_m["total_invested"],    ",.2f"),
        ("Ending Value ($)",       dca_m["ending_value"],      mom_m["ending_value"],      ",.2f"),
        ("Total Return (%)",       dca_m["total_return_pct"],  mom_m["total_return_pct"],  ".2f"),
        ("Annualised Return (%)",  dca_m["annualized_return"], mom_m["annualized_return"], ".2f"),
        ("Max Drawdown (%)",       dca_m["max_drawdown_pct"],  mom_m["max_drawdown_pct"],  ".2f"),
        ("Volatility Ann. (%)",    dca_m["volatility_pct"],    mom_m["volatility_pct"],    ".2f"),
        ("Sharpe Ratio",           dca_m["sharpe_ratio"],      mom_m["sharpe_ratio"],      ".4f"),
    ]

    for label, dval, mval, fmt in rows:
        print(
            f"  {label:<26} "
            f"{format(dval, fmt):>15} "
            f"{format(mval, fmt):>15}"
        )

    # Compute and display the return multiplier (ending / invested)
    dca_mult = dca_m["ending_value"] / dca_m["total_invested"]
    mom_mult = mom_m["ending_value"] / mom_m["total_invested"]
    print(
        f"  {'Return Multiplier (×)':<26} "
        f"{dca_mult:>15.2f} "
        f"{mom_mult:>15.2f}"
    )
    print("═" * 62)
    print("  Note: Expected return multiplier ≈ 2.7× for both strategies.")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 55)
    print("  QQQ Investment Strategy Analysis")
    print(f"  Period            : {START_DATE}  →  {END_DATE}")
    print(f"  Monthly Investment: ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Risk-Free Rate    : {RISK_FREE_RATE * 100:.1f} % per annum")
    print("═" * 55)

    # ── Run DCA backtest ─────────────────────────────────────────────────────
    dca_annual, dca_total, dca_portfolio, dca_metrics = backtest_dca(
        TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT
    )

    # ── Run Momentum backtest ────────────────────────────────────────────────
    mom_annual, mom_total, mom_portfolio, mom_metrics = momentum_strategy(
        TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT
    )

    # ── Print side-by-side comparison ────────────────────────────────────────
    print_comparison(dca_metrics, mom_metrics)

    # ── Re-fetch daily data for the chart (reuses the same source) ───────────
    raw_daily = fetch_data(TICKER, START_DATE, END_DATE)
    daily_df, _ = preprocess_data(raw_daily)

    # Chart save path: same directory as this script
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    chart_path  = os.path.join(script_dir, "qqq_strategy_analysis.png")

    # ── Generate visualisations ───────────────────────────────────────────────
    plot_results(dca_portfolio, mom_portfolio, daily_df, save_path=chart_path)

    print("\n[INFO] Analysis complete.")
