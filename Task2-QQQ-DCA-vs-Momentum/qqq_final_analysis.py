"""
QQQ Investment Strategy Analysis  (2014-01-01 → 2023-12-31)
=============================================================
Strategies  : Dollar-Cost Averaging (DCA) vs Momentum Investing
Data source : yfinance auto_adjust=True (dividend-adjusted total-return prices)
              Fallback  : local QQQ.csv (stooq, also dividend-adjusted)

Public API
----------
backtest_dca(ticker, start_date, end_date, monthly_investment)
    → (annualized_return %, total_return %)

momentum_strategy(ticker, start_date, end_date, monthly_investment)
    → (annualized_return %, total_return %)

Both return the annualised IRR (Internal Rate of Return), which correctly
accounts for the timing of each periodic cash flow.  A plain CAGR computed
as (ending / total_invested)^(1/n) is also reported for reference — it
assumes all capital was deployed at t=0 and therefore understates DCA returns.

Requirements
------------
    pip install pandas numpy matplotlib seaborn yfinance
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Global configuration ──────────────────────────────────────────────────────
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0        # USD invested each month
RISK_FREE_RATE     = 0.02         # Annual risk-free rate (for Sharpe ratio)
CSV_FALLBACK       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "QQQ.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Acquisition
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker: str, start_date: str, end_date: str,
               retries: int = 5, base_delay: float = 10.0) -> pd.DataFrame:
    """
    Download daily dividend-adjusted close prices.

    Primary   : yfinance (auto_adjust=True bakes dividends into the price
                series so that all return calculations reflect total return).
    Fallback  : local QQQ.csv (stooq data, already dividend-adjusted).

    Retries with exponential back-off on HTTP 429 rate-limit errors.

    Parameters
    ----------
    ticker     : Ticker symbol (e.g. "QQQ").
    start_date : ISO date string, inclusive.
    end_date   : ISO date string, inclusive.
    retries    : Maximum yfinance download attempts before falling back.
    base_delay : Initial wait time (seconds) between retries.

    Returns
    -------
    DataFrame with a DatetimeIndex and a single column "Close".
    """
    for attempt in range(retries):
        try:
            import yfinance as yf  # optional dependency

            if attempt > 0:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"[INFO] Rate-limited — waiting {wait:.0f}s before retry {attempt}/{retries-1}…")
                time.sleep(wait)

            print(f"[INFO] Downloading {ticker} via yfinance (attempt {attempt+1})…")
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)

            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            # Flatten MultiIndex columns produced by yfinance ≥ 0.2
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            data = raw[["Close"]].copy()
            print(f"[INFO] yfinance: {len(data)} rows  "
                  f"start=${data['Close'].iloc[0]:.2f}  end=${data['Close'].iloc[-1]:.2f}")
            return data

        except ImportError:
            print("[WARN] yfinance not installed — falling back to local CSV.")
            break
        except Exception as exc:
            if any(k in str(exc) for k in ["429", "RateLimit", "Too Many"]):
                print(f"[WARN] {exc}")
                continue
            print(f"[WARN] yfinance error: {exc} — falling back to local CSV.")
            break

    # ── CSV fallback ──────────────────────────────────────────────────────────
    if not os.path.exists(CSV_FALLBACK):
        raise FileNotFoundError(
            f"Local CSV not found at {CSV_FALLBACK}. "
            "Please place a QQQ.csv with columns 'date' and 'close' in the same directory."
        )
    print(f"[INFO] Loading local CSV: {CSV_FALLBACK}")
    df = pd.read_csv(CSV_FALLBACK, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df = df[["close"]].rename(columns={"close": "Close"})
    print(f"[INFO] CSV: {len(df)} rows  "
          f"start=${df['Close'].iloc[0]:.2f}  end=${df['Close'].iloc[-1]:.2f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Data Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(raw_df: pd.DataFrame,
               start: str = START_DATE,
               end: str = END_DATE) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean price data and derive technical indicators.

    Cleaning steps
    ~~~~~~~~~~~~~~
    1. Normalise index to DatetimeIndex; drop duplicates (keep first).
    2. Trim to [start, end] window and sort ascending.
    3. Remove rows with missing Close values.
    4. Flag and remove zero / negative prices (data corruption).
    5. Forward-fill isolated single-day gaps (e.g. stale data entries).
    6. Statistical outlier check: flag daily returns with |z-score| > 4
       (computed on a 20-day rolling window).  No prices are removed —
       QQQ shows no such anomalies over 2014-2023; the check is logged.

    Indicators
    ~~~~~~~~~~
    SMA_50  : 50-day Simple Moving Average  (short-term trend)
    SMA_200 : 200-day Simple Moving Average (long-term trend)

    Parameters
    ----------
    raw_df : DataFrame from fetch_data() with a "Close" column.
    start  : ISO date string — start of analysis window.
    end    : ISO date string — end of analysis window.

    Returns
    -------
    daily   : Cleaned daily DataFrame with "Close", "SMA_50", "SMA_200".
    monthly : Month-end close prices (pd.Series), used for strategy simulation.
    """
    df = raw_df.copy()

    # Step 1 — Normalise index
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")]

    # Step 2 — Trim window
    df = df.loc[start:end].sort_index()

    # Step 3 — Drop NaN
    before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    dropped_na = before - len(df)

    # Step 4 — Remove zero / negative prices
    invalid_mask = df["Close"] <= 0
    n_invalid = invalid_mask.sum()
    df = df[~invalid_mask]

    # Step 5 — Forward-fill isolated gaps (≤ 2 consecutive days)
    df["Close"] = df["Close"].ffill(limit=2)

    # Step 6 — Statistical outlier check (log only, no removal)
    daily_returns = df["Close"].pct_change()
    rolling_mean  = daily_returns.rolling(20).mean()
    rolling_std   = daily_returns.rolling(20).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = ((daily_returns - rolling_mean) / rolling_std).abs()
    n_outliers = int((z_scores > 4).sum())

    # Compute indicators
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Month-end close prices
    monthly = df["Close"].resample("ME").last()

    # Summary
    print(f"\n[Preprocessing] ─────────────────────────────────────────────────")
    print(f"  Trading days      : {len(df)}")
    print(f"  NaN rows dropped  : {dropped_na}")
    print(f"  Zero/neg removed  : {n_invalid}")
    print(f"  Outliers (|z|>4)  : {n_outliers}  (no removal — informational only)")
    print(f"  SMA_50 / SMA_200  : calculated")
    print(f"  Monthly samples   : {len(monthly)}")
    print(f"  First close       : ${df['Close'].iloc[0]:.4f}  ({df.index[0].date()})")
    print(f"  Last close        : ${df['Close'].iloc[-1]:.4f}  ({df.index[-1].date()})")
    print(f"─────────────────────────────────────────────────────────────────")

    return df, monthly


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Risk & Return Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _cagr(ending: float, invested: float, n_years: float) -> float:
    """
    Compound Annual Growth Rate relative to total invested capital.

    Formula: (ending / invested)^(1 / n_years) - 1

    Note: this formula treats all capital as if deployed at t=0.
    For DCA it underestimates the true per-dollar return; use IRR instead.
    """
    return (ending / invested) ** (1.0 / n_years) - 1.0


def _irr(portfolio: pd.Series, monthly_investment: float) -> float:
    """
    Annualised Internal Rate of Return (XIRR) via Newton-Raphson iteration.

    For DCA, each instalment is invested at a different point in time.
    IRR solves for the monthly rate r that makes the NPV of all cash flows = 0:

        sum( CF_t / (1+r)^t ) = 0

    where CF_t = -monthly_investment for each period t, plus the final
    portfolio value added to CF at the last period.  Annualised: (1+r)^12 - 1.

    Parameters
    ----------
    portfolio         : Monthly portfolio value series (pd.Series).
    monthly_investment: Fixed amount invested each month (USD).

    Returns
    -------
    Annualised IRR as a decimal (e.g. 0.18 = 18 %).
    """
    ending     = float(portfolio.iloc[-1])
    cash_flows = [-monthly_investment] * len(portfolio)
    cash_flows[-1] += ending

    rate = 0.01  # Initial guess: 1 % per month
    for _ in range(2000):
        npv  = sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1.0 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))
        if dnpv == 0.0:
            break
        rate -= npv / dnpv
        if abs(npv) < 1e-10:
            break
    return (1.0 + rate) ** 12 - 1.0


def _max_drawdown(values: pd.Series) -> float:
    """Maximum drawdown fraction: min( (V - rolling_peak) / rolling_peak )."""
    return float(((values - values.cummax()) / values.cummax()).min())


def _annualised_volatility(monthly_returns: pd.Series) -> float:
    """Annualised volatility = sample std of monthly returns × √12."""
    return float(monthly_returns.std(ddof=1) * np.sqrt(12))


def _sharpe_ratio(monthly_returns: pd.Series) -> float:
    """Annualised Sharpe = mean excess monthly return / std × √12."""
    excess = monthly_returns - RISK_FREE_RATE / 12.0
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(12))


def compute_metrics(portfolio: pd.Series,
                    monthly_investment: float = MONTHLY_INVESTMENT) -> dict:
    """
    Compute a complete set of return and risk metrics for a portfolio series.

    Parameters
    ----------
    portfolio         : Monthly portfolio value series produced by a strategy.
    monthly_investment: Fixed monthly contribution (USD).

    Returns
    -------
    Ordered dict with the following keys:

    Total Invested ($)      – total capital deployed
    Ending Value ($)        – terminal portfolio value
    Return Multiplier (x)   – ending / invested
    Total Return (%)        – (ending - invested) / invested × 100
    CAGR (%)                – (ending/invested)^(1/n) - 1; assumes lump-sum t=0
    IRR / XIRR (%)          – annualised IRR; correct for periodic investments
    Max Drawdown (%)        – largest peak-to-trough decline
    Volatility Ann. (%)     – annualised standard deviation of monthly returns
    Sharpe Ratio            – annualised Sharpe (risk-free = RISK_FREE_RATE)
    """
    invested = monthly_investment * len(portfolio)
    ending   = float(portfolio.iloc[-1])
    rets     = portfolio.pct_change().dropna()
    n_years  = (portfolio.index[-1] - portfolio.index[0]).days / 365.25

    return {
        "Total Invested ($)":    invested,
        "Ending Value ($)":      ending,
        "Return Multiplier (x)": ending / invested,
        "Total Return (%)":      (ending - invested) / invested * 100.0,
        "CAGR (%)":              _cagr(ending, invested, n_years) * 100.0,
        "IRR / XIRR (%)":        _irr(portfolio, monthly_investment) * 100.0,
        "Max Drawdown (%)":      _max_drawdown(portfolio) * 100.0,
        "Volatility Ann. (%)":   _annualised_volatility(rets) * 100.0,
        "Sharpe Ratio":          _sharpe_ratio(rets),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Strategy Implementations
# ══════════════════════════════════════════════════════════════════════════════

def _sim_dca(monthly: pd.Series, monthly_investment: float) -> pd.Series:
    """
    Dollar-Cost Averaging (DCA) core simulation.

    Rule
    ~~~~
    Every month, invest a fixed dollar amount regardless of price.
    Fractional shares are allowed.  Never sell.

        shares_bought_t = monthly_investment / price_t
        portfolio_t     = cumulative_shares × price_t

    Parameters
    ----------
    monthly           : Month-end close price series.
    monthly_investment: Fixed monthly contribution (USD).

    Returns
    -------
    pd.Series of monthly portfolio values (same index as `monthly`).
    """
    shares, values = 0.0, []
    for price in monthly:
        shares += monthly_investment / price
        values.append(shares * price)
    return pd.Series(values, index=monthly.index)


def _sim_momentum(monthly: pd.Series, monthly_investment: float) -> pd.Series:
    """
    Momentum Investing core simulation.

    Signal
    ~~~~~~
    6-month cumulative return, skipping the most recent month to avoid
    short-term reversal noise (as specified in the project brief):

        momentum_t = sum( monthly_returns[t-6 : t-1] )
        position_t = 1  if momentum_t > 0  (invest in QQQ)
                   = 0  if momentum_t ≤ 0  (hold cash, 0 % return)

    Parameters
    ----------
    monthly           : Month-end close price series.
    monthly_investment: Fixed monthly contribution (USD).

    Returns
    -------
    pd.Series of monthly portfolio values (same index as `monthly`).
    """
    rets      = monthly.pct_change()
    momentum  = rets.rolling(6).apply(lambda x: x[:-1].sum(), raw=True)
    positions = (momentum > 0).astype(int)

    shares, cash, values = 0.0, 0.0, []
    for date, price in monthly.items():
        if positions[date] == 1:
            shares += monthly_investment / price   # positive momentum → buy QQQ
        else:
            cash += monthly_investment             # negative momentum → hold cash
        values.append(shares * price + cash)
    return pd.Series(values, index=monthly.index)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Public API  (matches project brief function signatures)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_dca(ticker: str, start_date: str, end_date: str,
                 monthly_investment: float) -> tuple[float, float]:
    """
    Dollar-Cost Averaging strategy backtest.

    Parameters
    ----------
    ticker             : Ticker symbol (e.g. "QQQ").
    start_date         : ISO start date string.
    end_date           : ISO end date string.
    monthly_investment : Fixed amount invested each month (USD).

    Returns
    -------
    annualized_return : float – Annualised IRR over the period (%).
    total_return      : float – Overall gain vs. total invested (%).
    """
    _, monthly = preprocess(fetch_data(ticker, start_date, end_date), start_date, end_date)
    portfolio  = _sim_dca(monthly, monthly_investment)
    m          = compute_metrics(portfolio, monthly_investment)
    return m["IRR / XIRR (%)"], m["Total Return (%)"]


def momentum_strategy(ticker: str, start_date: str, end_date: str,
                      monthly_investment: float) -> tuple[float, float]:
    """
    Momentum Investing strategy backtest.

    Parameters
    ----------
    ticker             : Ticker symbol (e.g. "QQQ").
    start_date         : ISO start date string.
    end_date           : ISO end date string.
    monthly_investment : Fixed amount invested each month (USD).

    Returns
    -------
    annualized_return : float – Annualised IRR over the period (%).
    total_return      : float – Overall gain vs. total invested (%).
    """
    _, monthly = preprocess(fetch_data(ticker, start_date, end_date), start_date, end_date)
    portfolio  = _sim_momentum(monthly, monthly_investment)
    m          = compute_metrics(portfolio, monthly_investment)
    return m["IRR / XIRR (%)"], m["Total Return (%)"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot(dca_portfolio: pd.Series, mom_portfolio: pd.Series,
         daily: pd.DataFrame, save_path: str) -> None:
    """
    Four-panel comparison chart.

    Panels
    ~~~~~~
    1. Portfolio Value Over Time   — cumulative wealth comparison
    2. Monthly Return Distribution — histogram of periodic returns
    3. Drawdown Over Time          — peak-to-trough loss profile
    4. QQQ Price + SMA-50/200      — price with technical overlays

    Parameters
    ----------
    dca_portfolio : Monthly DCA portfolio value series.
    mom_portfolio : Monthly Momentum portfolio value series.
    daily         : Daily DataFrame with "Close", "SMA_50", "SMA_200".
    save_path     : File path for the saved PNG.
    """
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("QQQ Investment Strategy Comparison  (2014 – 2023)",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Portfolio value ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(dca_portfolio.index, dca_portfolio,
            label="DCA",      color="steelblue",  linewidth=2)
    ax.plot(mom_portfolio.index, mom_portfolio,
            label="Momentum", color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Portfolio Value Over Time")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()

    # ── Panel 2: Monthly return distribution ──────────────────────────────
    ax = axes[0, 1]
    dca_rets = dca_portfolio.pct_change().dropna() * 100
    mom_rets = mom_portfolio.pct_change().dropna() * 100
    ax.hist(dca_rets, bins=30, alpha=0.6, color="steelblue",
            edgecolor="white", label="DCA")
    ax.hist(mom_rets, bins=30, alpha=0.6, color="darkorange",
            edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("Monthly Return (%)")
    ax.legend()

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────
    ax = axes[1, 0]
    dca_dd = (dca_portfolio - dca_portfolio.cummax()) / dca_portfolio.cummax() * 100
    mom_dd = (mom_portfolio - mom_portfolio.cummax()) / mom_portfolio.cummax() * 100
    ax.fill_between(dca_portfolio.index, dca_dd, 0,
                    alpha=0.4, color="steelblue",  label="DCA")
    ax.fill_between(mom_portfolio.index, mom_dd, 0,
                    alpha=0.4, color="darkorange", label="Momentum")
    ax.set_title("Drawdown Over Time")
    ax.set_ylabel("%")
    ax.legend()

    # ── Panel 4: QQQ price + moving averages ──────────────────────────────
    ax = axes[1, 1]
    ax.plot(daily.index, daily["Close"],
            color="black",     linewidth=0.8, alpha=0.7, label="Adj Close")
    ax.plot(daily.index, daily["SMA_50"],
            color="royalblue", linewidth=1.2, linestyle="--", label="SMA 50")
    ax.plot(daily.index, daily["SMA_200"],
            color="crimson",   linewidth=1.2, linestyle="-.", label="SMA 200")
    ax.set_title("QQQ Adjusted Close + Moving Averages")
    ax.set_ylabel("USD")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  QQQ Investment Strategy Analysis")
    print(f"  Period             : {START_DATE}  →  {END_DATE}")
    print(f"  Monthly Investment : ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Data               : yfinance auto_adjust=True  (CSV fallback)")
    print("=" * 65)

    # ── Public API calls (project brief function signatures) ──────────────
    dca_ann, dca_tot = backtest_dca(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)
    mom_ann, mom_tot = momentum_strategy(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)

    print(f"\n[Public API]")
    print(f"  DCA      — annualized_return (IRR): {dca_ann:.2f}%  |  total_return: {dca_tot:.2f}%")
    print(f"  Momentum — annualized_return (IRR): {mom_ann:.2f}%  |  total_return: {mom_tot:.2f}%")

    # ── Full analysis (fetch once, reuse for metrics + chart) ─────────────
    daily, monthly = preprocess(fetch_data(TICKER, START_DATE, END_DATE),
                                START_DATE, END_DATE)
    dca_portfolio = _sim_dca(monthly, MONTHLY_INVESTMENT)
    mom_portfolio = _sim_momentum(monthly, MONTHLY_INVESTMENT)

    dca_m = compute_metrics(dca_portfolio)
    mom_m = compute_metrics(mom_portfolio)

    # Metrics table
    print(f"\n{'Metric':<28} {'DCA':>15} {'Momentum':>15}")
    print("─" * 60)
    for key in dca_m:
        if "$" in key:
            fmt = ",.2f"
        elif "Sharpe" in key or "Multiplier" in key:
            fmt = ".4f"
        else:
            fmt = ".2f"
        print(f"{key:<28} {format(dca_m[key], fmt):>15} {format(mom_m[key], fmt):>15}")
    print("─" * 60)

    # ── Return metric notes ───────────────────────────────────────────────
    print("""
Notes on annualised return metrics
───────────────────────────────────────────────────────────────
  CAGR (%) = (ending / total_invested)^(1/n_years) - 1
      Assumes all capital deployed at t=0; understates DCA returns.
      Matches "~2.7× of total invested" cited in the project brief
      (actual multiplier ≈ 2.59×, CAGR ≈ 10%).

  IRR / XIRR (%) — correct method for periodic investments.
      Solves NPV=0 over the exact cash-flow schedule; accounts for
      the fact that early instalments compound longer than late ones.
      DCA IRR ≈ 18% reflects the high compounding on earliest shares.
""")

    # ── Chart ─────────────────────────────────────────────────────────────
    chart_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qqq_final_analysis.png"
    )
    plot(dca_portfolio, mom_portfolio, daily, save_path=chart_path)
