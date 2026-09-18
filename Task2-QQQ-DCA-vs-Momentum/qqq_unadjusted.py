"""
QQQ DCA vs Momentum  —  Unadjusted (raw) price edition  (2014-01-01 → 2023-12-31)
===================================================================================
Hypothesis  : Using raw (unadjusted) close prices instead of dividend-adjusted
              prices to see whether DCA annualised IRR changes toward ~14%.

Key difference vs qqq_final_analysis.py
----------------------------------------
  auto_adjust = False  →  yfinance returns the ACTUAL historical close price
                          (what you would have paid on that day).
                          Dividends are NOT baked into the price series.

  auto_adjust = True   →  yfinance retroactively scales all historical prices
                          downward to reflect dividend reinvestment, making the
                          starting price appear lower and total-return CAGR larger.

Why this matters for DCA
-------------------------
  auto_adjust=True:  Jan-2014 price ≈ $79  (adjusted)  → more shares per $500
  auto_adjust=False: Jan-2014 price ≈ $87  (actual)    → fewer shares per $500

  → Unadjusted prices → fewer cheap early shares → lower ending value → lower IRR.

NOTE  This file requires yfinance to be installed:
          pip install yfinance
      If yfinance is unavailable the script exits with a clear error message.
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

# ── Config ────────────────────────────────────────────────────────────────────
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0
RISK_FREE_RATE     = 0.02


# ══════════════════════════════════════════════════════════════════════════════
# Data Acquisition — unadjusted close prices
# ══════════════════════════════════════════════════════════════════════════════

def fetch_unadjusted(ticker: str, start_date: str, end_date: str,
                     retries: int = 5, base_delay: float = 10.0) -> pd.DataFrame:
    """
    Download UNADJUSTED daily close prices via yfinance (auto_adjust=False).

    With auto_adjust=False:
      - 'Close'     = actual historical close price (what investors paid)
      - 'Adj Close' = dividend-adjusted price (included for comparison)

    We use 'Close' (unadjusted) for the strategy simulation.
    Dividends are ignored — this represents the pure price-appreciation return.

    Returns
    -------
    DataFrame with columns: Close (unadjusted), Adj_Close (adjusted), DatetimeIndex.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed.\n"
            "Run:  pip install yfinance\n"
            "Then re-run this script."
        )

    for attempt in range(retries):
        try:
            if attempt > 0:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"[INFO] Rate-limited — waiting {wait:.0f}s before retry {attempt}/{retries-1}…")
                time.sleep(wait)

            print(f"[INFO] Downloading {ticker} via yfinance auto_adjust=False (attempt {attempt+1})…")
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=False, progress=False)

            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            # Flatten MultiIndex columns (yfinance >= 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Keep both Close (unadjusted) and Adj Close for reference
            cols = {}
            if "Close" in raw.columns:
                cols["Close"] = raw["Close"]
            if "Adj Close" in raw.columns:
                cols["Adj_Close"] = raw["Adj Close"]

            data = pd.DataFrame(cols)
            data.index = pd.to_datetime(data.index)

            print(f"[INFO] Rows: {len(data)}")
            print(f"[INFO] Unadjusted  Close — start: ${data['Close'].iloc[0]:.4f}   "
                  f"end: ${data['Close'].iloc[-1]:.4f}")
            if "Adj_Close" in data.columns:
                print(f"[INFO] Adjusted    Close — start: ${data['Adj_Close'].iloc[0]:.4f}   "
                      f"end: ${data['Adj_Close'].iloc[-1]:.4f}")
            return data

        except Exception as exc:
            if any(k in str(exc) for k in ["429", "RateLimit", "Too Many"]):
                print(f"[WARN] {exc}")
                continue
            raise RuntimeError(f"yfinance download failed: {exc}") from exc

    raise RuntimeError(f"Download failed after {retries} attempts.")


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(raw_df: pd.DataFrame,
               start: str = START_DATE,
               end: str = END_DATE) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean data and compute indicators.  Same pipeline as qqq_final_analysis.py
    but operating on the unadjusted 'Close' column.
    """
    df = raw_df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[start:end].sort_index()

    # Drop NaN rows
    before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    dropped_na = before - len(df)

    # Remove zero / negative prices
    invalid = df["Close"] <= 0
    df = df[~invalid]

    # Forward-fill isolated gaps
    df["Close"] = df["Close"].ffill(limit=2)

    # Statistical outlier check (informational only)
    daily_ret    = df["Close"].pct_change()
    roll_mean    = daily_ret.rolling(20).mean()
    roll_std     = daily_ret.rolling(20).std()
    z_scores     = ((daily_ret - roll_mean) / roll_std).abs()
    n_outliers   = int((z_scores > 4).sum())

    # Indicators
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    monthly = df["Close"].resample("ME").last()

    print(f"\n[Preprocessing] ─────────────────────────────────────────────────")
    print(f"  Trading days      : {len(df)}")
    print(f"  NaN dropped       : {dropped_na}")
    print(f"  Zero/neg removed  : {int(invalid.sum())}")
    print(f"  Outliers (|z|>4)  : {n_outliers}")
    print(f"  Monthly samples   : {len(monthly)}")
    print(f"  First close (unaj): ${df['Close'].iloc[0]:.4f}  ({df.index[0].date()})")
    print(f"  Last  close (unaj): ${df['Close'].iloc[-1]:.4f}  ({df.index[-1].date()})")
    print(f"─────────────────────────────────────────────────────────────────")
    return df, monthly


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def _irr(portfolio: pd.Series, monthly_investment: float) -> float:
    """Annualised IRR (XIRR) via Newton-Raphson."""
    ending     = float(portfolio.iloc[-1])
    cash_flows = [-monthly_investment] * len(portfolio)
    cash_flows[-1] += ending
    r = 0.01
    for _ in range(2000):
        npv  = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows))
        if dnpv == 0:
            break
        r -= npv / dnpv
        if abs(npv) < 1e-10:
            break
    return (1 + r) ** 12 - 1


def _cagr(ending: float, invested: float, n_years: float) -> float:
    return (ending / invested) ** (1.0 / n_years) - 1.0


def _max_drawdown(values: pd.Series) -> float:
    return float(((values - values.cummax()) / values.cummax()).min())


def _sharpe(rets: pd.Series) -> float:
    excess = rets - RISK_FREE_RATE / 12
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(12))


def compute_metrics(portfolio: pd.Series,
                    monthly_investment: float = MONTHLY_INVESTMENT) -> dict:
    invested = monthly_investment * len(portfolio)
    ending   = float(portfolio.iloc[-1])
    rets     = portfolio.pct_change().dropna()
    n_years  = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    return {
        "Total Invested ($)":    invested,
        "Ending Value ($)":      ending,
        "Return Multiplier (x)": ending / invested,
        "Total Return (%)":      (ending - invested) / invested * 100,
        "CAGR (%)":              _cagr(ending, invested, n_years) * 100,
        "IRR / XIRR (%)":        _irr(portfolio, monthly_investment) * 100,
        "Max Drawdown (%)":      _max_drawdown(portfolio) * 100,
        "Volatility Ann. (%)":   rets.std(ddof=1) * np.sqrt(12) * 100,
        "Sharpe Ratio":          _sharpe(rets),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Strategy simulations
# ══════════════════════════════════════════════════════════════════════════════

def _sim_dca(monthly: pd.Series, monthly_investment: float) -> pd.Series:
    """DCA: invest fixed amount every month, never sell."""
    shares, values = 0.0, []
    for price in monthly:
        shares += monthly_investment / price
        values.append(shares * price)
    return pd.Series(values, index=monthly.index)


def _sim_momentum(monthly: pd.Series, monthly_investment: float) -> pd.Series:
    """Momentum: invest when 6-month momentum (skip last month) > 0, else hold cash."""
    rets      = monthly.pct_change()
    momentum  = rets.rolling(6).apply(lambda x: x[:-1].sum(), raw=True)
    positions = (momentum > 0).astype(int)

    shares, cash, values = 0.0, 0.0, []
    for date, price in monthly.items():
        if positions[date] == 1:
            shares += monthly_investment / price
        else:
            cash   += monthly_investment
        values.append(shares * price + cash)
    return pd.Series(values, index=monthly.index)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def backtest_dca(ticker: str, start_date: str, end_date: str,
                 monthly_investment: float) -> tuple[float, float]:
    """DCA backtest on unadjusted prices.  Returns (IRR %, total_return %)."""
    _, monthly = preprocess(fetch_unadjusted(ticker, start_date, end_date),
                            start_date, end_date)
    m = compute_metrics(_sim_dca(monthly, monthly_investment), monthly_investment)
    return m["IRR / XIRR (%)"], m["Total Return (%)"]


def momentum_strategy(ticker: str, start_date: str, end_date: str,
                      monthly_investment: float) -> tuple[float, float]:
    """Momentum backtest on unadjusted prices.  Returns (IRR %, total_return %)."""
    _, monthly = preprocess(fetch_unadjusted(ticker, start_date, end_date),
                            start_date, end_date)
    m = compute_metrics(_sim_momentum(monthly, monthly_investment), monthly_investment)
    return m["IRR / XIRR (%)"], m["Total Return (%)"]


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot(dca_p: pd.Series, mom_p: pd.Series,
         daily: pd.DataFrame, save_path: str) -> None:
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("QQQ Investment Strategy  —  Unadjusted Prices  (2014 – 2023)",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(dca_p.index, dca_p, label="DCA",      color="steelblue",  linewidth=2)
    ax.plot(mom_p.index, mom_p, label="Momentum", color="darkorange", linewidth=2, linestyle="--")
    ax.set_title("Portfolio Value Over Time"); ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()

    ax = axes[0, 1]
    ax.hist(dca_p.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="steelblue",  edgecolor="white", label="DCA")
    ax.hist(mom_p.pct_change().dropna() * 100, bins=30, alpha=0.6,
            color="darkorange", edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution"); ax.set_xlabel("%"); ax.legend()

    ax = axes[1, 0]
    dca_dd = (dca_p - dca_p.cummax()) / dca_p.cummax() * 100
    mom_dd = (mom_p - mom_p.cummax()) / mom_p.cummax() * 100
    ax.fill_between(dca_p.index, dca_dd, 0, alpha=0.4, color="steelblue",  label="DCA")
    ax.fill_between(mom_p.index, mom_dd, 0, alpha=0.4, color="darkorange", label="Momentum")
    ax.set_title("Drawdown Over Time"); ax.set_ylabel("%"); ax.legend()

    ax = axes[1, 1]
    ax.plot(daily.index, daily["Close"],   color="black",     linewidth=0.8, alpha=0.7, label="Unadj Close")
    ax.plot(daily.index, daily["SMA_50"],  color="royalblue", linewidth=1.2, linestyle="--", label="SMA 50")
    ax.plot(daily.index, daily["SMA_200"], color="crimson",   linewidth=1.2, linestyle="-.", label="SMA 200")
    ax.set_title("QQQ Unadjusted Close + Moving Averages"); ax.set_ylabel("USD"); ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  QQQ Unadjusted Price Analysis")
    print(f"  Period             : {START_DATE}  →  {END_DATE}")
    print(f"  Monthly Investment : ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Price series       : UNADJUSTED close (auto_adjust=False)")
    print("=" * 65)

    # Public API
    dca_ann, dca_tot = backtest_dca(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)
    mom_ann, mom_tot = momentum_strategy(TICKER, START_DATE, END_DATE, MONTHLY_INVESTMENT)

    print(f"\n[Public API]")
    print(f"  DCA      — IRR: {dca_ann:.2f}%  |  Total Return: {dca_tot:.2f}%")
    print(f"  Momentum — IRR: {mom_ann:.2f}%  |  Total Return: {mom_tot:.2f}%")

    # Full metrics
    raw   = fetch_unadjusted(TICKER, START_DATE, END_DATE)
    daily, monthly = preprocess(raw, START_DATE, END_DATE)
    dca_portfolio  = _sim_dca(monthly, MONTHLY_INVESTMENT)
    mom_portfolio  = _sim_momentum(monthly, MONTHLY_INVESTMENT)

    dca_m = compute_metrics(dca_portfolio)
    mom_m = compute_metrics(mom_portfolio)

    print(f"\n{'Metric':<28} {'DCA':>15} {'Momentum':>15}")
    print("─" * 60)
    for key in dca_m:
        fmt = ",.2f" if "$" in key else ".4f" if "Sharpe" in key or "Multiplier" in key else ".2f"
        print(f"{key:<28} {format(dca_m[key], fmt):>15} {format(mom_m[key], fmt):>15}")
    print("─" * 60)

    # Price comparison note
    if "Adj_Close" in raw.columns:
        adj_monthly = raw["Adj_Close"].resample("ME").last().loc[START_DATE:END_DATE]
        unaj_start  = monthly.iloc[0]
        adj_start   = adj_monthly.iloc[0]
        print(f"\n[Price comparison — Jan 2014 month-end]")
        print(f"  Unadjusted : ${unaj_start:.4f}")
        print(f"  Adjusted   : ${adj_start:.4f}")
        print(f"  Difference : {(unaj_start - adj_start) / adj_start * 100:+.2f}% "
              f"(unadjusted is higher → fewer early shares bought → lower IRR)")

    chart_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qqq_unadjusted.png"
    )
    plot(dca_portfolio, mom_portfolio, daily, save_path=chart_path)
