"""
QQQ Investment Strategy Analysis

This version follows the project requirement more closely:
- Use QQQ price data from 2014-01-01 to 2023-12-31.
- Use yfinance first; if it fails, use direct Stooq CSV download.
- Do not use pandas_datareader.
- Use the unadjusted Close price, not Adj Close, because the project strategy
  simulates buying QQQ shares directly from market prices.
- Calculate annualized return with XIRR / money-weighted IRR, not CAGR.
- Simulate:
    1. DCA: invest $500 every month and never sell.
    2. Momentum: invest $500 only when the momentum signal is positive; never sell.
- Report total return, annualized return, max drawdown, volatility, and Sharpe ratio.

Install:
    pip install pandas numpy matplotlib seaborn yfinance

Run:
    python qqq_investment_strategy_analysis_final.py
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


TICKER = "QQQ"
START_DATE = "2014-01-01"
END_DATE = "2023-12-31"
MONTHLY_INVESTMENT = 500.0
RISK_FREE_RATE = 0.0


@dataclass
class Result:
    """Stores the backtest result for one strategy."""
    strategy: str
    total_invested: float
    ending_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    portfolio_value: pd.Series


def get_daily_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily unadjusted close prices using yfinance or direct Stooq CSV."""
    try:
        prices = get_daily_close_from_yfinance(ticker, start, end)
        if not prices.empty:
            return prices
    except Exception as error:
        print(f"yfinance failed, trying Stooq. Reason: {error}")

    return get_daily_close_from_stooq(ticker, start, end)


def get_daily_close_from_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily unadjusted close prices from yfinance."""
    import yfinance as yf

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].dropna().sort_index()
    close.name = "close"
    return close


def get_daily_close_from_stooq(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily unadjusted close prices directly from Stooq CSV."""
    params = urlencode(
        {
            "s": f"{ticker.lower()}.us",
            "d1": pd.Timestamp(start).strftime("%Y%m%d"),
            "d2": pd.Timestamp(end).strftime("%Y%m%d"),
            "i": "d",
        }
    )
    url = f"https://stooq.com/q/d/l/?{params}"

    data = pd.read_csv(url)

    if data.empty or "Close" not in data.columns:
        raise ValueError("No usable data returned from Stooq.")

    data["Date"] = pd.to_datetime(data["Date"])
    close = data.set_index("Date")["Close"].dropna().sort_index()
    close.name = "close"
    return close


def get_monthly_close(daily_close: pd.Series) -> pd.Series:
    """Use the last available close price of each month."""
    return daily_close.resample("ME").last().dropna()


def xnpv(rate: float, cashflows: pd.Series) -> float:
    """Calculate NPV for dated cash flows."""
    first_date = cashflows.index[0]
    years = (cashflows.index - first_date).days / 365.25
    return float(np.sum(cashflows.values / (1.0 + rate) ** years))


def xirr(cashflows: pd.Series) -> float:
    """Calculate annualized money-weighted IRR for dated cash flows."""
    cashflows = cashflows.groupby(level=0).sum().sort_index()

    if not cashflows.lt(0).any() or not cashflows.gt(0).any():
        return np.nan

    low, high = -0.9999, 10.0
    low_value = xnpv(low, cashflows)
    high_value = xnpv(high, cashflows)

    if low_value * high_value > 0:
        return np.nan

    for _ in range(200):
        mid = (low + high) / 2.0
        mid_value = xnpv(mid, cashflows)

        if abs(mid_value) < 1e-8:
            return mid

        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value

    return (low + high) / 2.0


def calculate_result(strategy: str, portfolio: pd.Series, invested: pd.Series) -> Result:
    """Calculate performance and risk metrics."""
    portfolio = portfolio.dropna()
    invested = invested.reindex(portfolio.index).fillna(0.0)

    total_invested = float(invested.sum())
    ending_value = float(portfolio.iloc[-1])
    total_return = ending_value / total_invested - 1.0

    cashflows = -invested.copy()
    cashflows.iloc[-1] += ending_value
    annualized_return = xirr(cashflows)

    returns = portfolio.pct_change().dropna()
    volatility = float(returns.std(ddof=1) * np.sqrt(12))

    drawdown = portfolio / portfolio.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    monthly_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / 12.0) - 1.0
    sharpe_ratio = float(((returns - monthly_rf).mean() / returns.std(ddof=1)) * np.sqrt(12))

    return Result(
        strategy=strategy,
        total_invested=total_invested,
        ending_value=ending_value,
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        portfolio_value=portfolio,
    )


def backtest_dca(monthly_close: pd.Series) -> Result:
    """Invest $500 every month and hold all shares."""
    invested = pd.Series(MONTHLY_INVESTMENT, index=monthly_close.index)
    shares = (invested / monthly_close).cumsum()
    portfolio = shares * monthly_close

    return calculate_result("DCA", portfolio, invested)


def backtest_momentum(monthly_close: pd.Series) -> Result:
    """Invest $500 only when 6-month momentum, skipping the latest month, is positive."""
    monthly_returns = monthly_close.pct_change()
    momentum = monthly_returns.rolling(6).apply(lambda x: x[:-1].sum(), raw=False)
    signal = (momentum > 0).astype(int).where(momentum.notna(), 0)

    invested = MONTHLY_INVESTMENT * signal
    shares = (invested / monthly_close).cumsum()
    portfolio = shares * monthly_close

    return calculate_result("Momentum", portfolio, invested)


def make_table(results: list[Result]) -> pd.DataFrame:
    """Create a comparison table."""
    rows = []

    for result in results:
        rows.append(
            {
                "Strategy": result.strategy,
                "Total Invested": result.total_invested,
                "Ending Value": result.ending_value,
                "Total Return": result.total_return,
                "Annualized Return (XIRR)": result.annualized_return,
                "Max Drawdown": result.max_drawdown,
                "Volatility": result.volatility,
                "Sharpe Ratio": result.sharpe_ratio,
            }
        )

    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame) -> None:
    """Print formatted results."""
    output = table.copy()

    for column in ["Total Invested", "Ending Value"]:
        output[column] = output[column].map(lambda x: f"${x:,.2f}")

    for column in ["Total Return", "Annualized Return (XIRR)", "Max Drawdown", "Volatility"]:
        output[column] = output[column].map(lambda x: f"{x:.2%}")

    output["Sharpe Ratio"] = output["Sharpe Ratio"].map(lambda x: f"{x:.2f}")

    print("\nQQQ Investment Strategy Analysis")
    print("=" * 90)
    print(output.to_string(index=False))
    print("=" * 90)
    print("Annualized return uses XIRR / money-weighted IRR, not CAGR.")


def plot_results(results: list[Result]) -> None:
    """Plot portfolio values for both strategies."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 6))
    for result in results:
        plt.plot(result.portfolio_value.index, result.portfolio_value, label=result.strategy)

    plt.title("QQQ Strategy Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qqq_portfolio_values.png", dpi=150)
    plt.show()


def main() -> None:
    """Run the full analysis."""
    daily_close = get_daily_close(TICKER, START_DATE, END_DATE)
    monthly_close = get_monthly_close(daily_close)

    results = [
        backtest_dca(monthly_close),
        backtest_momentum(monthly_close),
    ]

    table = make_table(results)
    print_table(table)
    table.to_csv("qqq_strategy_results.csv", index=False)
    plot_results(results)


if __name__ == "__main__":
    main()
