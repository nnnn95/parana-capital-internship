"""
QQQ Investment Strategy Analysis

This script implements the project requirements from the QQQ Investment Strategy Analysis brief:
- Download QQQ historical data from 2014-01-01 to 2023-12-31.
- Use yfinance first, then fall back to Stooq through pandas_datareader.
- Simulate two strategies:
  1. Dollar-Cost Averaging (DCA): invest $500 monthly and never sell.
  2. Momentum investing: invest $500 in months when 6-month momentum, skipping the most recent month, is positive.
     Existing shares are held; new monthly cash is only invested when the momentum signal is positive.
- Calculate total return and annualized return.
- Annualized return is calculated with the IRR/XIRR-style cash-flow method, not CAGR.
- Calculate optional risk metrics: maximum drawdown, annualized volatility, and Sharpe ratio.
- Create comparison charts.

Required packages:
    pandas, numpy, matplotlib, seaborn, yfinance
Optional fallback package:
    pandas_datareader

Run:
    python qqq_investment_strategy_analysis.py
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


@dataclass
class StrategyResult:
    """Container for strategy outputs."""
    name: str
    annualized_return: float
    total_return: float
    ending_value: float
    total_invested: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    monthly_portfolio: pd.Series
    monthly_contributions: pd.Series


def download_qqq_data(
    ticker: str = "QQQ",
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Download adjusted historical price data.

    The script tries yfinance first and uses Stooq as a fallback.
    Stooq often returns data in descending date order, so this function sorts the index.
    """
    end_for_download = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        import yfinance as yf

        data = yf.download(
            ticker,
            start=start_date,
            end=end_for_download,
            auto_adjust=False,
            progress=False,
        )

        if data is not None and not data.empty:
            data = data.copy()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
            output = data[[price_col]].rename(columns={price_col: "Adj Close"})
            output.index = pd.to_datetime(output.index)
            output = output.sort_index().dropna()
            return output

    except Exception as exc:
        print(f"yfinance download failed: {exc}")

    try:
        from pandas_datareader import data as web

        stooq_ticker = f"{ticker}.US"
        data = web.DataReader(stooq_ticker, "stooq", start_date, end_date)

        if data is None or data.empty:
            raise ValueError("Stooq returned no data.")

        data = data.sort_index()
        price_col = "Close"
        output = data[[price_col]].rename(columns={price_col: "Adj Close"})
        output.index = pd.to_datetime(output.index)
        output = output.dropna()
        return output

    except Exception as exc:
        raise RuntimeError(
            "Unable to download QQQ data from yfinance or Stooq. "
            "Please check your internet connection and package installation."
        ) from exc


def prepare_monthly_prices(price_data: pd.DataFrame) -> pd.Series:
    """Convert daily adjusted prices to month-end adjusted prices."""
    monthly_prices = price_data["Adj Close"].resample("ME").last().dropna()
    monthly_prices.name = "QQQ Monthly Price"
    return monthly_prices


def solve_irr(
    cash_flows: list[float],
    guess: float = 0.01,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> float:
    """
    Solve periodic IRR using Newton-Raphson with a bisection fallback.

    Cash flows are assumed to be equally spaced monthly.
    Negative values represent invested cash. Positive values represent cash received.
    """
    if not any(cf < 0 for cf in cash_flows) or not any(cf > 0 for cf in cash_flows):
        return np.nan

    def npv(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** idx) for idx, cf in enumerate(cash_flows))

    def d_npv(rate: float) -> float:
        return sum(
            -idx * cf / ((1.0 + rate) ** (idx + 1))
            for idx, cf in enumerate(cash_flows)
            if idx > 0
        )

    rate = guess
    for _ in range(max_iterations):
        if rate <= -0.999999:
            break
        value = npv(rate)
        derivative = d_npv(rate)
        if abs(derivative) < 1e-12:
            break
        new_rate = rate - value / derivative
        if abs(new_rate - rate) < tolerance:
            return new_rate
        rate = new_rate

    # Bisection fallback over a practical monthly-rate range.
    low, high = -0.9999, 10.0
    low_value, high_value = npv(low), npv(high)

    if low_value * high_value > 0:
        return np.nan

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        mid_value = npv(mid)

        if abs(mid_value) < tolerance:
            return mid

        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value

    return (low + high) / 2.0


def annualized_return_from_irr(
    monthly_contributions: pd.Series,
    ending_value: float,
) -> float:
    """
    Calculate annualized return using monthly IRR, not CAGR.

    The monthly cash-flow sequence is:
    - each monthly investment as a negative cash flow
    - final portfolio value as a positive terminal cash flow
    """
    cash_flows = (-monthly_contributions).astype(float).tolist()
    cash_flows.append(float(ending_value))

    monthly_irr = solve_irr(cash_flows)

    if np.isnan(monthly_irr):
        return np.nan

    return (1.0 + monthly_irr) ** 12 - 1.0


def calculate_total_return(ending_value: float, total_invested: float) -> float:
    """Calculate total return relative to total invested capital."""
    if total_invested <= 0:
        return np.nan
    return ending_value / total_invested - 1.0


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate maximum drawdown from a portfolio value series."""
    values = portfolio_values.dropna()
    if values.empty:
        return np.nan

    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def calculate_volatility(portfolio_values: pd.Series, periods_per_year: int = 12) -> float:
    """Calculate annualized volatility from monthly portfolio returns."""
    returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized Sharpe ratio from monthly portfolio returns."""
    returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan

    periodic_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - periodic_risk_free_rate
    volatility = returns.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return float(excess_returns.mean() / volatility * math.sqrt(periods_per_year))


def backtest_dca(
    ticker: str = "QQQ",
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    monthly_investment: float = 500.0,
    price_loader: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
) -> StrategyResult:
    """
    Backtest the DCA strategy.

    This function keeps the requested project-style signature and returns
    annualized return and total return inside the StrategyResult object.
    """
    loader = price_loader or download_qqq_data
    price_data = loader(ticker, start_date, end_date)
    monthly_prices = prepare_monthly_prices(price_data)

    contributions = pd.Series(monthly_investment, index=monthly_prices.index, dtype=float)
    shares_bought = contributions / monthly_prices
    cumulative_shares = shares_bought.cumsum()
    portfolio_values = cumulative_shares * monthly_prices

    ending_value = float(portfolio_values.iloc[-1])
    total_invested = float(contributions.sum())

    return StrategyResult(
        name="DCA",
        annualized_return=annualized_return_from_irr(contributions, ending_value),
        total_return=calculate_total_return(ending_value, total_invested),
        ending_value=ending_value,
        total_invested=total_invested,
        max_drawdown=calculate_max_drawdown(portfolio_values),
        volatility=calculate_volatility(portfolio_values),
        sharpe_ratio=calculate_sharpe_ratio(portfolio_values),
        monthly_portfolio=portfolio_values,
        monthly_contributions=contributions,
    )


def momentum_strategy(
    ticker: str = "QQQ",
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    monthly_investment: float = 500.0,
    price_loader: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
) -> StrategyResult:
    """
    Backtest the momentum investing strategy.

    The signal follows the project reference:
        momentum = monthly_returns.rolling(window=6).apply(lambda x: x[:-1].sum())
        positions = (momentum > 0).astype(int)

    Interpretation:
    - If the signal is 1, invest the scheduled monthly amount into QQQ.
    - If the signal is 0, keep that month's scheduled capital in cash.
    - Previously purchased shares are not sold.
    """
    loader = price_loader or download_qqq_data
    price_data = loader(ticker, start_date, end_date)
    monthly_prices = prepare_monthly_prices(price_data)

    monthly_returns = monthly_prices.pct_change()
    momentum = monthly_returns.rolling(window=6).apply(lambda x: x[:-1].sum(), raw=False)
    positions = (momentum > 0).astype(int)

    # Avoid investing before the momentum window is available.
    positions = positions.where(momentum.notna(), 0)

    contributions = pd.Series(monthly_investment, index=monthly_prices.index, dtype=float)
    invested_contributions = contributions * positions

    shares_bought = invested_contributions / monthly_prices
    cumulative_shares = shares_bought.cumsum()
    portfolio_values = cumulative_shares * monthly_prices

    ending_value = float(portfolio_values.iloc[-1])
    total_invested = float(invested_contributions.sum())

    return StrategyResult(
        name="Momentum",
        annualized_return=annualized_return_from_irr(invested_contributions, ending_value),
        total_return=calculate_total_return(ending_value, total_invested),
        ending_value=ending_value,
        total_invested=total_invested,
        max_drawdown=calculate_max_drawdown(portfolio_values),
        volatility=calculate_volatility(portfolio_values),
        sharpe_ratio=calculate_sharpe_ratio(portfolio_values),
        monthly_portfolio=portfolio_values,
        monthly_contributions=invested_contributions,
    )


def build_results_table(results: list[StrategyResult]) -> pd.DataFrame:
    """Create a formatted strategy comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "Strategy": result.name,
                "Total Invested": result.total_invested,
                "Ending Value": result.ending_value,
                "Total Return": result.total_return,
                "Annualized Return (IRR)": result.annualized_return,
                "Max Drawdown": result.max_drawdown,
                "Volatility": result.volatility,
                "Sharpe Ratio": result.sharpe_ratio,
            }
        )

    return pd.DataFrame(rows)


def plot_strategy_comparison(results: list[StrategyResult]) -> None:
    """Create charts comparing the strategies."""
    sns.set_theme(style="whitegrid")

    portfolio_df = pd.concat(
        [result.monthly_portfolio.rename(result.name) for result in results],
        axis=1,
    )

    plt.figure(figsize=(12, 6))
    for column in portfolio_df.columns:
        plt.plot(portfolio_df.index, portfolio_df[column], label=column)

    plt.title("QQQ Strategy Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qqq_strategy_portfolio_value.png", dpi=150)
    plt.show()

    metrics = build_results_table(results)
    plot_data = metrics.set_index("Strategy")[["Total Return", "Annualized Return (IRR)", "Max Drawdown"]]

    plt.figure(figsize=(10, 6))
    plot_data.plot(kind="bar", ax=plt.gca())
    plt.title("QQQ Strategy Return and Drawdown Comparison")
    plt.xlabel("Strategy")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("qqq_strategy_metric_comparison.png", dpi=150)
    plt.show()


def print_results_table(results_table: pd.DataFrame) -> None:
    """Print a readable results table."""
    display_table = results_table.copy()

    currency_columns = ["Total Invested", "Ending Value"]
    percent_columns = ["Total Return", "Annualized Return (IRR)", "Max Drawdown", "Volatility"]

    for column in currency_columns:
        display_table[column] = display_table[column].map(lambda x: f"${x:,.2f}")

    for column in percent_columns:
        display_table[column] = display_table[column].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")

    display_table["Sharpe Ratio"] = display_table["Sharpe Ratio"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    print("\nQQQ Investment Strategy Analysis Results")
    print("=" * 80)
    print(display_table.to_string(index=False))
    print("=" * 80)
    print("\nNote: Annualized return is calculated using monthly IRR, not CAGR.")


def main() -> None:
    """Run the full analysis."""
    ticker = "QQQ"
    start_date = "2014-01-01"
    end_date = "2023-12-31"
    monthly_investment = 500.0

    dca_result = backtest_dca(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        monthly_investment=monthly_investment,
    )

    momentum_result = momentum_strategy(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        monthly_investment=monthly_investment,
    )

    results = [dca_result, momentum_result]
    results_table = build_results_table(results)

    print_results_table(results_table)
    results_table.to_csv("qqq_strategy_results.csv", index=False)

    plot_strategy_comparison(results)


if __name__ == "__main__":
    main()
