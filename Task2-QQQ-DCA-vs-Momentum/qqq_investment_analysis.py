"""
QQQ Investment Strategy Analysis (2014-01-01 ~ 2023-12-31)
===========================================================
Strategies: Dollar-Cost Averaging (DCA) vs Momentum Investing

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================
TICKER             = "QQQ"
START_DATE         = "2014-01-01"
END_DATE           = "2023-12-31"
MONTHLY_INVESTMENT = 500.0    # USD per month
RISK_FREE_RATE     = 0.02     # Annual risk-free rate for Sharpe ratio


# =============================================================================
# Step 1: Data Acquisition
# =============================================================================
def fetch_data(ticker, start_date, end_date):
    """
    Download historical price data for QQQ.

    Primary source: yfinance (with auto_adjust=True for dividend-adjusted prices)
    Fallback source: akshare (qfq adjusted)

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns
    -------
    pd.DataFrame
        Daily price data with 'Close' column
    """
    df = None

    # Try yfinance first
    try:
        import yfinance as yf
        print(f"[INFO] Attempting to download {ticker} from Yahoo Finance...")

        # auto_adjust=True returns dividend-adjusted prices
        raw_df = yf.download(ticker, start=start_date, end=end_date,
                            progress=False, auto_adjust=True)

        if not raw_df.empty and len(raw_df) > 100:
            # Handle MultiIndex columns (yfinance >= 0.2)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            df = raw_df[['Close']].copy()
            print(f"[INFO] Successfully downloaded {len(df)} rows from Yahoo Finance")
            print(f"[INFO] Using dividend-adjusted prices (auto_adjust=True)")
            return df
    except Exception as e:
        print(f"[WARN] Yahoo Finance failed: {e}")

    # Fallback to akshare
    try:
        import akshare as ak
        print(f"[INFO] Attempting to download {ticker} from akshare...")

        raw_df = ak.stock_us_daily(symbol=ticker, adjust='qfq')
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        raw_df.set_index('date', inplace=True)
        raw_df = raw_df.sort_index()
        raw_df = raw_df.loc[start_date:end_date]

        df = raw_df[['close']].rename(columns={'close': 'Close'})
        print(f"[INFO] Successfully downloaded {len(df)} rows from akshare")
        print(f"[INFO] Using qfq-adjusted prices")
        return df
    except Exception as e:
        print(f"[WARN] akshare failed: {e}")

    raise RuntimeError("Failed to download data from all sources")


# =============================================================================
# Step 2: Data Preprocessing
# =============================================================================
def preprocess_data(df, start_date, end_date):
    """
    Clean data and calculate additional indicators.

    Steps:
    1. Remove missing values
    2. Check for anomalies
    3. Calculate moving averages (SMA_50, SMA_200)
    4. Extract month-end prices

    Parameters
    ----------
    df : pd.DataFrame
        Raw price data
    start_date : str
        Start date for filtering
    end_date : str
        End date for filtering

    Returns
    -------
    tuple
        (daily_df, monthly_prices)
    """
    print("\n[INFO] Preprocessing data...")

    # Make a copy to avoid modifying original
    data = df.copy()

    # Set index name
    data.index.name = 'Date'

    # Filter to date range
    data = data.loc[start_date:end_date]

    # Step 1: Check and remove missing values
    missing_count = data['Close'].isnull().sum()
    if missing_count > 0:
        print(f"[INFO] Removing {missing_count} missing values")
        data.dropna(subset=['Close'], inplace=True)
    else:
        print(f"[INFO] No missing values found")

    # Step 2: Check for anomalies (e.g., zero or negative prices)
    anomalies = (data['Close'] <= 0).sum()
    if anomalies > 0:
        print(f"[WARN] Found {anomalies} anomalies (zero/negative prices)")
        data = data[data['Close'] > 0]

    # Step 3: Calculate additional indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Step 4: Extract month-end prices
    monthly = data['Close'].resample('ME').last()

    print(f"[INFO] Preprocessed: {len(data)} trading days, {len(monthly)} months")
    print(f"[INFO] Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"[INFO] Price range: ${data['Close'].iloc[0]:.2f} to ${data['Close'].iloc[-1]:.2f}")

    return data, monthly


# =============================================================================
# Step 3: Risk Metrics
# =============================================================================
def calculate_max_drawdown(values):
    """
    Calculate maximum drawdown.

    Max Drawdown = min((V - rolling_peak) / rolling_peak)
    """
    rolling_peak = values.cummax()
    drawdown = (values - rolling_peak) / rolling_peak
    return drawdown.min()


def calculate_volatility(returns):
    """
    Calculate annualized volatility from monthly returns.

    Annualized Volatility = std(monthly_returns) * sqrt(12)
    """
    return returns.std(ddof=1) * np.sqrt(12)


def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (mean_excess_return / std_excess_return) * sqrt(12)
    """
    excess_returns = returns - risk_free_rate / 12
    return (excess_returns.mean() / excess_returns.std(ddof=1)) * np.sqrt(12)


# =============================================================================
# Step 4: Return Calculation Methods
# =============================================================================
def calculate_cagr(ending_value, total_invested, n_years):
    """
    Calculate Compound Annual Growth Rate (CAGR).

    CAGR = (ending_value / total_invested)^(1/n_years) - 1

    Note: For DCA, this assumes all capital was invested at the start,
    which is NOT accurate for periodic investments.
    """
    return (ending_value / total_invested) ** (1 / n_years) - 1


def calculate_irr(portfolio_values, monthly_investment):
    """
    Calculate Internal Rate of Return (IRR) for periodic investments.

    IRR is the discount rate that makes NPV of cash flows equal to zero.
    This is the CORRECT method for DCA returns as it accounts for
    the timing of each investment.

    Cash flows: -$500 each month, +ending_value at the end
    """
    ending = portfolio_values.iloc[-1]
    n_periods = len(portfolio_values)

    # Build cash flows: negative for investments, positive at the end
    cash_flows = [-monthly_investment] * n_periods
    cash_flows[-1] += ending

    # Newton-Raphson iteration to find IRR
    rate = 0.01  # Initial guess: 1% monthly

    for _ in range(1000):
        # Calculate NPV and its derivative
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))

        if abs(dnpv) < 1e-10:
            break

        new_rate = rate - npv / dnpv

        # Prevent negative rates
        if new_rate < -0.99:
            new_rate = -0.5

        rate = new_rate

        if abs(npv) < 1e-8:
            break

    # Annualize monthly IRR
    annual_irr = (1 + rate) ** 12 - 1
    return annual_irr


# =============================================================================
# Step 5: Strategy Implementations
# =============================================================================
def simulate_dca(monthly_prices, monthly_investment):
    """
    Dollar-Cost Averaging (DCA) Strategy Simulation.

    Invest a fixed amount every month, never sell.

    Parameters
    ----------
    monthly_prices : pd.Series
        Month-end prices
    monthly_investment : float
        Amount to invest each month

    Returns
    -------
    pd.Series
        Portfolio value over time
    """
    shares = 0.0
    portfolio_values = []

    for price in monthly_prices:
        # Buy fractional shares
        shares += monthly_investment / price
        # Track portfolio value
        portfolio_values.append(shares * price)

    return pd.Series(portfolio_values, index=monthly_prices.index)


def simulate_momentum(monthly_prices, monthly_investment):
    """
    Momentum Investing Strategy Simulation.

    Invest when 6-month momentum (excluding most recent month) is positive.
    Otherwise, hold cash.

    Signal: sum of returns for months t-6 to t-2 (skip t-1 to avoid reversal)
    Position: 1 if signal > 0, else 0 (cash)

    Parameters
    ----------
    monthly_prices : pd.Series
        Month-end prices
    monthly_investment : float
        Amount to invest each month

    Returns
    -------
    pd.Series
        Portfolio value over time
    """
    # Calculate monthly returns
    returns = monthly_prices.pct_change()

    # Calculate 6-month momentum, skipping the most recent month
    # This avoids short-term reversal effects
    momentum = returns.rolling(window=6).apply(lambda x: x[:-1].sum(), raw=True)

    # Determine positions: 1 = invest in QQQ, 0 = hold cash
    positions = (momentum > 0).astype(int)

    # Simulate strategy
    shares = 0.0
    cash = 0.0
    portfolio_values = []

    for date, price in monthly_prices.items():
        if positions[date] == 1:
            # Positive momentum: invest in QQQ
            shares += monthly_investment / price
        else:
            # Negative momentum: hold cash
            cash += monthly_investment

        portfolio_values.append(shares * price + cash)

    return pd.Series(portfolio_values, index=monthly_prices.index)


# =============================================================================
# Step 6: Performance Metrics
# =============================================================================
def calculate_metrics(portfolio_values, monthly_investment=MONTHLY_INVESTMENT):
    """
    Calculate comprehensive performance metrics for a portfolio.

    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    n_months = len(portfolio_values)
    total_invested = monthly_investment * n_months
    ending_value = portfolio_values.iloc[-1]
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25

    # Calculate returns
    monthly_returns = portfolio_values.pct_change().dropna()

    return {
        'Total Invested ($)': total_invested,
        'Ending Value ($)': ending_value,
        'Return Multiplier (x)': ending_value / total_invested,
        'Total Return (%)': (ending_value - total_invested) / total_invested * 100,
        'Annualized Return (%) - CAGR': calculate_cagr(ending_value, total_invested, n_years) * 100,
        'Annualized Return (%) - IRR': calculate_irr(portfolio_values, monthly_investment) * 100,
        'Max Drawdown (%)': calculate_max_drawdown(portfolio_values) * 100,
        'Volatility Ann. (%)': calculate_volatility(monthly_returns) * 100,
        'Sharpe Ratio': calculate_sharpe_ratio(monthly_returns),
    }


# =============================================================================
# Step 7: Public API Functions (matching PDF specification)
# =============================================================================
def backtest_dca(ticker, start_date, end_date, monthly_investment):
    """
    DCA strategy backtest.

    Parameters
    ----------
    ticker : str
        Stock ticker
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    monthly_investment : float
        Monthly investment amount

    Returns
    -------
    tuple
        (annualized_return, total_return) in percentage
    """
    # Fetch and preprocess data
    raw_data = fetch_data(ticker, start_date, end_date)
    daily_data, monthly_prices = preprocess_data(raw_data, start_date, end_date)

    # Simulate DCA strategy
    portfolio = simulate_dca(monthly_prices, monthly_investment)

    # Calculate returns
    metrics = calculate_metrics(portfolio, monthly_investment)

    return metrics['Annualized Return (%) - IRR'], metrics['Total Return (%)']


def momentum_strategy(ticker, start_date, end_date, monthly_investment):
    """
    Momentum strategy backtest.

    Parameters
    ----------
    ticker : str
        Stock ticker
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    monthly_investment : float
        Monthly investment amount

    Returns
    -------
    tuple
        (annualized_return, total_return) in percentage
    """
    # Fetch and preprocess data
    raw_data = fetch_data(ticker, start_date, end_date)
    daily_data, monthly_prices = preprocess_data(raw_data, start_date, end_date)

    # Simulate momentum strategy
    portfolio = simulate_momentum(monthly_prices, monthly_investment)

    # Calculate returns
    metrics = calculate_metrics(portfolio, monthly_investment)

    return metrics['Annualized Return (%) - IRR'], metrics['Total Return (%)']


# =============================================================================
# Step 8: Visualization
# =============================================================================
def plot_results(dca_portfolio, momentum_portfolio, daily_data, save_path):
    """
    Generate visualization comparing both strategies.
    """
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("QQQ Investment Strategy Comparison (2014-2023)",
                 fontsize=14, fontweight="bold")

    # Plot 1: Portfolio Value
    ax = axes[0, 0]
    ax.plot(dca_portfolio.index, dca_portfolio,
            label="DCA", color="steelblue", linewidth=1.5)
    ax.plot(momentum_portfolio.index, momentum_portfolio,
            label="Momentum", color="darkorange", linewidth=1.5, linestyle="--")
    ax.set_title("Portfolio Value Over Time")
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()

    # Plot 2: Monthly Return Distribution
    ax = axes[0, 1]
    dca_returns = dca_portfolio.pct_change().dropna() * 100
    mom_returns = momentum_portfolio.pct_change().dropna() * 100
    ax.hist(dca_returns, bins=30, alpha=0.6, color="steelblue",
            edgecolor="white", label="DCA")
    ax.hist(mom_returns, bins=30, alpha=0.6, color="darkorange",
            edgecolor="white", label="Momentum")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Monthly Return Distribution")
    ax.set_xlabel("Return (%)")
    ax.legend()

    # Plot 3: Drawdown
    ax = axes[1, 0]
    dca_dd = (dca_portfolio - dca_portfolio.cummax()) / dca_portfolio.cummax() * 100
    mom_dd = (momentum_portfolio - momentum_portfolio.cummax()) / momentum_portfolio.cummax() * 100
    ax.fill_between(dca_portfolio.index, dca_dd, 0,
                    alpha=0.4, color="steelblue", label="DCA")
    ax.fill_between(momentum_portfolio.index, mom_dd, 0,
                    alpha=0.4, color="darkorange", label="Momentum")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()

    # Plot 4: QQQ Price and Moving Averages
    ax = axes[1, 1]
    ax.plot(daily_data.index, daily_data["Close"],
            color="black", linewidth=0.8, alpha=0.7, label="Adj Close")
    ax.plot(daily_data.index, daily_data["SMA_50"],
            color="royalblue", linewidth=1.2, linestyle="--", label="SMA 50")
    ax.plot(daily_data.index, daily_data["SMA_200"],
            color="crimson", linewidth=1.2, linestyle="-.", label="SMA 200")
    ax.set_title("QQQ Price with Moving Averages")
    ax.set_ylabel("USD")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Chart saved to: {save_path}")
    plt.show()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """
    Main function to run the complete analysis.
    """
    print("=" * 70)
    print("  QQQ Investment Strategy Analysis")
    print("=" * 70)
    print(f"  Ticker            : {TICKER}")
    print(f"  Period            : {START_DATE} to {END_DATE}")
    print(f"  Monthly Investment: ${MONTHLY_INVESTMENT:,.0f}")
    print(f"  Return Method     : IRR (Internal Rate of Return)")
    print("=" * 70)

    # Step 1: Fetch data
    raw_data = fetch_data(TICKER, START_DATE, END_DATE)

    # Step 2: Preprocess data
    daily_data, monthly_prices = preprocess_data(raw_data, START_DATE, END_DATE)

    # Step 3: Simulate strategies
    print("\n[INFO] Simulating DCA strategy...")
    dca_portfolio = simulate_dca(monthly_prices, MONTHLY_INVESTMENT)

    print("[INFO] Simulating Momentum strategy...")
    momentum_portfolio = simulate_momentum(monthly_prices, MONTHLY_INVESTMENT)

    # Step 4: Calculate metrics
    dca_metrics = calculate_metrics(dca_portfolio)
    momentum_metrics = calculate_metrics(momentum_portfolio)

    # Step 5: Print results
    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)

    print(f"\n{'Strategy':<15} {'Annual Return (IRR)':<20} {'Total Return':<15} {'Multiplier':<10}")
    print("-" * 60)
    print(f"{'DCA':<15} {dca_metrics['Annualized Return (%) - IRR']:>18.2f}% "
          f"{dca_metrics['Total Return (%)']:>13.2f}% "
          f"{dca_metrics['Return Multiplier (x)']:>9.2f}x")
    print(f"{'Momentum':<15} {momentum_metrics['Annualized Return (%) - IRR']:>18.2f}% "
          f"{momentum_metrics['Total Return (%)']:>13.2f}% "
          f"{momentum_metrics['Return Multiplier (x)']:>9.2f}x")

    print("\n" + "-" * 70)
    print("  Detailed Metrics")
    print("-" * 70)
    print(f"\n{'Metric':<35} {'DCA':>15} {'Momentum':>15}")
    print("-" * 67)

    for key in ['Total Invested ($)', 'Ending Value ($)', 'Return Multiplier (x)',
                'Total Return (%)', 'Annualized Return (%) - CAGR',
                'Annualized Return (%) - IRR', 'Max Drawdown (%)',
                'Volatility Ann. (%)', 'Sharpe Ratio']:
        dca_val = dca_metrics[key]
        mom_val = momentum_metrics[key]

        if '$' in key:
            fmt = ",.2f"
        elif 'Sharpe' in key:
            fmt = ".4f"
        else:
            fmt = ".2f"

        print(f"{key:<35} {format(dca_val, fmt):>15} {format(mom_val, fmt):>15}")

    print("-" * 67)

    # Step 6: Generate visualization
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "qqq_strategy_analysis.png")
    plot_results(dca_portfolio, momentum_portfolio, daily_data, chart_path)

    return dca_metrics, momentum_metrics


if __name__ == "__main__":
    main()
