"""
Complete Technical Analysis for FAANG Stocks
=============================================
Main script that runs all analyses and generates all required outputs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Technical Indicators Class
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators for trading strategy."""

    @staticmethod
    def calculate_sma(prices, window):
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(prices, window):
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        middle_band = TechnicalIndicators.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band


# ============================================================================
# MA-RSI Strategy Class
# ============================================================================

class MARSIStrategy:
    """MA-RSI Trading Strategy Implementation."""

    def __init__(self, ma_period=50, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70):
        """Initialize MA-RSI Strategy with parameters."""
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, prices):
        """Generate buy/sell signals based on MA-RSI strategy."""
        ma = TechnicalIndicators.calculate_sma(prices, self.ma_period)
        rsi = TechnicalIndicators.calculate_rsi(prices, self.rsi_period)

        signals = pd.DataFrame(index=prices.index)
        signals['Price'] = prices
        signals['MA'] = ma
        signals['RSI'] = rsi
        signals['Signal'] = 0
        signals['Position'] = 0

        rsi_prev = rsi.shift(1)

        # Buy Signal: Price above MA AND RSI crosses above oversold level
        buy_condition = (
            (prices > ma) &
            (rsi > self.rsi_oversold) &
            (rsi_prev <= self.rsi_oversold)
        )

        # Sell Signal: Price below MA AND RSI crosses below overbought level
        sell_condition = (
            (prices < ma) &
            (rsi < self.rsi_overbought) &
            (rsi_prev >= self.rsi_overbought)
        )

        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1
        signals['Position'] = signals['Signal'].replace(0, np.nan).ffill().fillna(0)

        return signals


# ============================================================================
# Backtest Engine Class
# ============================================================================

class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(self, initial_capital=100000, commission=0.001):
        """Initialize BacktestEngine."""
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(self, prices, signals, signal_column='Signal'):
        """Run backtest with given prices and signals."""
        results = pd.DataFrame(index=prices.index)
        results['Price'] = prices
        results['Signal'] = signals[signal_column]
        results['Returns'] = prices.pct_change()
        results['Position'] = signals['Position']
        results['Strategy_Returns'] = results['Position'].shift(1) * results['Returns']

        # Apply commission costs
        trades = results['Signal'].abs()
        commission_costs = trades * self.commission
        results['Strategy_Returns'] = results['Strategy_Returns'] - commission_costs

        # Calculate cumulative returns
        results['Cumulative_Returns'] = (1 + results['Returns']).cumprod()
        results['Strategy_Cumulative_Returns'] = (1 + results['Strategy_Returns'].fillna(0)).cumprod()
        results['Portfolio_Value'] = self.initial_capital * results['Strategy_Cumulative_Returns']

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        return {'results': results, 'metrics': metrics}

    def _calculate_metrics(self, results):
        """Calculate performance metrics."""
        strategy_returns = results['Strategy_Returns'].dropna()

        total_return = results['Strategy_Cumulative_Returns'].iloc[-1] - 1
        years = len(results) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1

        risk_free_rate = 0.02
        excess_returns = strategy_returns.mean() * 252 - risk_free_rate
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0

        cumulative = results['Strategy_Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        max_dd_date = drawdown.idxmin()
        peak_date = running_max.loc[:max_dd_date].idxmax()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'max_dd_date': max_dd_date,
            'peak_date': peak_date,
            'num_trades': results['Signal'].abs().sum()
        }


# ============================================================================
# Main Analysis Function
# ============================================================================

def fetch_data_with_retry(tickers, start_date, end_date, max_retries=3):
    """Fetch data with retry logic for network issues."""
    data = {}
    close_prices = pd.DataFrame()

    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                print(f"  Fetching {ticker}... (attempt {attempt + 1})")
                df = yf.download(ticker, start=start_date, end=end_date,
                                progress=False, auto_adjust=True)
                if not df.empty:
                    data[ticker] = df
                    if 'Close' in df.columns:
                        close_prices[ticker] = df['Close']
                    print(f"    Success: {len(df)} records")
                break
            except Exception as e:
                print(f"    Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"    Failed to fetch {ticker}")

    return data, close_prices


def main():
    """Main function to run the complete FAANG technical analysis."""
    print("=" * 70)
    print("FAANG Technical Analysis - MA-RSI Strategy")
    print("Project: Technical Analysis for FAANG Stocks")
    print("=" * 70)
    print()

    # Configuration
    FAANG_TICKERS = ['AAPL', 'AMZN', 'NFLX', 'META', 'GOOG']
    BENCHMARK = 'SPY'
    START_DATE = '2014-01-01'
    END_DATE = '2024-01-01'

    # Step 1: Fetch Data
    print("=" * 70)
    print("Step 1: Fetching Historical Data (2014-2024)")
    print("=" * 70)

    all_tickers = FAANG_TICKERS + [BENCHMARK]
    data, close_prices = fetch_data_with_retry(all_tickers, START_DATE, END_DATE)

    if close_prices.empty:
        print("ERROR: No data fetched. Please check your internet connection.")
        return None

    print(f"\nSuccessfully fetched data for {len(close_prices.columns)} tickers")
    print(f"Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print()

    # Step 2: Calculate Benchmark Returns
    print("=" * 70)
    print("Step 2: Preparing Benchmark (SPY) Data")
    print("=" * 70)

    if BENCHMARK in close_prices.columns:
        spy_returns = close_prices[BENCHMARK].pct_change().dropna()
        spy_cumulative = (1 + spy_returns).cumprod()
        print(f"SPY benchmark prepared: {len(spy_cumulative)} data points")
    else:
        spy_cumulative = None
        print("Warning: SPY benchmark not available")
    print()

    # Step 3: Define Strategy Configurations
    print("=" * 70)
    print("Step 3: MA-RSI Strategy Configurations")
    print("=" * 70)

    strategy_configs = [
        {'name': 'MA50-RSI14', 'ma_period': 50, 'rsi_period': 14},
        {'name': 'MA10-RSI9', 'ma_period': 10, 'rsi_period': 9},
        {'name': 'MA10-RSI25', 'ma_period': 10, 'rsi_period': 25},
    ]

    print("Testing the following configurations:")
    for config in strategy_configs:
        print(f"  - {config['name']}: MA({config['ma_period']}) + RSI({config['rsi_period']})")
    print()

    # Step 4: Run Backtests for FAANG Stocks
    print("=" * 70)
    print("Step 4: Running Backtests for FAANG Stocks")
    print("=" * 70)

    all_results = {}
    backtest_engine = BacktestEngine()

    # Use MA50-RSI14 as primary strategy
    primary_strategy = MARSIStrategy(ma_period=50, rsi_period=14)

    for ticker in FAANG_TICKERS:
        if ticker not in close_prices.columns:
            print(f"  Warning: {ticker} data not available, skipping...")
            continue

        print(f"\n  Backtesting {ticker}...")
        prices = close_prices[ticker].dropna()
        signals = primary_strategy.generate_signals(prices)
        backtest_result = backtest_engine.run_backtest(prices, signals)
        all_results[ticker] = backtest_result

        metrics = backtest_result['metrics']
        print(f"    Total Return: {metrics['total_return']*100:.2f}%")
        print(f"    Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"    Number of Trades: {int(metrics['num_trades'])}")

    print()

    # Step 5: Performance Summary Table
    print("=" * 70)
    print("Step 5: Performance Summary (MA50-RSI14 Strategy)")
    print("=" * 70)

    summary_data = []
    for ticker, result in all_results.items():
        metrics = result['metrics']
        summary_data.append({
            'Ticker': ticker,
            'Total Return': f"{metrics['total_return']*100:.2f}%",
            'Annualized Return': f"{metrics['annualized_return']*100:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
            'Num Trades': int(metrics['num_trades'])
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()

    # Step 6: Compare Different Strategy Configurations
    print("=" * 70)
    print("Step 6: Strategy Configuration Comparison (AAPL)")
    print("=" * 70)

    if 'AAPL' in close_prices.columns:
        aapl_prices = close_prices['AAPL'].dropna()
        comparison_results = {}

        for config in strategy_configs:
            strategy = MARSIStrategy(
                ma_period=config['ma_period'],
                rsi_period=config['rsi_period']
            )
            signals = strategy.generate_signals(aapl_prices)
            result = backtest_engine.run_backtest(aapl_prices, signals)
            comparison_results[config['name']] = result

        comparison_data = []
        for name, result in comparison_results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{metrics['total_return']*100:.2f}%",
                'Annualized Return': f"{metrics['annualized_return']*100:.2f}%",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
                'Num Trades': int(metrics['num_trades'])
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    print()

    # Step 7: Generate Visualizations
    print("=" * 70)
    print("Step 7: Generating Visualizations")
    print("=" * 70)

    output_dir = '/Users/wangnuo/Desktop/Technical Analysis for FAANG'

    # Figure 1: Cumulative Returns for All FAANG Stocks
    print("  Creating Figure 1: Cumulative Returns Comparison...")
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for idx, (ticker, result) in enumerate(all_results.items()):
        results = result['results']
        ax1.plot(results.index, results['Strategy_Cumulative_Returns'],
                label=ticker, linewidth=2, color=colors[idx % len(colors)], alpha=0.8)

    if spy_cumulative is not None:
        ax1.plot(spy_cumulative.index, spy_cumulative,
                label='SPY (Benchmark)', linewidth=2.5, color='black',
                linestyle='--', alpha=0.7)

    ax1.set_title('FAANG Stocks - MA50-RSI14 Strategy Cumulative Returns',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Returns', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(f'{output_dir}/fig1_cumulative_returns.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: fig1_cumulative_returns.png")
    plt.close()

    # Figure 2: Drawdown Analysis
    print("  Creating Figure 2: Drawdown Analysis...")
    n_stocks = len(all_results)
    fig2, axes = plt.subplots(n_stocks, 1, figsize=(14, 3 * n_stocks))

    if n_stocks == 1:
        axes = [axes]

    for idx, (ticker, backtest_result) in enumerate(all_results.items()):
        results = backtest_result['results']
        metrics = backtest_result['metrics']

        cumulative = results['Strategy_Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        ax = axes[idx]
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)

        max_dd = metrics['max_drawdown'] * 100
        ax.axhline(y=max_dd, color='black', linestyle='--',
                  label=f'Max DD: {max_dd:.2f}%')

        ax.set_title(f'{ticker} - Drawdown Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()
    fig2.savefig(f'{output_dir}/fig2_drawdown_analysis.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: fig2_drawdown_analysis.png")
    plt.close()

    # Figure 3: Portfolio Value vs Benchmark
    print("  Creating Figure 3: Portfolio Value vs SPY Benchmark...")
    fig3, ax3 = plt.subplots(figsize=(14, 8))

    # Calculate equal-weighted portfolio
    portfolio_value = pd.DataFrame()
    for ticker, result in all_results.items():
        portfolio_value[ticker] = result['results']['Portfolio_Value']

    portfolio_value['Total'] = portfolio_value.mean(axis=1)

    ax3.plot(portfolio_value.index, portfolio_value['Total'],
            label='Equal-Weighted Portfolio', linewidth=2.5, color='blue')

    if spy_cumulative is not None:
        benchmark_value = 100000 * spy_cumulative
        ax3.plot(benchmark_value.index, benchmark_value,
                label='SPY (Benchmark)', linewidth=2.5, color='gray',
                linestyle='--')

    ax3.set_title('Portfolio Total Value vs SPY Benchmark',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    fig3.savefig(f'{output_dir}/fig3_portfolio_vs_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: fig3_portfolio_vs_benchmark.png")
    plt.close()

    # Figure 4: Strategy Configuration Comparison for AAPL
    if 'AAPL' in comparison_results:
        print("  Creating Figure 4: Strategy Configuration Comparison...")
        fig4, ax4 = plt.subplots(figsize=(14, 8))

        colors = ['blue', 'green', 'red']
        for idx, (name, result) in enumerate(comparison_results.items()):
            results = result['results']
            ax4.plot(results.index, results['Strategy_Cumulative_Returns'],
                    label=name, linewidth=2, color=colors[idx], alpha=0.8)

        if spy_cumulative is not None:
            ax4.plot(spy_cumulative.index, spy_cumulative,
                    label='SPY (Benchmark)', linewidth=2.5, color='black',
                    linestyle='--', alpha=0.7)

        ax4.set_title('AAPL - MA-RSI Strategy Configuration Comparison',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Cumulative Returns', fontsize=12)
        ax4.legend(loc='upper left', fontsize=11)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        fig4.savefig(f'{output_dir}/fig4_strategy_comparison.png', dpi=150, bbox_inches='tight')
        print(f"    Saved: fig4_strategy_comparison.png")
        plt.close()

    # Figure 5: Price with Signals (AAPL Example)
    if 'AAPL' in all_results:
        print("  Creating Figure 5: AAPL Price with Trading Signals...")
        aapl_result = all_results['AAPL']
        results = aapl_result['results']
        signals = primary_strategy.generate_signals(close_prices['AAPL'].dropna())

        fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 10),
                                           gridspec_kw={'height_ratios': [2, 1]})

        # Price and MA
        ax5a.plot(results.index, results['Price'], label='Price', color='blue', linewidth=1)
        ax5a.plot(signals.index, signals['MA'], label='MA(50)', color='orange', linewidth=1.5)

        # Buy signals
        buy_signals = signals[signals['Signal'] == 1]
        ax5a.scatter(buy_signals.index, buy_signals['Price'],
                    marker='^', color='green', s=100, label='Buy Signal', zorder=5)

        # Sell signals
        sell_signals = signals[signals['Signal'] == -1]
        ax5a.scatter(sell_signals.index, sell_signals['Price'],
                    marker='v', color='red', s=100, label='Sell Signal', zorder=5)

        ax5a.set_title('AAPL - Price and Trading Signals (MA50-RSI14)', fontsize=14, fontweight='bold')
        ax5a.set_ylabel('Price ($)', fontsize=12)
        ax5a.legend(loc='upper left', fontsize=10)
        ax5a.grid(True, alpha=0.3)

        # RSI
        ax5b.plot(signals.index, signals['RSI'], label='RSI(14)', color='purple', linewidth=1)
        ax5b.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax5b.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax5b.fill_between(signals.index, 30, 70, alpha=0.1, color='gray')
        ax5b.set_title('Relative Strength Index (RSI)', fontsize=12)
        ax5b.set_xlabel('Date', fontsize=12)
        ax5b.set_ylabel('RSI', fontsize=12)
        ax5b.legend(loc='upper left', fontsize=10)
        ax5b.set_ylim(0, 100)
        ax5b.grid(True, alpha=0.3)

        plt.tight_layout()
        fig5.savefig(f'{output_dir}/fig5_aapl_signals.png', dpi=150, bbox_inches='tight')
        print(f"    Saved: fig5_aapl_signals.png")
        plt.close()

    # Figure 6: Performance Heatmap
    print("  Creating Figure 6: Performance Metrics Heatmap...")
    fig6, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sharpe Ratio Heatmap
    sharpe_data = {ticker: [result['metrics']['sharpe_ratio']]
                   for ticker, result in all_results.items()}
    sharpe_df = pd.DataFrame(sharpe_data, index=['Sharpe Ratio']).T
    sns.heatmap(sharpe_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[0])
    axes[0].set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    # Annualized Return Heatmap
    return_data = {ticker: [result['metrics']['annualized_return']*100]
                   for ticker, result in all_results.items()}
    return_df = pd.DataFrame(return_data, index=['Ann. Return %']).T
    sns.heatmap(return_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[1])
    axes[1].set_title('Annualized Return (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    # Max Drawdown Heatmap
    dd_data = {ticker: [result['metrics']['max_drawdown']*100]
               for ticker, result in all_results.items()}
    dd_df = pd.DataFrame(dd_data, index=['Max DD %']).T
    sns.heatmap(dd_df, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0, ax=axes[2])
    axes[2].set_title('Max Drawdown (%)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    plt.tight_layout()
    fig6.savefig(f'{output_dir}/fig6_performance_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: fig6_performance_heatmap.png")
    plt.close()

    print()

    # Step 8: Bonus - MACD and Bollinger Bands Analysis
    print("=" * 70)
    print("Step 8: Bonus - Enhanced Strategy Analysis")
    print("=" * 70)
    print("Calculating MACD and Bollinger Bands for additional analysis...")

    if 'AAPL' in close_prices.columns:
        aapl_prices = close_prices['AAPL'].dropna()

        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(aapl_prices)

        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(aapl_prices)

        # Create bonus visualization
        fig7, (ax7a, ax7b, ax7c) = plt.subplots(3, 1, figsize=(14, 12),
                                                  gridspec_kw={'height_ratios': [2, 1, 1]})

        # Price with Bollinger Bands
        ax7a.plot(aapl_prices.index, aapl_prices, label='Price', color='blue', linewidth=1)
        ax7a.plot(aapl_prices.index, upper, label='Upper BB', color='gray', linewidth=1, alpha=0.7)
        ax7a.plot(aapl_prices.index, middle, label='Middle BB', color='orange', linewidth=1, alpha=0.7)
        ax7a.plot(aapl_prices.index, lower, label='Lower BB', color='gray', linewidth=1, alpha=0.7)
        ax7a.fill_between(aapl_prices.index, upper, lower, alpha=0.1, color='gray')
        ax7a.set_title('AAPL - Price with Bollinger Bands', fontsize=14, fontweight='bold')
        ax7a.set_ylabel('Price ($)', fontsize=12)
        ax7a.legend(loc='upper left', fontsize=10)
        ax7a.grid(True, alpha=0.3)

        # MACD
        ax7b.plot(aapl_prices.index, macd_line, label='MACD', color='blue', linewidth=1)
        ax7b.plot(aapl_prices.index, signal_line, label='Signal', color='orange', linewidth=1)
        ax7b.bar(aapl_prices.index, histogram, label='Histogram',
                color=['green' if h >= 0 else 'red' for h in histogram], alpha=0.5)
        ax7b.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax7b.set_title('MACD Indicator', fontsize=12)
        ax7b.set_ylabel('MACD', fontsize=12)
        ax7b.legend(loc='upper left', fontsize=10)
        ax7b.grid(True, alpha=0.3)

        # RSI
        rsi = TechnicalIndicators.calculate_rsi(aapl_prices, 14)
        ax7c.plot(aapl_prices.index, rsi, label='RSI(14)', color='purple', linewidth=1)
        ax7c.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax7c.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax7c.fill_between(aapl_prices.index, 30, 70, alpha=0.1, color='gray')
        ax7c.set_title('RSI Indicator', fontsize=12)
        ax7c.set_xlabel('Date', fontsize=12)
        ax7c.set_ylabel('RSI', fontsize=12)
        ax7c.legend(loc='upper left', fontsize=10)
        ax7c.set_ylim(0, 100)
        ax7c.grid(True, alpha=0.3)

        plt.tight_layout()
        fig7.savefig(f'{output_dir}/fig7_bonus_indicators.png', dpi=150, bbox_inches='tight')
        print(f"    Saved: fig7_bonus_indicators.png")
        plt.close()

    print()

    # Final Summary
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Generated Files:")
    print("  1. fig1_cumulative_returns.png - FAANG cumulative returns vs SPY")
    print("  2. fig2_drawdown_analysis.png - Drawdown analysis for each stock")
    print("  3. fig3_portfolio_vs_benchmark.png - Portfolio value vs benchmark")
    print("  4. fig4_strategy_comparison.png - MA-RSI configuration comparison")
    print("  5. fig5_aapl_signals.png - AAPL price with trading signals")
    print("  6. fig6_performance_heatmap.png - Performance metrics heatmap")
    print("  7. fig7_bonus_indicators.png - MACD and Bollinger Bands analysis")
    print()

    return {
        'close_prices': close_prices,
        'all_results': all_results,
        'spy_cumulative': spy_cumulative,
        'comparison_results': comparison_results if 'comparison_results' in dir() else None
    }


if __name__ == "__main__":
    results = main()
