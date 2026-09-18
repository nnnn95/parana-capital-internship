"""
Bonus: Sector Comparison Analysis
==================================
Compare FAANG (Technology) with Financial and Energy sectors.

This module extends the main analysis to compare performance across different sectors.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import main analysis module
from faang_technical_analysis import (
    DataFetcher, TechnicalIndicators, MARSIStrategy,
    EnhancedMARSIStrategy, BacktestEngine, PerformanceAnalyzer
)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')


class SectorComparison:
    """
    Compare FAANG stocks with other sectors.
    """

    # Sector ETFs for comparison
    SECTOR_ETFS = {
        'Technology': 'XLK',      # Technology Select Sector SPDR
        'Financial': 'XLF',       # Financial Select Sector SPDR
        'Energy': 'XLE',          # Energy Select Sector SPDR
        'Healthcare': 'XLV',      # Health Care Select Sector SPDR
        'Consumer': 'XLP',        # Consumer Staples Select Sector SPDR
    }

    # Representative stocks from each sector
    SECTOR_STOCKS = {
        'FAANG (Tech)': ['AAPL', 'AMZN', 'NFLX', 'META', 'GOOG'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    }

    def __init__(self, start_date='2014-01-01', end_date='2024-01-01'):
        """
        Initialize SectorComparison.

        Parameters:
        -----------
        start_date : str
            Start date for data fetching
        end_date : str
            End date for data fetching
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.close_prices = {}

    def fetch_sector_data(self):
        """
        Fetch data for all sector stocks.
        """
        print("Fetching sector data...")

        all_tickers = []
        for sector, tickers in self.SECTOR_STOCKS.items():
            all_tickers.extend(tickers)

        # Add sector ETFs
        all_tickers.extend(list(self.SECTOR_ETFS.values()))

        # Remove duplicates
        all_tickers = list(set(all_tickers))

        fetcher = DataFetcher(start_date=self.start_date, end_date=self.end_date)

        for ticker in all_tickers:
            try:
                print(f"  Fetching {ticker}...")
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    self.data[ticker] = df
                    if 'Adj Close' in df.columns:
                        self.close_prices[ticker] = df['Adj Close']
                    elif 'Close' in df.columns:
                        self.close_prices[ticker] = df['Close']
            except Exception as e:
                print(f"    Error fetching {ticker}: {e}")

        print(f"Fetched data for {len(self.data)} tickers")
        return self.data

    def run_sector_backtest(self, strategy=None):
        """
        Run backtest for all sectors using MA-RSI strategy.

        Parameters:
        -----------
        strategy : MARSIStrategy, optional
            Trading strategy to use. Default is MA50-RSI14

        Returns:
        --------
        dict
            Dictionary with sector as key and results as value
        """
        if strategy is None:
            strategy = MARSIStrategy(ma_period=50, rsi_period=14)

        if not self.close_prices:
            self.fetch_sector_data()

        sector_results = {}
        backtest_engine = BacktestEngine()

        for sector, tickers in self.SECTOR_STOCKS.items():
            print(f"\nBacktesting {sector} sector...")
            sector_results[sector] = {}

            for ticker in tickers:
                if ticker not in self.close_prices:
                    print(f"  Warning: {ticker} data not available")
                    continue

                prices = self.close_prices[ticker].dropna()
                signals = strategy.generate_signals(prices)
                result = backtest_engine.run_backtest(prices, signals)
                sector_results[sector][ticker] = result

                metrics = result['metrics']
                print(f"  {ticker}: Return={metrics['total_return']*100:.2f}%, "
                      f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                      f"MaxDD={metrics['max_drawdown']*100:.2f}%")

        return sector_results

    def calculate_sector_portfolio(self, sector_results):
        """
        Calculate equal-weighted portfolio returns for each sector.

        Parameters:
        -----------
        sector_results : dict
            Results from run_sector_backtest

        Returns:
        --------
        pd.DataFrame
            DataFrame with sector portfolio cumulative returns
        """
        portfolio_returns = pd.DataFrame()

        for sector, ticker_results in sector_results.items():
            # Collect all strategy returns for this sector
            returns_list = []
            common_index = None

            for ticker, result in ticker_results.items():
                returns = result['results']['Strategy_Returns']
                returns_list.append(returns)

            if returns_list:
                # Equal weight portfolio
                combined_returns = pd.concat(returns_list, axis=1).mean(axis=1)
                cumulative = (1 + combined_returns).cumprod()
                portfolio_returns[sector] = cumulative

        return portfolio_returns

    def plot_sector_comparison(self, sector_results, benchmark_data=None):
        """
        Plot sector comparison chart.

        Parameters:
        -----------
        sector_results : dict
            Results from run_sector_backtest
        benchmark_data : pd.Series, optional
            Benchmark cumulative returns
        """
        portfolio_returns = self.calculate_sector_portfolio(sector_results)

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = {'FAANG (Tech)': 'blue', 'Financial': 'green', 'Energy': 'orange'}

        for sector in portfolio_returns.columns:
            ax.plot(portfolio_returns.index, portfolio_returns[sector],
                   label=sector, linewidth=2, color=colors.get(sector, 'gray'), alpha=0.8)

        # Add benchmark
        if benchmark_data is not None:
            ax.plot(benchmark_data.index, benchmark_data,
                   label='SPY (Benchmark)', linewidth=2, color='black',
                   linestyle='--', alpha=0.7)

        ax.set_title('Sector Comparison: FAANG vs Financial vs Energy',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_sector_summary(self, sector_results):
        """
        Generate summary table for all sectors.

        Parameters:
        -----------
        sector_results : dict
            Results from run_sector_backtest

        Returns:
        --------
        pd.DataFrame
            Summary table with sector averages
        """
        summary_data = []

        for sector, ticker_results in sector_results.items():
            total_returns = []
            sharpe_ratios = []
            max_drawdowns = []

            for ticker, result in ticker_results.items():
                metrics = result['metrics']
                total_returns.append(metrics['total_return'])
                sharpe_ratios.append(metrics['sharpe_ratio'])
                max_drawdowns.append(metrics['max_drawdown'])

            summary_data.append({
                'Sector': sector,
                'Avg Return': f"{np.mean(total_returns)*100:.2f}%",
                'Avg Sharpe': f"{np.mean(sharpe_ratios):.2f}",
                'Avg MaxDD': f"{np.mean(max_drawdowns)*100:.2f}%",
                'Best Stock Return': f"{max(total_returns)*100:.2f}%",
                'Worst Stock Return': f"{min(total_returns)*100:.2f}%"
            })

        return pd.DataFrame(summary_data)


def plot_individual_signals(prices, signals, ticker, save_path=None):
    """
    Plot price chart with MA, RSI, and buy/sell signals.

    Parameters:
    -----------
    prices : pd.Series
        Price series
    signals : pd.DataFrame
        DataFrame with signals
    ticker : str
        Stock ticker
    save_path : str, optional
        Path to save the figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                          gridspec_kw={'height_ratios': [3, 1, 1]})

    # Price and MA
    ax1.plot(prices.index, prices, label='Price', color='blue', linewidth=1)
    ax1.plot(signals.index, signals['MA'], label='MA', color='orange', linewidth=1.5)

    # Buy signals
    buy_signals = signals[signals['Signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Price'],
               marker='^', color='green', s=100, label='Buy Signal', zorder=5)

    # Sell signals
    sell_signals = signals[signals['Signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['Price'],
               marker='v', color='red', s=100, label='Sell Signal', zorder=5)

    ax1.set_title(f'{ticker} - Price and Trading Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2.plot(signals.index, signals['RSI'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.fill_between(signals.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_title('Relative Strength Index (RSI)', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Cumulative returns
    cumulative_returns = (1 + prices.pct_change()).cumprod()
    ax3.plot(cumulative_returns.index, cumulative_returns,
            label='Buy & Hold', color='gray', linewidth=1.5)

    # Strategy returns would be calculated separately
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Cumulative Returns', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_drawdowns(results_dict, benchmark_returns=None, save_path=None):
    """
    Plot drawdown charts for all stocks in a single figure.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with ticker as key and backtest results as value
    benchmark_returns : pd.Series, optional
        Benchmark cumulative returns
    save_path : str, optional
        Path to save the figure
    """
    n_stocks = len(results_dict)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(14, 3 * n_stocks))

    if n_stocks == 1:
        axes = [axes]

    for idx, (ticker, backtest_result) in enumerate(results_dict.items()):
        results = backtest_result['results']
        metrics = backtest_result['metrics']

        # Calculate drawdown
        cumulative = results['Strategy_Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        ax = axes[idx]
        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)

        max_dd = metrics['max_drawdown'] * 100
        ax.axhline(y=max_dd, color='black', linestyle='--',
                  label=f'Max DD: {max_dd:.2f}%')

        ax.set_title(f'{ticker} - Drawdown', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=10)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.xlabel('Date', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_heatmap(comparison_results, metric='sharpe_ratio', save_path=None):
    """
    Create heatmap comparing strategy performance across stocks.

    Parameters:
    -----------
    comparison_results : dict
        Dictionary with strategy name as key and results dict as value
    metric : str
        Metric to compare ('sharpe_ratio', 'annualized_return', 'max_drawdown')
    save_path : str, optional
        Path to save the figure
    """
    # Extract metrics for each strategy and ticker
    data = {}

    for strategy_name, ticker_results in comparison_results.items():
        data[strategy_name] = {}
        for ticker, result in ticker_results.items():
            data[strategy_name][ticker] = result['metrics'][metric]

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    if metric == 'sharpe_ratio':
        cmap = 'RdYlGn'
        fmt = '.2f'
        title = 'Sharpe Ratio Comparison'
    elif metric == 'annualized_return':
        cmap = 'RdYlGn'
        fmt = '.2%'
        title = 'Annualized Return Comparison'
    else:
        cmap = 'RdYlGn_r'
        fmt = '.2%'
        title = 'Max Drawdown Comparison'

    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, center=0, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def run_bonus_analysis():
    """
    Run the bonus analysis: sector comparison and enhanced visualizations.
    """
    print("=" * 60)
    print("Bonus Analysis: Sector Comparison")
    print("=" * 60)
    print()

    # Initialize sector comparison
    sector_comp = SectorComparison(start_date='2014-01-01', end_date='2024-01-01')

    # Fetch data and run backtest
    sector_results = sector_comp.run_sector_backtest()

    # Generate summary
    print("\nSector Performance Summary:")
    print("-" * 80)
    summary = sector_comp.get_sector_summary(sector_results)
    print(summary.to_string(index=False))

    # Get SPY benchmark
    spy_data = yf.download('SPY', start='2014-01-01', end='2024-01-01', progress=False)
    spy_returns = spy_data['Adj Close'].pct_change()
    spy_cumulative = (1 + spy_returns).cumprod()

    # Plot sector comparison
    fig1 = sector_comp.plot_sector_comparison(sector_results, spy_cumulative)
    fig1.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/sector_comparison.png',
                 dpi=150, bbox_inches='tight')
    print("\nSaved: sector_comparison.png")

    # Plot individual stock signals for AAPL
    print("\nGenerating individual stock analysis...")
    fetcher = DataFetcher()
    aapl_prices = sector_comp.close_prices['AAPL']

    strategy = MARSIStrategy(ma_period=50, rsi_period=14)
    signals = strategy.generate_signals(aapl_prices)

    fig2 = plot_individual_signals(
        aapl_prices, signals, 'AAPL',
        save_path='/Users/wangnuo/Desktop/Technical Analysis for FAANG/aapl_signals.png'
    )
    print("Saved: aapl_signals.png")

    plt.close('all')

    print("\n" + "=" * 60)
    print("Bonus Analysis Complete!")
    print("=" * 60)

    return sector_results


if __name__ == "__main__":
    results = run_bonus_analysis()
