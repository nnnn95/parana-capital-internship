"""
Technical Analysis for FAANG Stocks
====================================
This module implements MA-RSI technical analysis strategy for FAANG stocks
(AAPL, AMZN, NFLX, META, GOOG) and compares performance with SP500 benchmark.

Author: Intern Project
Date: 2024
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataFetcher:
    """
    Fetch historical stock data for FAANG stocks and SP500.
    """

    # FAANG stock tickers
    FAANG_TICKERS = ['AAPL', 'AMZN', 'NFLX', 'META', 'GOOG']
    BENCHMARK = 'SPY'  # SP500 ETF as benchmark

    def __init__(self, start_date='2014-01-01', end_date='2024-01-01'):
        """
        Initialize DataFetcher with date range.

        Parameters:
        -----------
        start_date : str
            Start date for data fetching (format: YYYY-MM-DD)
        end_date : str
            End date for data fetching (format: YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self, tickers=None):
        """
        Fetch historical daily data for given tickers.

        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols. If None, fetches FAANG + SPY

        Returns:
        --------
        dict
            Dictionary with ticker as key and DataFrame as value
        """
        if tickers is None:
            tickers = self.FAANG_TICKERS + [self.BENCHMARK]

        data = {}
        print("Fetching stock data...")

        for ticker in tickers:
            try:
                print(f"  Fetching {ticker}...")
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    data[ticker] = df
                    print(f"    {ticker}: {len(df)} records fetched")
            except Exception as e:
                print(f"    Error fetching {ticker}: {e}")

        return data

    def get_close_prices(self, data):
        """
        Extract adjusted close prices from fetched data.

        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames from fetch_data()

        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices for all tickers
        """
        close_prices = pd.DataFrame()

        for ticker, df in data.items():
            if 'Adj Close' in df.columns:
                close_prices[ticker] = df['Adj Close']
            elif 'Close' in df.columns:
                close_prices[ticker] = df['Close']

        return close_prices


class TechnicalIndicators:
    """
    Calculate technical indicators for trading strategy.
    """

    @staticmethod
    def calculate_sma(prices, window):
        """
        Calculate Simple Moving Average.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Moving average window period

        Returns:
        --------
        pd.Series
            Simple Moving Average series
        """
        return prices.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(prices, window):
        """
        Calculate Exponential Moving Average.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Moving average window period

        Returns:
        --------
        pd.Series
            Exponential Moving Average series
        """
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(prices, window=14):
        """
        Calculate Relative Strength Index (RSI).

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            RSI calculation period (default: 14)

        Returns:
        --------
        pd.Series
            RSI series (0-100 scale)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Parameters:
        -----------
        prices : pd.Series
            Price series
        fast : int
            Fast EMA period (default: 12)
        slow : int
            Slow EMA period (default: 26)
        signal : int
            Signal line period (default: 9)

        Returns:
        --------
        tuple
            (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Moving average window period (default: 20)
        num_std : int
            Number of standard deviations (default: 2)

        Returns:
        --------
        tuple
            (Upper Band, Middle Band, Lower Band)
        """
        middle_band = TechnicalIndicators.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return upper_band, middle_band, lower_band


class MARSIStrategy:
    """
    MA-RSI Trading Strategy Implementation.

    Strategy Logic:
    - Buy Signal: Price above MA AND RSI crosses above 30 (from oversold)
    - Sell Signal: Price below MA AND RSI crosses below 70 (from overbought)
    """

    def __init__(self, ma_period=50, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70):
        """
        Initialize MA-RSI Strategy with parameters.

        Parameters:
        -----------
        ma_period : int
            Moving average period (default: 50)
        rsi_period : int
            RSI calculation period (default: 14)
        rsi_oversold : int
            RSI oversold threshold (default: 30)
        rsi_overbought : int
            RSI overbought threshold (default: 70)
        """
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, prices):
        """
        Generate buy/sell signals based on MA-RSI strategy.

        Parameters:
        -----------
        prices : pd.Series
            Price series

        Returns:
        --------
        pd.DataFrame
            DataFrame with prices, MA, RSI, and signals
        """
        # Calculate indicators
        ma = TechnicalIndicators.calculate_sma(prices, self.ma_period)
        rsi = TechnicalIndicators.calculate_rsi(prices, self.rsi_period)

        # Create signals DataFrame
        signals = pd.DataFrame(index=prices.index)
        signals['Price'] = prices
        signals['MA'] = ma
        signals['RSI'] = rsi

        # Initialize signal column (1: buy, -1: sell, 0: hold)
        signals['Signal'] = 0
        signals['Position'] = 0

        # Previous RSI values for crossover detection
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

        # Calculate positions (cumulative signal)
        signals['Position'] = signals['Signal'].replace(0, np.nan).ffill().fillna(0)

        return signals

    def __str__(self):
        return f"MA-RSI Strategy (MA={self.ma_period}, RSI={self.rsi_period})"


class EnhancedMARSIStrategy(MARSIStrategy):
    """
    Enhanced MA-RSI Strategy with MACD and Bollinger Bands.

    Additional confirmation signals:
    - MACD crossover for trend confirmation
    - Bollinger Band position for volatility assessment
    """

    def __init__(self, ma_period=50, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70,
                 use_macd=True, use_bollinger=True):
        """
        Initialize Enhanced MA-RSI Strategy.

        Parameters:
        -----------
        ma_period : int
            Moving average period
        rsi_period : int
            RSI calculation period
        rsi_oversold : int
            RSI oversold threshold
        rsi_overbought : int
            RSI overbought threshold
        use_macd : bool
            Whether to use MACD for confirmation
        use_bollinger : bool
            Whether to use Bollinger Bands for confirmation
        """
        super().__init__(ma_period, rsi_period, rsi_oversold, rsi_overbought)
        self.use_macd = use_macd
        self.use_bollinger = use_bollinger

    def generate_signals(self, prices):
        """
        Generate enhanced buy/sell signals with MACD and Bollinger confirmation.

        Parameters:
        -----------
        prices : pd.Series
            Price series

        Returns:
        --------
        pd.DataFrame
            DataFrame with prices, indicators, and enhanced signals
        """
        # Get base signals
        signals = super().generate_signals(prices)

        # Add MACD indicators
        if self.use_macd:
            macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(prices)
            signals['MACD'] = macd_line
            signals['MACD_Signal'] = signal_line
            signals['MACD_Histogram'] = histogram

            # MACD confirmation: MACD line above signal line for buy confirmation
            macd_bullish = macd_line > signal_line

        # Add Bollinger Bands
        if self.use_bollinger:
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(prices)
            signals['BB_Upper'] = upper
            signals['BB_Middle'] = middle
            signals['BB_Lower'] = lower

            # Bollinger confirmation: price near lower band for buy opportunity
            bb_buy_zone = prices <= middle

        # Enhanced signal logic
        base_signal = signals['Signal'].copy()
        enhanced_signal = pd.Series(0, index=prices.index)

        # Buy with confirmation
        buy_mask = (base_signal == 1)
        if self.use_macd:
            buy_mask = buy_mask & macd_bullish
        if self.use_bollinger:
            buy_mask = buy_mask & bb_buy_zone

        # Sell with confirmation
        sell_mask = (base_signal == -1)

        enhanced_signal.loc[buy_mask] = 1
        enhanced_signal.loc[sell_mask] = -1

        signals['Enhanced_Signal'] = enhanced_signal
        signals['Position'] = enhanced_signal.replace(0, np.nan).ffill().fillna(0)

        return signals


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    """

    def __init__(self, initial_capital=100000, commission=0.001):
        """
        Initialize BacktestEngine.

        Parameters:
        -----------
        initial_capital : float
            Starting capital amount (default: $100,000)
        commission : float
            Commission rate per trade (default: 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run_backtest(self, prices, signals, signal_column='Signal'):
        """
        Run backtest with given prices and signals.

        Parameters:
        -----------
        prices : pd.Series
            Price series
        signals : pd.DataFrame
            DataFrame with trading signals
        signal_column : str
            Column name for signals ('Signal' or 'Enhanced_Signal')

        Returns:
        --------
        dict
            Dictionary with backtest results including positions, returns, and metrics
        """
        # Create results DataFrame
        results = pd.DataFrame(index=prices.index)
        results['Price'] = prices
        results['Signal'] = signals[signal_column]

        # Calculate daily returns
        results['Returns'] = prices.pct_change()

        # Calculate strategy returns (position * daily return)
        results['Position'] = signals['Position']
        results['Strategy_Returns'] = results['Position'].shift(1) * results['Returns']

        # Apply commission costs
        trades = results['Signal'].abs()
        commission_costs = trades * self.commission
        results['Strategy_Returns'] = results['Strategy_Returns'] - commission_costs

        # Calculate cumulative returns
        results['Cumulative_Returns'] = (1 + results['Returns']).cumprod()
        results['Strategy_Cumulative_Returns'] = (1 + results['Strategy_Returns'].fillna(0)).cumprod()

        # Calculate portfolio value
        results['Portfolio_Value'] = self.initial_capital * results['Strategy_Cumulative_Returns']

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        return {
            'results': results,
            'metrics': metrics
        }

    def _calculate_metrics(self, results):
        """
        Calculate performance metrics.

        Parameters:
        -----------
        results : pd.DataFrame
            DataFrame with backtest results

        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        strategy_returns = results['Strategy_Returns'].dropna()

        # Annualized Return (assuming 252 trading days)
        total_return = results['Strategy_Cumulative_Returns'].iloc[-1] - 1
        years = len(results) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = strategy_returns.mean() * 252 - risk_free_rate
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0

        # Maximum Drawdown
        cumulative = results['Strategy_Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Find drawdown dates
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


class PerformanceAnalyzer:
    """
    Analyze and compare strategy performance across multiple stocks.
    """

    def __init__(self, results_dict, benchmark_returns=None):
        """
        Initialize PerformanceAnalyzer.

        Parameters:
        -----------
        results_dict : dict
            Dictionary with ticker as key and backtest results as value
        benchmark_returns : pd.Series, optional
            Benchmark (SPY) cumulative returns for comparison
        """
        self.results_dict = results_dict
        self.benchmark_returns = benchmark_returns

    def get_summary_table(self):
        """
        Generate summary table of performance metrics.

        Returns:
        --------
        pd.DataFrame
            Summary table with metrics for each stock
        """
        summary_data = []

        for ticker, backtest_result in self.results_dict.items():
            metrics = backtest_result['metrics']
            summary_data.append({
                'Ticker': ticker,
                'Total Return': f"{metrics['total_return']*100:.2f}%",
                'Annualized Return': f"{metrics['annualized_return']*100:.2f}%",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
                'Num Trades': int(metrics['num_trades'])
            })

        return pd.DataFrame(summary_data)

    def plot_cumulative_returns(self, title="Cumulative Returns Comparison"):
        """
        Plot cumulative returns for all stocks and benchmark.

        Parameters:
        -----------
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot individual stock returns
        for ticker, backtest_result in self.results_dict.items():
            results = backtest_result['results']
            ax.plot(results.index, results['Strategy_Cumulative_Returns'],
                   label=ticker, linewidth=1.5, alpha=0.8)

        # Plot benchmark
        if self.benchmark_returns is not None:
            ax.plot(self.benchmark_returns.index, self.benchmark_returns,
                   label='SPY (Benchmark)', linewidth=2, color='black',
                   linestyle='--', alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_drawdowns(self, ticker, title=None):
        """
        Plot cumulative returns with drawdown periods highlighted.

        Parameters:
        -----------
        ticker : str
            Stock ticker to plot
        title : str, optional
            Custom plot title
        """
        if ticker not in self.results_dict:
            print(f"Ticker {ticker} not found in results")
            return None

        results = self.results_dict[ticker]['results']
        metrics = self.results_dict[ticker]['metrics']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Plot cumulative returns
        ax1.plot(results.index, results['Strategy_Cumulative_Returns'],
                label='Strategy Returns', color='blue', linewidth=1.5)

        if self.benchmark_returns is not None:
            ax1.plot(self.benchmark_returns.index, self.benchmark_returns,
                    label='SPY (Benchmark)', color='gray', linewidth=1.5,
                    linestyle='--', alpha=0.7)

        # Mark maximum drawdown point
        ax1.axvline(x=metrics['max_dd_date'], color='red', linestyle=':',
                   alpha=0.7, label=f'Max DD Date')
        ax1.scatter([metrics['max_dd_date']],
                   [results.loc[metrics['max_dd_date'], 'Strategy_Cumulative_Returns']],
                   color='red', s=100, zorder=5, marker='v')

        plot_title = title or f'{ticker} - Cumulative Returns with Drawdown'
        ax1.set_title(plot_title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Returns', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot drawdown
        cumulative = results['Strategy_Cumulative_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100

        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax2.axhline(y=metrics['max_drawdown']*100, color='black', linestyle='--',
                   label=f'Max DD: {metrics["max_drawdown"]*100:.2f}%')
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_portfolio_comparison(self, portfolio_weights=None):
        """
        Plot portfolio total value compared to benchmark.

        Parameters:
        -----------
        portfolio_weights : dict, optional
            Dictionary with ticker as key and weight as value
        """
        if portfolio_weights is None:
            # Equal weight portfolio
            n_stocks = len(self.results_dict)
            portfolio_weights = {ticker: 1/n_stocks for ticker in self.results_dict.keys()}

        # Calculate portfolio value
        portfolio_value = pd.DataFrame()

        for ticker, weight in portfolio_weights.items():
            if ticker in self.results_dict:
                results = self.results_dict[ticker]['results']
                portfolio_value[ticker] = results['Portfolio_Value'] * weight

        portfolio_value['Total'] = portfolio_value.sum(axis=1)

        # Calculate benchmark value
        if self.benchmark_returns is not None:
            benchmark_value = 100000 * self.benchmark_returns
        else:
            benchmark_value = None

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(portfolio_value.index, portfolio_value['Total'],
               label='Portfolio Value', linewidth=2, color='blue')

        if benchmark_value is not None:
            ax.plot(benchmark_value.index, benchmark_value,
                   label='SPY (Benchmark)', linewidth=2, color='gray',
                   linestyle='--')

        ax.set_title('Portfolio Total Value vs SPY Benchmark',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        return fig


def compare_strategies(prices, strategy_configs, ticker):
    """
    Compare different MA-RSI strategy configurations.

    Parameters:
    -----------
    prices : pd.Series
        Price series
    strategy_configs : list
        List of strategy configuration dictionaries
    ticker : str
        Stock ticker for labeling

    Returns:
    --------
    tuple
        (comparison DataFrame, results dictionary)
    """
    results = {}
    comparison_data = []

    for config in strategy_configs:
        strategy = MARSIStrategy(
            ma_period=config['ma_period'],
            rsi_period=config['rsi_period']
        )

        signals = strategy.generate_signals(prices)

        backtest = BacktestEngine()
        backtest_result = backtest.run_backtest(prices, signals)

        config_name = f"MA{config['ma_period']}-RSI{config['rsi_period']}"
        results[config_name] = backtest_result

        metrics = backtest_result['metrics']
        comparison_data.append({
            'Strategy': config_name,
            'Annualized Return': f"{metrics['annualized_return']*100:.2f}%",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{metrics['max_drawdown']*100:.2f}%",
            'Num Trades': int(metrics['num_trades'])
        })

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df, results


def plot_strategy_comparison(results_dict, ticker, benchmark_returns=None):
    """
    Plot cumulative returns for different strategy configurations.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with strategy name as key and backtest results as value
    ticker : str
        Stock ticker
    benchmark_returns : pd.Series, optional
        Benchmark cumulative returns
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, (strategy_name, backtest_result) in enumerate(results_dict.items()):
        results = backtest_result['results']
        ax.plot(results.index, results['Strategy_Cumulative_Returns'],
               label=strategy_name, linewidth=1.5,
               color=colors[idx % len(colors)], alpha=0.8)

    if benchmark_returns is not None:
        ax.plot(benchmark_returns.index, benchmark_returns,
               label='SPY (Benchmark)', linewidth=2, color='black',
               linestyle='--', alpha=0.7)

    ax.set_title(f'{ticker} - Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Main function to run the FAANG technical analysis.
    """
    print("=" * 60)
    print("FAANG Technical Analysis - MA-RSI Strategy")
    print("=" * 60)
    print()

    # Step 1: Fetch Data
    print("Step 1: Fetching historical data...")
    fetcher = DataFetcher(start_date='2014-01-01', end_date='2024-01-01')
    data = fetcher.fetch_data()

    close_prices = fetcher.get_close_prices(data)
    print(f"\nFetched data for {len(close_prices.columns)} tickers")
    print(f"Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print()

    # Extract benchmark (SPY) returns
    spy_returns = close_prices['SPY'].pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod()

    # Step 2: Define strategy configurations
    strategy_configs = [
        {'ma_period': 50, 'rsi_period': 14},  # Long-term MA, standard RSI
        {'ma_period': 10, 'rsi_period': 9},   # Short-term MA, short RSI
        {'ma_period': 10, 'rsi_period': 25},  # Short-term MA, long RSI
    ]

    print("Step 2: Testing different MA-RSI configurations...")
    print("Configurations:")
    for config in strategy_configs:
        print(f"  - MA{config['ma_period']}-RSI{config['rsi_period']}")
    print()

    # Step 3: Run backtests for each FAANG stock
    print("Step 3: Running backtests...")

    faang_tickers = ['AAPL', 'AMZN', 'NFLX', 'META', 'GOOG']
    all_results = {}

    # Use the best performing configuration (MA50-RSI14) for main analysis
    best_strategy = MARSIStrategy(ma_period=50, rsi_period=14)
    backtest_engine = BacktestEngine()

    for ticker in faang_tickers:
        if ticker not in close_prices.columns:
            print(f"  Warning: {ticker} data not available")
            continue

        print(f"  Backtesting {ticker}...")
        prices = close_prices[ticker].dropna()

        signals = best_strategy.generate_signals(prices)
        backtest_result = backtest_engine.run_backtest(prices, signals)
        all_results[ticker] = backtest_result

    print()

    # Step 4: Display performance summary
    print("Step 4: Performance Summary (MA50-RSI14 Strategy)")
    print("-" * 80)

    analyzer = PerformanceAnalyzer(all_results, spy_cumulative)
    summary_table = analyzer.get_summary_table()
    print(summary_table.to_string(index=False))
    print()

    # Step 5: Compare strategies for AAPL
    print("Step 5: Comparing different MA-RSI configurations for AAPL...")
    print("-" * 80)

    aapl_prices = close_prices['AAPL'].dropna()
    comparison_df, comparison_results = compare_strategies(
        aapl_prices, strategy_configs, 'AAPL'
    )
    print(comparison_df.to_string(index=False))
    print()

    # Step 6: Create visualizations
    print("Step 6: Creating visualizations...")

    # Plot cumulative returns for all FAANG stocks
    fig1 = analyzer.plot_cumulative_returns(
        title="FAANG Stocks - MA50-RSI14 Strategy Cumulative Returns"
    )
    fig1.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/faang_cumulative_returns.png',
                 dpi=150, bbox_inches='tight')
    print("  Saved: faang_cumulative_returns.png")

    # Plot drawdown for AAPL
    fig2 = analyzer.plot_drawdowns('AAPL')
    fig2.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/aapl_drawdown.png',
                 dpi=150, bbox_inches='tight')
    print("  Saved: aapl_drawdown.png")

    # Plot portfolio comparison
    fig3 = analyzer.plot_portfolio_comparison()
    fig3.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/portfolio_comparison.png',
                 dpi=150, bbox_inches='tight')
    print("  Saved: portfolio_comparison.png")

    # Plot strategy comparison for AAPL
    fig4 = plot_strategy_comparison(comparison_results, 'AAPL', spy_cumulative)
    fig4.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/strategy_comparison.png',
                 dpi=150, bbox_inches='tight')
    print("  Saved: strategy_comparison.png")

    print()

    # Step 7: Enhanced Strategy with MACD and Bollinger Bands
    print("Step 7: Testing Enhanced MA-RSI Strategy (with MACD & Bollinger Bands)...")
    print("-" * 80)

    enhanced_strategy = EnhancedMARSIStrategy(
        ma_period=50, rsi_period=14, use_macd=True, use_bollinger=True
    )

    enhanced_results = {}
    for ticker in faang_tickers:
        if ticker not in close_prices.columns:
            continue

        prices = close_prices[ticker].dropna()
        signals = enhanced_strategy.generate_signals(prices)

        # Use enhanced signal
        backtest_result = backtest_engine.run_backtest(
            prices, signals, signal_column='Enhanced_Signal'
        )
        enhanced_results[ticker] = backtest_result

    enhanced_analyzer = PerformanceAnalyzer(enhanced_results, spy_cumulative)
    enhanced_summary = enhanced_analyzer.get_summary_table()
    print(enhanced_summary.to_string(index=False))
    print()

    # Plot enhanced strategy results
    fig5 = enhanced_analyzer.plot_cumulative_returns(
        title="Enhanced MA-RSI Strategy (with MACD & Bollinger Bands)"
    )
    fig5.savefig('/Users/wangnuo/Desktop/Technical Analysis for FAANG/enhanced_strategy_returns.png',
                 dpi=150, bbox_inches='tight')
    print("  Saved: enhanced_strategy_returns.png")

    print()
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    # Close all figures
    plt.close('all')

    return {
        'close_prices': close_prices,
        'all_results': all_results,
        'enhanced_results': enhanced_results,
        'comparison_results': comparison_results,
        'spy_cumulative': spy_cumulative
    }


if __name__ == "__main__":
    results = main()
