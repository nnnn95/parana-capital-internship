# Technical Analysis for FAANG Stocks

## Project Overview

This project implements MA-RSI technical analysis for FAANG stocks (AAPL, AMZN, NFLX, META, GOOG) over a 10-year period (2014-2024). The analysis compares different parameter settings and evaluates strategy performance against the SP500 benchmark.

## Project Structure

```
Technical Analysis for FAANG/
├── run_all_analysis.py          # Main analysis script
├── faang_technical_analysis.py  # Core analysis classes
├── bonus_sector_comparison.py   # Bonus: Sector comparison
├── README.md                    # Project documentation
│
├── fig1_cumulative_returns.png  # FAANG cumulative returns
├── fig2_drawdown_analysis.png   # Drawdown analysis
├── fig3_portfolio_vs_benchmark.png  # Portfolio vs SPY
├── fig4_strategy_comparison.png # MA-RSI configuration comparison
├── fig5_aapl_signals.png        # Trading signals visualization
├── fig6_performance_heatmap.png # Performance metrics heatmap
└── fig7_bonus_indicators.png    # MACD & Bollinger Bands
```

## Key Features

### 1. Data Collection
- Fetched 10 years of daily data for FAANG stocks and SPY benchmark
- Data source: Yahoo Finance (yfinance library)
- Period: 2014-01-01 to 2024-01-01

### 2. Technical Indicators Implemented
- **Simple Moving Average (SMA)**: Trend identification
- **Relative Strength Index (RSI)**: Momentum oscillator
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility indicator

### 3. MA-RSI Strategy Logic
- **Buy Signal**: Price above MA AND RSI crosses above 30 (from oversold)
- **Sell Signal**: Price below MA AND RSI crosses below 70 (from overbought)

### 4. Strategy Configurations Tested
| Strategy | MA Period | RSI Period |
|----------|-----------|------------|
| MA50-RSI14 | 50 | 14 |
| MA10-RSI9 | 10 | 9 |
| MA10-RSI25 | 10 | 25 |

### 5. Performance Metrics
- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades

## Performance Results (MA50-RSI14 Strategy)

| Ticker | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown | # Trades |
|--------|-------------|-------------------|--------------|--------------|----------|
| AAPL | 349.70% | 16.25% | 0.61 | -38.52% | 13 |
| AMZN | -82.19% | -15.87% | -0.43 | -90.71% | 12 |
| NFLX | -32.33% | -3.84% | 0.08 | -86.39% | 23 |
| META | 32.06% | 2.82% | 0.20 | -77.06% | 17 |
| GOOG | -37.92% | -4.66% | -0.11 | -66.97% | 11 |

## Strategy Comparison (AAPL)

| Strategy | Annualized Return | Sharpe Ratio | Max Drawdown | # Trades |
|----------|-------------------|--------------|--------------|----------|
| MA50-RSI14 | 16.25% | 0.61 | -38.52% | 13 |
| MA10-RSI9 | -11.91% | -0.38 | -88.23% | 70 |
| MA10-RSI25 | -15.59% | -0.54 | -85.65% | 35 |

## Key Findings

1. **Best Performing Stock**: AAPL with 349.70% total return and 0.61 Sharpe ratio
2. **Best Strategy Configuration**: MA50-RSI14 outperforms shorter-term configurations
3. **Trade Frequency**: Shorter MA periods generate more trades but lower returns
4. **Risk Management**: MA50-RSI14 has the lowest maximum drawdown among tested configurations

## Bonus Features

### MACD Analysis
- MACD Line: 12-period EMA minus 26-period EMA
- Signal Line: 9-period EMA of MACD Line
- Used for trend confirmation

### Bollinger Bands
- Middle Band: 20-period SMA
- Upper/Lower Bands: ±2 standard deviations
- Used for volatility assessment

## Requirements

```python
# Python 3.7+
pip install numpy pandas yfinance matplotlib seaborn
```

## Usage

Run the main analysis script:
```bash
python3 run_all_analysis.py
```

## Output Visualizations

1. **fig1_cumulative_returns.png**: Compares FAANG strategy returns against SPY benchmark
2. **fig2_drawdown_analysis.png**: Shows drawdown patterns for each stock
3. **fig3_portfolio_vs_benchmark.png**: Equal-weighted portfolio value vs SPY
4. **fig4_strategy_comparison.png**: Different MA-RSI configurations on AAPL
5. **fig5_aapl_signals.png**: Price chart with buy/sell signals
6. **fig6_performance_heatmap.png**: Heatmap of key performance metrics
7. **fig7_bonus_indicators.png**: MACD and Bollinger Bands analysis

## Notes

- The MA-RSI strategy performs best with longer MA periods (50 days)
- Short-term configurations (MA10) generate too many false signals
- Only AAPL outperformed the SPY benchmark using this strategy
- Consider combining with additional indicators for better results

## Author

Intern Project - 2024
