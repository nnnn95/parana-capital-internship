# Parana Capital 实习项目

Parana Capital 量化研究实习期间完成的三项投资分析任务，覆盖数据采集、策略回测、技术分析与风险评估。完整技术文档见 [`Parana实习日志.md`](./Parana实习日志.md)。

## 任务总览

| 任务 | 目录 | 内容 |
|---|---|---|
| 任务一 | [`Task1-QQQ-TQQQ-Data-Collection/`](./Task1-QQQ-TQQQ-Data-Collection) | QQQ vs TQQQ 日线行情采集管线 + 1×/3× 杠杆 ETF 长期收益对比 |
| 任务二 | [`Task2-QQQ-DCA-vs-Momentum/`](./Task2-QQQ-DCA-vs-Momentum) | QQQ 定投策略：DCA（定额定投）vs 动量投资 10 年回测 |
| 任务三 | [`Task3-FAANG-Technical-Analysis/`](./Task3-FAANG-Technical-Analysis) | FAANG 股票 MA-RSI 技术分析策略回测 + 板块横向对比 |

---

## 任务一｜QQQ vs TQQQ 数据采集与杠杆 ETF 对比

构建 QQQ（1 倍纳指 100 ETF）与 TQQQ（3 倍杠杆纳指 ETF）的日线行情采集管线，产出 2014-01-02 至 2023-12-29 共 2,516 个交易日的干净数据集，并基于数据对比杠杆 vs 非杠杆 ETF 的长期复利放大效应。

- 数据源：Stooq 历史行情 API，含 3 次重试 + 退避、响应合法性校验、字段归一化
- 关键结论：QQQ 10 年总收益约 +412%（年化 17.75%），TQQQ 总收益约 +1,957%（年化 35.31%）；TQQQ 累计收益约为 QQQ 的 4.75 倍，验证 3 倍杠杆在上行通道的复利放大效应

详见 [`Task1-QQQ-TQQQ-Data-Collection/`](./Task1-QQQ-TQQQ-Data-Collection)（`fetch_data.py` / `main_dr.py` + `QQQ.csv` / `TQQQ.csv`）。

## 任务二｜QQQ 定投策略：DCA vs 动量投资

2014-01-01 至 2023-12-31 共 10 年回测，每月投入 500 美元，对比 DCA 定投与 6 月动量信号择时策略，计算年化收益、总收益、最大回撤、波动率、Sharpe，并手写 Newton-Raphson 迭代求解 XIRR（货币加权收益率）。

- 数据源三级 fallback：yfinance（auto_adjust 复权）→ akshare（qfq）→ 本地 Stooq CSV
- 方法论亮点：DCA 是分期现金流，CAGR 会高估收益，故额外手写 IRR 求解；动量信号剔除最近一月以规避短期反转噪声
- 回测结果：DCA 收益倍数 2.59 倍、Sharpe 1.50；动量 2.17 倍、Sharpe 1.52；DCA 绝对收益占优，风险调整后两者持平

文件夹含多个迭代脚本（探索未复权 vs 复权数据源、CAGR vs XIRR 算法口径），定性结论在不同口径下高度稳健。详见 [`Task2-QQQ-DCA-vs-Momentum/`](./Task2-QQQ-DCA-vs-Momentum) 与结果报告 `qqq_strategy_results.md`。

## 任务三｜FAANG 股票技术分析（MA-RSI 策略回测）

对 AAPL、AMZN、NFLX、META、GOOG 实现 MA-RSI 技术分析策略，回测 2014-01-01 至 2024-01-01 共 10 年，以 SPY 为基准计算年化收益、Sharpe、最大回撤，对比不同 MA/RSI 参数，附加 MACD + 布林带增强，并与金融、能源板块横向对比。

- 技术指标全部手写：SMA / EMA / RSI / MACD / 布林带
- 前视偏差规避：策略收益用 `Position.shift(1) * 日收益率` 计算
- 关键结论：长周期 MA50-RSI14 显著优于短周期（交易次数从 70 降至 13，最大回撤从 -88% 收窄至 -38%，年化由负转正）；10 年窗口内仅 AAPL 跑赢 SPY 基准

详见 [`Task3-FAANG-Technical-Analysis/`](./Task3-FAANG-Technical-Analysis)（含自带 `README.md` 与 7 张主图）。

---

## 数据源与工程化要点

三任务覆盖 yfinance、Stooq、akshare 三个数据源并互为 fallback，含重试退避、响应合法性校验、MultiIndex 处理、SSL/超时配置、字段归一化等工程细节。

## 环境依赖

```bash
pip install pandas numpy matplotlib seaborn yfinance
# 可选：akshare（任务二备用数据源）
```

## 运行

```bash
# 任务一：采集 QQQ/TQQQ 数据
python3 Task1-QQQ-TQQQ-Data-Collection/fetch_data.py

# 任务二：DCA vs 动量回测
python3 Task2-QQQ-DCA-vs-Momentum/qqq_final_analysis.py

# 任务三：FAANG 技术分析（生成全部图表）
python3 Task3-FAANG-Technical-Analysis/run_all_analysis.py
```

## 作者

Parana Capital 量化研究实习项目
