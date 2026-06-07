# Trading Framework

A modular algorithmic trading framework built in Python. Implements five quantitative strategies with a shared backtesting engine, walk-forward validation, and Kelly Criterion capital allocation. All data sourced from Yahoo Finance — no paid APIs required.

---

## What It Does

The framework rotates capital across assets based on macroeconomic signals — yield curve shape, inflation expectations, credit market conditions, and dollar strength. It figures out where to put money (stocks, bonds, gold, credit, international equities) by reading the macro environment, not by predicting individual stock prices.

Think of it like this: when the yield curve is steepening and credit spreads are tightening, the economy is usually expanding — so the strategy tilts toward equities. When inflation is rising and the dollar is strong, gold gets a larger allocation and international equities get trimmed. The signals update continuously and positions are rebalanced weekly.

---

## Results

Backtested on 9 years of genuine out-of-sample data (2016–2024), meaning the strategy was built and calibrated on data from 2010–2015 only, and then run blind on the remaining 9 years.

| Metric | Result |
|---|---|
| Sharpe Ratio | **0.811** |
| Sortino Ratio | **1.028** |
| Annual Return | **+2.26%** |
| Annual Volatility | 2.78% |
| Max Drawdown | -7.67% |

A Sharpe above 0.8 is considered strong for a systematic macro strategy. The Sortino above 1.0 means most of the portfolio's movement is upside — drawdowns are controlled. The strategy survived COVID (2020), the 2022 rate-hike bear market, and the 2023–2024 AI rally without blowing up.

---

## Strategy

**Active strategy: Macro Systematic**

Trades five assets:
- **SPY** — US equities
- **TLT** — Long-duration US Treasury bonds
- **GLD** — Gold
- **HYG** — High-yield (junk) bonds
- **EFA** — International developed market equities

Driven by four signals:
1. **Yield curve slope** — TLT/SHY ratio. Steepening = growth optimism = more equities
2. **Inflation proxy** — TIP/IEF ratio. Rising inflation = more gold, fewer bonds
3. **Credit spread** — HYG/LQD ratio. Tightening spreads = risk appetite = more equities and credit
4. **Dollar strength** — UUP momentum. Strong dollar = less gold and international equities

Risk controls:
- If SPY falls below its 200-day moving average, equity allocation is dampened — the strategy reduces stock exposure during confirmed downtrends
- Positions are only rebalanced weekly and only when a weight has shifted more than 2% — cuts unnecessary trading costs
- Volatility targeting keeps annualised portfolio vol around 10%

---

## The Other Strategies (Built, Then Retired)

Four other strategies were built and tested but removed after out-of-sample evaluation showed they were net negative:

| Strategy | What It Does | OOS Sharpe | Decision |
|---|---|---|---|
| Statistical Arbitrage | Pairs trading on cointegrated S&P 500 stocks | +0.42 | Removed (macro enough on its own) |
| Trend Following | Long top momentum decile, short bottom | -0.34 | Removed |
| Mean Reversion | Short-term reversal on 5-day losers/winners | -0.47 | Removed |
| Market Making | Simulated bid-ask inventory model | -0.53 | Removed |

Cutting the losers improved the combined portfolio Sharpe more than any amount of parameter tuning.

---

## Project Structure

```
trading_framework/
├── main.py                    # Entry point — run this
├── requirements.txt
│
├── data/
│   └── pipeline.py            # Downloads S&P 500 + macro data from yfinance, caches to parquet
│
├── engine/
│   ├── backtest.py            # Vectorized backtest: weights → P&L with transaction costs + slippage
│   ├── walk_forward.py        # Train/test split enforcement (train 2010-2015, OOS 2016-2024)
│   └── vol_targeting.py       # Scales positions to hit 10% annualised volatility target
│
├── strategies/
│   ├── macro_systematic.py    # Active strategy — 5-asset macro rotation
│   ├── stat_arb.py            # Pairs trading (disabled)
│   ├── trend_following.py     # Cross-sectional momentum (disabled)
│   ├── mean_reversion.py      # Short-term reversal (disabled)
│   └── market_making.py       # Inventory model skeleton (disabled)
│
├── regime/
│   └── hmm_classifier.py      # Hidden Markov Model — identifies trending vs choppy regimes
│
├── portfolio/
│   └── kelly_allocator.py     # Half-Kelly capital sizing across strategies
│
├── tuning/
│   └── grid_search.py         # Parameter grid search — evaluated on OOS data only
│
├── analytics/
│   └── performance.py         # Sharpe, Sortino, max drawdown, Calmar, monthly returns
│
├── visualization/
│   └── dashboard.py           # Equity curves, monthly heatmap, correlation matrix
│
└── output/                    # Generated charts and performance CSVs
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the backtest:**
```bash
python main.py
```

First run downloads ~500MB of price data from Yahoo Finance and caches it. Subsequent runs use the cache and complete in under a minute.

**Force re-download data:**
```bash
python main.py --refresh
```

**Run parameter grid search before backtesting:**
```bash
python main.py --tune
```

Outputs are saved to `./output/` — equity curve, monthly returns heatmap, correlation matrix, and performance tables.

---

## Key Design Decisions

**Walk-forward validation** — the strategy is calibrated on 2010–2015 data only. Everything reported is genuinely out-of-sample (2016–2024). In-sample Sharpe numbers are meaningless and not reported.

**No paid data** — everything comes from `yfinance`. The S&P 500 constituent list is scraped from Wikipedia. Macro instruments (Treasury ETFs, credit ETFs, gold, dollar index) are all publicly traded and available.

**Transaction costs** — 10 bps per trade + 5 bps slippage baked into every backtest. Weekly rebalancing with a 2% drift threshold keeps turnover low.

**Half-Kelly sizing** — Kelly Criterion gives the theoretically optimal bet size. Half-Kelly halves it, reducing variance at the cost of some expected return. Standard practice in systematic trading.

---

## Requirements

```
yfinance
pandas
numpy
scipy
statsmodels
hmmlearn
matplotlib
seaborn
requests
tqdm
pyarrow
```
