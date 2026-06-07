# ICT Trading Bot

A fully systematic, backtested trading bot implementing **ICT (Inner Circle Trader)** methodology. Detects institutional price-action structures in market data and generates entries/exits with strict risk management.

---

## What It Does

The bot identifies five core ICT concepts and combines them into a rules-based trading system:

| Concept | Description |
|---------|-------------|
| **Fair Value Gap (FVG)** | 3-candle price imbalance where the market tends to return and fill |
| **Order Block (OB)** | Last opposing candle before a strong institutional displacement move |
| **Breaker Block** | Mitigated OB that flips polarity — former support becomes resistance |
| **Market Structure Shift (MSS)** | Break of a swing high/low confirmed by a displacement candle |
| **OTE** | Optimal Trade Entry — Fibonacci 61.8–78.6% retracement scoring |

**Entry logic:** MSS confirms direction → price returns to an unmitigated FVG or OB → Kill Zone active (or Mon–Thu on daily) → ADX trending → daily bias aligned → enter.

**Exit logic:** ATR-based stop below/above zone edge, partial exit 50% at 1.5R (trail stop to breakeven), full exit at 2.0R.

---

## Project Structure

```
ict-bot/
├── ict_data.py            # Data download + daily bias (50/200 EMA)
├── ict_structures.py      # FVG, OB, Breaker, MSS, Kill Zone, ADX, OTE
├── ict_backtest.py        # 1h single-instrument backtest engine
├── ict_portfolio.py       # Daily multi-instrument portfolio engine
├── ict_viz.py             # Annotated charts, equity curve, drawdown plots
├── build_notebook.py      # Generates ICT_Bot.ipynb (2yr hourly)
├── build_notebook_15y.py  # Generates ICT_Bot_15Y.ipynb (15yr daily)
├── ICT_Bot.ipynb          # SPY 1h notebook — 2 years
├── ICT_Bot_15Y.ipynb      # SPY+QQQ+GLD+TLT daily notebook — 15 years
└── DEVLOG.md              # Full development log
```

---

## Results

### Hourly Backtest — SPY, 2 Years (v2)

| Metric | Value |
|--------|-------|
| Total trades | 67 |
| Win rate | 40.3% |
| Avg R per trade | +0.39R |
| Profit factor | 1.43× |
| Sharpe (annual) | +0.46 |
| Max drawdown | -16.4% |
| Total P&L | **+$17,572 (+17.6%)** |
| CAGR | 4.81% |
| Beta (β) | 0.07 |
| Alpha (α, Jensen) | -1.28%/yr |
| Calmar (γ) | 0.29 |

---

### Daily Portfolio Backtest — SPY + QQQ + GLD + TLT, 15 Years (v3)

| Metric | Value |
|--------|-------|
| Total trades | 121 |
| Trades / year | ~8.1 |
| Win rate | 52.1% |
| Avg R per trade | +0.38R |
| Profit factor | 1.52× |
| Sharpe (annual) | 0.53 |
| Sharpe (per-trade) | 0.75 |
| Max drawdown | **-5.12%** |
| Calmar (γ) | **1.22** |
| Beta (β) | **0.008** (near-zero market exposure) |
| Alpha (α, Jensen) | **+2.17% / year** |
| R skewness | +0.19 (slight right tail) |
| Total P&L | **+$148,024 (+148%)** |
| **CAGR (compounded)** | **6.25% / year** |
| SPY buy-and-hold CAGR | 13.65% / year |
| Final equity (from $100k) | **$248,024** |

Key characteristic: **near-zero beta (0.008)** — returns are almost entirely uncorrelated with the market. The strategy generates true alpha through selective ICT entries rather than riding market beta.

---

## Strategy Parameters

```python
# Risk
INITIAL_CAPITAL  = 100_000
RISK_PCT         = 0.01       # 1% equity per trade
REWARD_R         = 2.0        # Full target
PARTIAL_R        = 1.5        # Partial exit (50%) and trail to breakeven
MAX_CONCURRENT   = 4          # Portfolio cap

# Filters
ADX_THRESHOLD       = 18.0    # Long entries
ADX_SHORT_THRESHOLD = 25.0    # Short entries (stricter)
MSS_EXPIRY_BARS     = 45      # Daily bars (~9 weeks)
MAX_ZONE_AGE        = 30      # Bars before zone is stale

# Stops
ATR_STOP_BUFFER  = 1.0        # ATR units below zone edge
MIN_STOP_PCT     = 0.005      # 0.5% minimum floor

# Costs
TRANSACTION_COST = 0.0005     # 5bps per side
RF_ANNUAL        = 0.04       # Idle cash earns T-bill rate
```

---

## How to Run

**Prerequisites:**
```bash
pip install yfinance pandas numpy matplotlib jupyter
```

**Generate and open notebooks:**
```bash
# 2-year hourly backtest (SPY)
python build_notebook.py
jupyter notebook ICT_Bot.ipynb

# 15-year daily portfolio backtest
python build_notebook_15y.py
jupyter notebook ICT_Bot_15Y.ipynb
```

---

## What We Improved (v1 → v3)

The v1 strategy lost 88% of capital. Key root-cause fixes:

1. **ATR-based stops** — replaced a fatal fixed 0.1% buffer that was tighter than 1h noise (~$0.55 on a $550 stock vs $2–4 ATR)
2. **50/200 EMA bias filter** — replaced 20/50 to filter regime vs micro-oscillations
3. **OB displacement threshold 1.5× → 2.0× ATR** — requires strong conviction moves only
4. **MSS swing 5 → 10 candles, body > 1.5× ATR** — fewer but higher-conviction breaks
5. **Zone staleness filter** — discard FVGs/OBs older than MAX_ZONE_AGE bars
6. **ADX filter** — no entries in ranging/choppy markets
7. **Short filter: ADX > 25 + price < SMA200** — prevents counter-trend shorts in bull markets
8. **Partial exit at 1.5R** — dramatically reduces variance, improves Calmar ratio
9. **Idle capital T-bill return** — biggest CAGR lift; idle cash earns 4%/yr in money market
10. **Multi-instrument (4 assets)** — increases signal frequency without increasing correlation

See `DEVLOG.md` for the full development log including bugs, fixes, and learnings.

---

## Limitations

- Low trade frequency (~8 trades/year per instrument on daily bars) makes statistical significance limited
- yfinance data quality may differ from actual broker fills
- No slippage model beyond flat 5bps transaction cost
- Daily bars miss intraday kill zone timing signals
- Live trading would require broker API integration

---

*Built using Python, pandas, numpy, yfinance, and matplotlib.*
