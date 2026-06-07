"""
Script to generate the ICT bot Jupyter notebook.
Run: python3 build_notebook.py
Then: jupyter notebook ICT_Bot.ipynb
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ── Cell 0: title ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# ICT Trading Bot — Systematic Backtest

**Strategy:** Fair Value Gaps · Order Blocks · Breaker Blocks (optional) · Market Structure Shifts
**Instrument:** SPY 1-Hour · 2 Years
**Capital:** $100,000 · 1% risk/trade · 2.5R target · 0.05% transaction cost
**Bias filter:** Daily 50/200 EMA (golden/death cross) · Kill Zone: NY Open 13:30–16:00 UTC

---
### Tuning parameters (edit Cell 3)
| Parameter | Default | Effect |
|-----------|---------|--------|
| `OB_ATR_MULT` | 2.0 | Minimum displacement strength for Order Block |
| `MSS_EXPIRY_BARS` | 250 | Bars before an MSS is considered stale |
| `MAX_ZONE_AGE` | 100 | Max age of FVG/OB zone to remain tradeable |
| `ATR_STOP_BUFFER` | 0.5 | ATR multiples of cushion below/above zone edge |
| `USE_BREAKERS` | False | Enable Breaker Block signals (more trades, higher drawdown) |
"""))

# ── Cell 1: imports ─────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 130

from ict_data import download_hourly, download_daily, compute_daily_bias, align_bias_to_hourly
from ict_structures import (
    detect_fvgs, update_fvg_mitigation,
    detect_order_blocks, update_ob_mitigation,
    detect_breaker_blocks, detect_mss, add_kill_zone_column,
)
from ict_backtest import run_backtest, compute_metrics
import ict_backtest as bt
from ict_viz import plot_price_with_structures, plot_equity_and_drawdown, print_metrics_table

print("All modules loaded.")
"""))

# ── Cell 2: download data ────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 1. Data Download"))
cells.append(new_code_cell("""\
TICKER = "SPY"

df_1h    = download_hourly(TICKER, period="2y")
df_daily = download_daily(TICKER, period="3y")

print(f"Hourly bars: {len(df_1h)}")
print(f"Daily bars:  {len(df_daily)}")
df_1h.tail(3)
"""))

# ── Cell 3: strategy parameters ──────────────────────────────────────────────
cells.append(new_markdown_cell("## 2. Strategy Parameters"))
cells.append(new_code_cell("""\
# ── All tunable parameters live here ──────────────────────────────────────
OB_ATR_MULT     = 2.0   # displacement strength for Order Block detection
MSS_EXPIRY_BARS = 250   # bars before an MSS loses its gating power
MAX_ZONE_AGE    = 100   # max bars a FVG/OB zone stays tradeable
ATR_STOP_BUFFER = 0.5   # ATR multiples of cushion beyond zone edge for stop
USE_BREAKERS    = False  # True = add Breaker Block signals (higher trades, higher DD)

# Push parameters into backtest engine
bt.MSS_EXPIRY_BARS = MSS_EXPIRY_BARS
bt.MAX_ZONE_AGE    = MAX_ZONE_AGE
bt.ATR_STOP_BUFFER = ATR_STOP_BUFFER

print(f"Parameters set: OB_ATR={OB_ATR_MULT}, MSS_expiry={MSS_EXPIRY_BARS}bars, "
      f"zone_age={MAX_ZONE_AGE}bars, stop_buf={ATR_STOP_BUFFER}xATR, breakers={USE_BREAKERS}")
"""))

# ── Cell 4: daily bias ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 3. Daily Bias (50/200 EMA — Golden/Death Cross)"))
cells.append(new_code_cell("""\
daily_bias = compute_daily_bias(df_daily, ema_fast=50, ema_slow=200)
hourly_bias = align_bias_to_hourly(df_1h, daily_bias)

bull_days = (daily_bias == 'bull').sum()
bear_days = (daily_bias == 'bear').sum()
print(f"Bull days: {bull_days}  |  Bear days: {bear_days}  |  "
      f"Bull %: {bull_days/(bull_days+bear_days)*100:.1f}%")

fig, ax = plt.subplots(figsize=(16, 3), facecolor='#0d0d0d')
ax.set_facecolor('#0d0d0d')
close = df_daily['close']
ema50  = close.ewm(span=50, adjust=False).mean()
ema200 = close.ewm(span=200, adjust=False).mean()
ax.plot(close.index, close,  color='#e0e0e0', linewidth=0.8, label='SPY Daily', alpha=0.7)
ax.plot(ema50.index,  ema50,  color='#00cc66', linewidth=1.4, label='EMA 50')
ax.plot(ema200.index, ema200, color='#ff6600', linewidth=1.4, label='EMA 200')
ax.fill_between(close.index, ema50, ema200,
                where=ema50 >= ema200, alpha=0.12, color='#00cc66', label='Bull regime')
ax.fill_between(close.index, ema50, ema200,
                where=ema50 < ema200,  alpha=0.12, color='#ff4444', label='Bear regime')
ax.legend(fontsize=8, framealpha=0.3, facecolor='#0d0d0d')
ax.set_title('Daily Bias Filter — 50/200 EMA', color='#cccccc')
ax.tick_params(colors='#cccccc')
for spine in ax.spines.values():
    spine.set_edgecolor('#333333')
plt.tight_layout()
plt.show()
"""))

# ── Cell 5: structure detection ──────────────────────────────────────────────
cells.append(new_markdown_cell("## 4. ICT Structure Detection"))
cells.append(new_code_cell("""\
print("Detecting Fair Value Gaps (min 0.15 ATR gap size)...")
fvgs = detect_fvgs(df_1h, min_gap_atr=0.15)
fvgs = update_fvg_mitigation(fvgs, df_1h)

print(f"Detecting Order Blocks (displacement >= {OB_ATR_MULT}x ATR)...")
obs  = detect_order_blocks(df_1h, atr_multiplier=OB_ATR_MULT)
obs  = update_ob_mitigation(obs, df_1h)

print("Detecting Breaker Blocks (from mitigated OBs)...")
breakers = detect_breaker_blocks(obs, df_1h)
if not USE_BREAKERS:
    breakers = pd.DataFrame(columns=['type','top','bottom','bar_index','timestamp','origin_type'])
    print("  Breakers disabled (USE_BREAKERS=False)")

print("Detecting Market Structure Shifts (10-candle swing, 1.5x ATR body)...")
mss = detect_mss(df_1h, swing_n=10)

print("Adding Kill Zone labels...")
df_1h = add_kill_zone_column(df_1h)

bull_fvg = (fvgs['type'] == 'bull').sum()
bear_fvg = (fvgs['type'] == 'bear').sum()
mit_fvg  = fvgs['mitigated'].sum()

print(f"\\n{'='*45}")
print(f"  FVGs detected:     {len(fvgs):>5}  (bull: {bull_fvg}, bear: {bear_fvg})")
print(f"  FVGs mitigated:    {mit_fvg:>5}  ({mit_fvg/len(fvgs)*100:.0f}%)")
print(f"  Order Blocks:      {len(obs):>5}  (bull: {(obs['type']=='bull').sum()}, bear: {(obs['type']=='bear').sum()})")
print(f"  Breaker Blocks:    {len(breakers):>5}")
print(f"  MSS signals:       {len(mss):>5}  (bull: {(mss['type']=='bull').sum()}, bear: {(mss['type']=='bear').sum()})")
kz_bars = df_1h['kill_zone'].notna().sum()
print(f"  Kill Zone bars:    {kz_bars:>5}  ({kz_bars/len(df_1h)*100:.1f}% of data)")
print(f"{'='*45}")
"""))

# ── Cell 6: annotated chart structures only ───────────────────────────────────
cells.append(new_markdown_cell("## 5. Annotated Price Chart — Structures Only (First 500 Bars)"))
cells.append(new_code_cell("""\
fig = plot_price_with_structures(
    df_1h, fvgs, obs, breakers, mss,
    trades=[],
    title=f"{TICKER} 1H — ICT Structures (first 500 bars)",
    max_bars=500, start_bar=0,
)
plt.show()
"""))

# ── Cell 7: backtest ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 6. Backtest Execution"))
cells.append(new_code_cell("""\
print("Running backtest...")
trades, equity_curve = run_backtest(
    df_1h, fvgs, obs, breakers, mss, daily_bias
)
print(f"Complete. Total trades: {len(trades)}")
"""))

# ── Cell 8: metrics ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 7. Performance Metrics"))
cells.append(new_code_cell("""\
if not trades:
    print("No trades generated. Try increasing MSS_EXPIRY_BARS or relaxing ATR_STOP_BUFFER.")
else:
    metrics = compute_metrics(trades, equity_curve)
    print_metrics_table(metrics)
"""))

# ── Cell 9: trade log ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 8. Trade Log"))
cells.append(new_code_cell("""\
trade_log = metrics['trade_log']
print(f"Showing last 20 of {len(trade_log)} trades:")
display_cols = ['direction','entry_time','exit_time','entry_price','exit_price',
                'exit_reason','pnl','r_multiple','kill_zone','signal_source','ote_score']
trade_log[display_cols].tail(20).style\\
    .format({'pnl': '${:.2f}', 'r_multiple': '{:.2f}R', 'ote_score': '{:.3f}',
             'entry_price': '{:.2f}', 'exit_price': '{:.2f}'})\\
    .applymap(lambda v: 'color: #00cc66' if isinstance(v, (int, float)) and v > 0 else
                        'color: #ff4444' if isinstance(v, (int, float)) and v < 0 else '',
              subset=['pnl', 'r_multiple'])
"""))

# ── Cell 10: equity curve ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 9. Equity Curve & Drawdown"))
cells.append(new_code_cell("""\
fig = plot_equity_and_drawdown(equity_curve, trades)
plt.show()
"""))

# ── Cell 11: chart with trades ────────────────────────────────────────────────
cells.append(new_markdown_cell("## 10. Price Chart with Trades (Most Recent 500 Bars)"))
cells.append(new_code_cell("""\
start = max(0, len(df_1h) - 500)
fig = plot_price_with_structures(
    df_1h, fvgs, obs, breakers, mss,
    trades=trades,
    title=f"{TICKER} 1H — ICT Trades (last 500 bars)",
    max_bars=500, start_bar=start,
)
plt.show()
"""))

# ── Cell 12: kill zone & source analysis ──────────────────────────────────────
cells.append(new_markdown_cell("## 11. Kill Zone & Signal Source Analysis"))
cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0d0d0d')
tl = metrics['trade_log']

kz = metrics['kill_zone_stats']
ax = axes[0]
ax.set_facecolor('#0d0d0d')
bar_colors = ['#00cc66' if v >= 0.286 else '#ff4444' for v in kz['win_rate']]
bars = ax.bar(kz.index, kz['win_rate'] * 100, color=bar_colors, alpha=0.85)
ax.axhline(28.6, color='#ffcc00', linewidth=1, linestyle='--', label='Break-even (28.6%)')
ax.set_title('Win Rate by Kill Zone', color='#cccccc', fontsize=11)
ax.set_ylabel('Win Rate %', color='#cccccc')
ax.set_ylim(0, 100)
ax.legend(fontsize=7, framealpha=0.3, facecolor='#0d0d0d')
for bar, val in zip(bars, kz['win_rate'] * 100):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', color='#cccccc', fontsize=9)

ax = axes[1]
ax.set_facecolor('#0d0d0d')
colors_r = ['#00cc66' if v > 0 else '#ff4444' for v in kz['avg_r']]
ax.bar(kz.index, kz['avg_r'], color=colors_r, alpha=0.85)
ax.axhline(0, color='#555555', linewidth=0.8)
ax.set_title('Avg R-Multiple by Kill Zone', color='#cccccc', fontsize=11)
ax.set_ylabel('Avg R', color='#cccccc')

ss = metrics['source_stats']
ax = axes[2]
ax.set_facecolor('#0d0d0d')
colors_s = ['#00cc66' if v > 0 else '#ff4444' for v in ss['total_pnl']]
ax.bar(ss.index, ss['total_pnl'], color=colors_s, alpha=0.85)
ax.set_title('Total P&L by Signal Source', color='#cccccc', fontsize=11)
ax.set_ylabel('P&L ($)', color='#cccccc')
ax.tick_params(colors='#cccccc', labelrotation=15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

for ax in axes:
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

plt.tight_layout()
plt.show()
"""))

# ── Cell 13: OTE score analysis ───────────────────────────────────────────────
cells.append(new_markdown_cell("## 12. OTE Score Distribution vs Performance"))
cells.append(new_code_cell("""\
tl = metrics['trade_log'].copy()
tl['ote_bucket'] = pd.cut(tl['ote_score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
ote_stats = tl.groupby('ote_bucket', observed=True).agg(
    count=('pnl', 'count'),
    win_rate=('pnl', lambda x: (x > 0).mean()),
    avg_r=('r_multiple', 'mean'),
).reset_index()

print("OTE Score vs Performance (1.0 = perfectly at 61.8-78.6% fib retracement):")
print(ote_stats.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d0d0d')
for ax in (ax1, ax2):
    ax.set_facecolor('#0d0d0d')
    ax.tick_params(colors='#cccccc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

ax1.bar(ote_stats['ote_bucket'].astype(str), ote_stats['win_rate'] * 100,
        color='#4da6ff', alpha=0.85)
ax1.axhline(28.6, color='#ffcc00', linewidth=1, linestyle='--', label='Break-even')
ax1.set_title('Win Rate by OTE Score', color='#cccccc')
ax1.set_ylabel('Win Rate %', color='#cccccc')
ax1.set_xlabel('OTE Score (closer to 1 = tighter fib alignment)', color='#cccccc')
ax1.legend(fontsize=8, framealpha=0.3, facecolor='#0d0d0d')

colors_ote = ['#00cc66' if v > 0 else '#ff4444' for v in ote_stats['avg_r']]
ax2.bar(ote_stats['ote_bucket'].astype(str), ote_stats['avg_r'],
        color=colors_ote, alpha=0.85)
ax2.axhline(0, color='#555555', linewidth=0.8)
ax2.set_title('Avg R by OTE Score', color='#cccccc')
ax2.set_ylabel('Avg R', color='#cccccc')
ax2.set_xlabel('OTE Score', color='#cccccc')

plt.tight_layout()
plt.show()
print("\\nTip: if high-OTE trades outperform, add 'ote_score >= 0.4' as a signal gate.")
"""))

# ── Cell 14: monthly P&L heatmap ──────────────────────────────────────────────
cells.append(new_markdown_cell("## 13. Monthly P&L Heatmap"))
cells.append(new_code_cell("""\
tl = metrics['trade_log'].copy()
tl['year']  = tl['exit_time'].dt.year
tl['month'] = tl['exit_time'].dt.month
monthly = tl.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)

# Pad to 12 months
for m in range(1, 13):
    if m not in monthly.columns:
        monthly[m] = 0
monthly = monthly[sorted(monthly.columns)]

fig, ax = plt.subplots(figsize=(14, max(3, 2.5 * len(monthly))), facecolor='#0d0d0d')
ax.set_facecolor('#0d0d0d')

import matplotlib.colors as mcolors
max_val = abs(monthly.values).max() if monthly.values.size > 0 else 1
norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
im = ax.imshow(monthly.values, cmap=plt.cm.RdYlGn, norm=norm, aspect='auto')

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.set_xticks(range(12))
ax.set_xticklabels(month_names, color='#cccccc')
ax.set_yticks(range(len(monthly)))
ax.set_yticklabels([str(y) for y in monthly.index], color='#cccccc')

for i in range(len(monthly)):
    for j in range(12):
        val = monthly.values[i, j]
        ax.text(j, i, f'${val:,.0f}', ha='center', va='center', fontsize=7,
                color='black' if abs(val) > max_val * 0.3 else '#cccccc')

plt.colorbar(im, ax=ax, label='Monthly P&L ($)')
ax.set_title('Monthly P&L Heatmap', color='#cccccc', fontsize=12)
plt.tight_layout()
plt.show()
"""))

# ── Cell 15: parameter sensitivity ───────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 14. Parameter Sensitivity — Quick Re-run Guide

Edit **Cell 3** to adjust parameters, then re-run cells 3→7→8.

| Config | MSS_EXPIRY | USE_BREAKERS | Expected effect |
|--------|-----------|-------------|----------------|
| Conservative | 150 | False | Fewer trades, lower DD |
| **Default (current)** | **250** | **False** | **Best risk-adjusted** |
| Aggressive | 250 | True | More trades, ~2× DD |
"""))

# ── Build notebook ─────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.9.6"}

output_path = "ICT_Bot.ipynb"
with open(output_path, "w") as f:
    nbformat.write(nb, f)

print(f"Notebook written: {output_path}")
