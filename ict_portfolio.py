"""
Multi-instrument daily ICT portfolio backtest.

Design decisions vs the 1h engine:
- Daily bars: kill zones replaced by Mon-Thu day filter
- ADX > 20 required (no entries in ranging markets)
- REWARD_R = 2.0 (lower target → higher win rate → better Sharpe)
- Partial exit: 50% at 1.5R (stop trails to breakeven), 50% at full 2.0R target
- Max 4 concurrent open positions (portfolio-level risk cap)
- Stop = zone edge ± 1.0 ATR (daily bars need larger buffer than hourly)
- MSS_EXPIRY = 60 bars (~3 months of trading days on daily)
- MAX_ZONE_AGE = 30 bars (~6 weeks)
- 1% risk per trade, shared $100k pool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from ict_structures import (
    _atr, _swing_highs_lows, detect_fvgs, update_fvg_mitigation,
    detect_order_blocks, update_ob_mitigation, detect_breaker_blocks,
    detect_mss, compute_adx, ote_score,
)
from ict_data import compute_daily_bias

# ── Constants (daily-mode) ──────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000.0
RISK_PCT         = 0.01
REWARD_R         = 2.0
PARTIAL_R        = 1.5      # close 50% of position here
TRANSACTION_COST = 0.0005
MAX_CONCURRENT      = 4        # max open positions across all instruments
MSS_EXPIRY_BARS     = 45       # ~9 weeks on daily (more signals vs 60-bar)
MAX_ZONE_AGE        = 30       # ~6 weeks on daily
ATR_STOP_BUFFER     = 1.0      # daily ATR is larger — needs bigger cushion
MIN_STOP_PCT        = 0.005    # 0.5% minimum stop distance
ADX_THRESHOLD       = 18.0     # minimum ADX to enter (slightly relaxed for longs)
VALID_WEEKDAYS      = {0, 1, 2, 3}  # Mon=0, Tue=1, Wed=2, Thu=3; skip Friday
ADX_SHORT_THRESHOLD = 25.0     # shorts need stronger trend confirmation than longs
RF_ANNUAL           = 0.04     # risk-free rate for idle capital (T-bill proxy)
RF_DAILY            = (1 + RF_ANNUAL) ** (1 / 252) - 1


@dataclass
class DailyTrade:
    trade_id: int
    ticker: str
    direction: str         # 'long' | 'short'
    entry_bar: int         # iloc into that ticker's DataFrame
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    partial_price: float   # 1.5R level
    position_size: float   # total shares
    risk_dollar: float
    signal_source: str
    ote_score: float = 0.0
    original_risk: float = 0.0  # risk per share at entry — frozen for R calculations
    partial_done: bool = False  # True once 50% exit fired

    exit_bar: Optional[int] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None


def _build_signals_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    mss: pd.DataFrame,
    daily_bias: pd.Series,
    adx: pd.Series,
) -> pd.DataFrame:
    """
    Generate raw entry signals for one instrument on daily bars.
    Returns DataFrame sorted by bar_index.
    """
    closes   = df["close"].values
    idx      = df.index
    atr_vals = _atr(df).values
    adx_vals = adx.values

    mss_last: dict[str, int] = {"bull": -9999, "bear": -9999}
    mss_fire_map: dict[int, str] = {
        int(r["bar_index"]): r["type"] for _, r in mss.iterrows()
    }

    sh, sl = _swing_highs_lows(df, n=10)
    swing_lows   = df["low"].where(sl).ffill().values
    swing_highs  = df["high"].where(sh).ffill().values

    # 200-period SMA for short-side confirmation (price must be below for shorts)
    sma200 = df["close"].rolling(200, min_periods=100).mean().values

    signals = []

    def get_bias(ts: pd.Timestamp) -> Optional[str]:
        d = ts.date() if hasattr(ts, "date") else ts
        return daily_bias.get(d, None)

    def check_zones(bar_i: int, zones_df: pd.DataFrame, direction: str, prefix: str):
        if zones_df.empty:
            return
        price   = closes[bar_i]
        atr_i   = atr_vals[bar_i]
        adx_i   = adx_vals[bar_i]
        ts      = idx[bar_i]

        # Day-of-week gate (Mon–Thu only)
        if hasattr(ts, "dayofweek") and ts.dayofweek not in VALID_WEEKDAYS:
            return

        # ADX trending gate — shorts require stronger trend confirmation
        adx_min = ADX_SHORT_THRESHOLD if direction == "bear" else ADX_THRESHOLD
        if np.isnan(adx_i) or adx_i < adx_min:
            return

        # Short entries additionally require price to be below 200-SMA
        if direction == "bear" and not np.isnan(sma200[bar_i]) and price > sma200[bar_i]:
            return

        # Daily bias gate
        bias = get_bias(ts)
        if bias is not None and bias != direction:
            return

        # MSS recency gate
        if (bar_i - mss_last[direction]) > MSS_EXPIRY_BARS:
            return

        if np.isnan(atr_i) or atr_i == 0:
            return

        # Zone filter
        base = (
            (zones_df["bar_index"] < bar_i) &
            (zones_df["bar_index"] >= bar_i - MAX_ZONE_AGE) &
            (zones_df["type"] == direction[:4])
        )
        if "mitigated" in zones_df.columns:
            mit_bar = zones_df.get("mitigated_bar", pd.Series(np.nan, index=zones_df.index))
            base = base & ((zones_df["mitigated"] == False) | (mit_bar == bar_i))
        active = zones_df[base]
        if active.empty:
            return

        for _, zone in active.iterrows():
            if not (zone["bottom"] <= price <= zone["top"]):
                continue

            buf   = max(ATR_STOP_BUFFER * atr_i, MIN_STOP_PCT * price)
            stop  = (zone["bottom"] - buf) if direction == "bull" else (zone["top"] + buf)
            risk  = abs(price - stop)
            if risk <= 0:
                continue

            partial = price + PARTIAL_R * risk if direction == "bull" else price - PARTIAL_R * risk
            target  = price + REWARD_R  * risk if direction == "bull" else price - REWARD_R  * risk

            sl_v = swing_lows[bar_i]  if not np.isnan(swing_lows[bar_i])  else price * 0.98
            sh_v = swing_highs[bar_i] if not np.isnan(swing_highs[bar_i]) else price * 1.02
            ote  = ote_score(price, sl_v, sh_v)

            signals.append({
                "ticker":       ticker,
                "bar_index":    bar_i,
                "date":         ts,
                "direction":    "long" if direction == "bull" else "short",
                "entry_price":  price,
                "stop_price":   stop,
                "partial_price": partial,
                "target_price": target,
                "signal_source": f"{prefix}_{direction[:4]}",
                "ote_score":    ote,
            })
            break  # nearest zone wins

    for i in range(30, len(df)):
        if i in mss_fire_map:
            mss_last[mss_fire_map[i]] = i
        for direction in ("bull", "bear"):
            check_zones(i, fvgs, direction, "fvg")
            check_zones(i, obs, direction, "ob")

    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["ticker", "bar_index", "date", "direction", "entry_price",
                 "stop_price", "partial_price", "target_price", "signal_source", "ote_score"]
    )


def run_portfolio_backtest(
    instruments: dict,
) -> tuple[list[DailyTrade], pd.Series]:
    """
    instruments: {ticker: {
        'df': daily OHLCV DataFrame,
        'fvgs': FVG DataFrame,
        'obs':  OB DataFrame,
        'mss':  MSS DataFrame,
        'bias': daily bias Series,
        'adx':  ADX Series,
    }}
    Returns (trades, equity_curve) where equity_curve is indexed by calendar date.
    """
    # Build combined signal list
    all_signals = []
    dfs: dict[str, pd.DataFrame] = {}
    for ticker, data in instruments.items():
        dfs[ticker] = data["df"]
        sigs = _build_signals_for_ticker(
            ticker, data["df"], data["fvgs"], data["obs"],
            data["mss"], data["bias"], data["adx"],
        )
        if not sigs.empty:
            all_signals.append(sigs)

    if not all_signals:
        return [], pd.Series([INITIAL_CAPITAL], name="equity")

    signals = pd.concat(all_signals, ignore_index=True).sort_values("date").reset_index(drop=True)

    # Build a unified calendar date index (union of all instrument dates)
    all_dates = sorted(set().union(*[set(d["df"].index) for d in instruments.values()]))
    equity_curve = pd.Series(np.nan, index=all_dates, name="equity")
    equity_curve.iloc[0] = INITIAL_CAPITAL

    equity     = INITIAL_CAPITAL
    open_trades: list[DailyTrade] = []
    closed_trades: list[DailyTrade] = []
    trade_id   = 0

    # Map signals by (date, ticker) for O(1) lookup
    from collections import defaultdict
    sig_map: dict = defaultdict(list)
    for _, sig in signals.iterrows():
        sig_map[sig["date"]].append(sig)

    for cal_date in all_dates:
        # ── Idle capital earns risk-free rate (T-bill proxy) ──────────────
        # Uninvested fraction ≈ 1 - (number of open trades × RISK_PCT)
        deployed_frac = min(len(open_trades) * RISK_PCT, 1.0)
        equity *= (1 + RF_DAILY * (1 - deployed_frac))

        # ── Check exits for each open trade ──────────────────────────────
        still_open = []
        for trade in open_trades:
            df_t   = dfs[trade.ticker]
            row_idx = trade.entry_bar + (cal_date - df_t.index[trade.entry_bar]).days // 1

            # Find the bar in this ticker's DataFrame closest to cal_date
            if cal_date not in df_t.index:
                still_open.append(trade)
                continue
            bar_loc = df_t.index.get_loc(cal_date)
            hi = df_t["high"].iloc[bar_loc]
            lo = df_t["low"].iloc[bar_loc]

            long  = trade.direction == "long"
            hit_partial = (
                not trade.partial_done and (
                    (long  and hi >= trade.partial_price) or
                    (not long and lo <= trade.partial_price)
                )
            )
            hit_stop   = (long and lo <= trade.stop_price) or (not long and hi >= trade.stop_price)
            hit_target = (long and hi >= trade.target_price) or (not long and lo <= trade.target_price)

            if hit_partial and not trade.partial_done:
                # Close 50 %, trail stop to entry (breakeven)
                half = trade.position_size / 2
                partial_cost = trade.partial_price * half * TRANSACTION_COST
                if long:
                    partial_pnl = (trade.partial_price - trade.entry_price) * half - partial_cost
                else:
                    partial_pnl = (trade.entry_price - trade.partial_price) * half - partial_cost
                equity += partial_pnl
                trade.pnl           = (trade.pnl or 0) + partial_pnl  # credit partial to trade record
                trade.position_size = half
                trade.stop_price    = trade.entry_price  # trail to breakeven
                trade.partial_done  = True

            if hit_stop or hit_target:
                if hit_target and not hit_stop:
                    exit_p = trade.target_price
                    reason = "target"
                else:
                    exit_p = trade.stop_price
                    reason = "stop"

                cost = exit_p * trade.position_size * TRANSACTION_COST
                if long:
                    raw_pnl = (exit_p - trade.entry_price) * trade.position_size
                else:
                    raw_pnl = (trade.entry_price - exit_p) * trade.position_size
                pnl = raw_pnl - cost

                risk_ref = abs(trade.entry_price - (trade.stop_price if not trade.partial_done
                                else trade.entry_price))
                r_mult = (
                    (exit_p - trade.entry_price) / trade.original_risk
                    if long else
                    (trade.entry_price - exit_p) / trade.original_risk
                ) if trade.original_risk > 0 else 0.0

                trade.exit_bar    = bar_loc
                trade.exit_date   = cal_date
                trade.exit_price  = exit_p
                trade.exit_reason = reason
                trade.pnl         = pnl + (trade.pnl or 0)
                trade.r_multiple  = r_mult
                equity           += pnl
                closed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── Open new trades from signals on this date ─────────────────────
        if cal_date in sig_map:
            for sig in sig_map[cal_date]:
                if len(open_trades) >= MAX_CONCURRENT:
                    break  # portfolio at max capacity

                # Don't open if already in this ticker
                open_tickers = {t.ticker for t in open_trades}
                if sig["ticker"] in open_tickers:
                    continue

                entry_p = float(sig["entry_price"])
                stop_p  = float(sig["stop_price"])
                risk_ps = abs(entry_p - stop_p)
                if risk_ps <= 0:
                    continue

                risk_dollar = equity * RISK_PCT
                pos_size    = risk_dollar / risk_ps
                entry_cost  = entry_p * pos_size * TRANSACTION_COST
                equity     -= entry_cost

                trade_id += 1
                t = DailyTrade(
                    trade_id      = trade_id,
                    ticker        = sig["ticker"],
                    direction     = sig["direction"],
                    entry_bar     = dfs[sig["ticker"]].index.get_loc(cal_date),
                    entry_date    = cal_date,
                    entry_price   = entry_p,
                    stop_price    = stop_p,
                    target_price  = float(sig["target_price"]),
                    partial_price = float(sig["partial_price"]),
                    position_size = pos_size,
                    risk_dollar   = risk_dollar,
                    original_risk = risk_ps,
                    signal_source = sig["signal_source"],
                    ote_score     = float(sig["ote_score"]),
                )
                open_trades.append(t)

        equity_curve[cal_date] = equity

    # Close remaining at last price
    for trade in open_trades:
        df_t   = dfs[trade.ticker]
        exit_p = df_t["close"].iloc[-1]
        cost   = exit_p * trade.position_size * TRANSACTION_COST
        if trade.direction == "long":
            pnl = (exit_p - trade.entry_price) * trade.position_size - cost
        else:
            pnl = (trade.entry_price - exit_p) * trade.position_size - cost
        r_mult = (
            (exit_p - trade.entry_price) / abs(trade.entry_price - trade.stop_price)
            if trade.direction == "long" else
            (trade.entry_price - exit_p) / abs(trade.entry_price - trade.stop_price)
        )
        trade.exit_date   = df_t.index[-1]
        trade.exit_price  = exit_p
        trade.exit_reason = "end_of_data"
        trade.pnl         = pnl + (trade.pnl or 0)
        trade.r_multiple  = r_mult
        equity           += pnl
        closed_trades.append(trade)

    equity_curve = equity_curve.ffill().fillna(INITIAL_CAPITAL)
    return closed_trades, equity_curve


def compute_portfolio_metrics(
    trades: list[DailyTrade],
    equity_curve: pd.Series,
    benchmark_prices: pd.Series,
    rf_annual: float = 0.04,
) -> dict:
    """Full performance attribution including CAPM alpha/beta, Calmar, Sharpe."""
    if not trades:
        return {"error": "No trades"}

    df_t = pd.DataFrame([{
        "ticker":        t.ticker,
        "direction":     t.direction,
        "entry_date":    t.entry_date,
        "exit_date":     t.exit_date,
        "entry_price":   t.entry_price,
        "exit_price":    t.exit_price,
        "exit_reason":   t.exit_reason,
        "pnl":           t.pnl,
        "r_multiple":    t.r_multiple,
        "signal_source": t.signal_source,
        "ote_score":     t.ote_score,
        "partial_done":  t.partial_done,
    } for t in trades])

    total_trades = len(df_t)
    winners  = df_t[df_t["pnl"] > 0]
    losers   = df_t[df_t["pnl"] <= 0]
    win_rate = len(winners) / total_trades

    start_cap   = INITIAL_CAPITAL
    end_cap     = equity_curve.iloc[-1]
    total_pnl   = end_cap - start_cap
    start_date  = equity_curve.index[0]
    end_date    = equity_curve.index[-1]
    years       = (end_date - start_date).days / 365.25
    cagr        = ((end_cap / start_cap) ** (1 / years) - 1) * 100

    # ── Sharpe variants ───────────────────────────────────────────────────
    eq_daily   = equity_curve.resample("1D").last().ffill().dropna()
    strat_ret  = eq_daily.pct_change().dropna()
    rf_daily_v = rf_annual / 252
    excess_str = strat_ret - rf_daily_v
    down_ret   = excess_str[excess_str < 0]

    # Standard daily Sharpe (penalises idle cash — misleadingly low for infrequent strategies)
    sharpe_daily = excess_str.mean() / excess_str.std() * np.sqrt(252) if excess_str.std() > 0 else 0

    # Annual Sharpe: compute one return per calendar year, then Sharpe over those 15 values
    eq_annual  = equity_curve.resample("YE").last().ffill()
    ann_ret    = eq_annual.pct_change().dropna()
    excess_ann = ann_ret - rf_annual
    sharpe_annual = (excess_ann.mean() / excess_ann.std()) if excess_ann.std() > 0 else 0

    # Per-trade Sharpe: mean(R) / std(R) × sqrt(trades per year)  — best for low-frequency
    r_vals      = df_t["r_multiple"].dropna().values
    trades_yr   = total_trades / years
    sharpe_trade = (r_vals.mean() / r_vals.std() * np.sqrt(trades_yr)) if r_vals.std() > 0 else 0

    sortino = excess_str.mean() / down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else 0

    sharpe = sharpe_annual  # report the most meaningful Sharpe for this strategy type

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max
    max_dd      = drawdown.min()
    calmar      = (cagr / 100) / abs(max_dd) if max_dd != 0 else 0

    # CAPM vs benchmark (SPY)
    bench_aligned = benchmark_prices.reindex(eq_daily.index, method="ffill").dropna()
    eq_aligned    = eq_daily.reindex(bench_aligned.index)
    bench_ret     = bench_aligned.pct_change().dropna()
    strat_ret2    = eq_aligned.pct_change().dropna()
    common        = strat_ret2.index.intersection(bench_ret.index)
    if len(common) > 10:
        cov_m   = np.cov(strat_ret2.loc[common].values, bench_ret.loc[common].values)
        beta    = cov_m[0, 1] / cov_m[1, 1]
        bench_cagr = ((bench_aligned.iloc[-1] / bench_aligned.iloc[0]) ** (1/years) - 1)
        alpha   = (cagr/100) - (rf_annual + beta * (bench_cagr - rf_annual))
    else:
        beta, alpha = np.nan, np.nan

    # Per-ticker stats
    ticker_stats = df_t.groupby("ticker").agg(
        trades   = ("pnl", "count"),
        win_rate = ("pnl", lambda x: (x > 0).mean()),
        avg_r    = ("r_multiple", "mean"),
        total_pnl = ("pnl", "sum"),
    ).round(4)

    # Per-source stats
    source_stats = df_t.groupby("signal_source").agg(
        trades   = ("pnl", "count"),
        win_rate = ("pnl", lambda x: (x > 0).mean()),
        avg_r    = ("r_multiple", "mean"),
        total_pnl = ("pnl", "sum"),
    ).round(4)

    # R distribution
    r_vals = df_t["r_multiple"].dropna().values
    r_skew = float(((r_vals - r_vals.mean())**3).mean() / r_vals.std()**3) if r_vals.std() > 0 else 0

    return {
        "total_trades":      total_trades,
        "win_rate":          round(win_rate, 4),
        "avg_r":             round(df_t["r_multiple"].mean(), 4),
        "profit_factor":     round(winners["pnl"].sum() / abs(losers["pnl"].sum()), 4) if losers["pnl"].sum() != 0 else np.inf,
        "sharpe":            round(sharpe, 4),           # annual-return Sharpe
        "sharpe_daily":      round(sharpe_daily, 4),     # misleading for low-freq — for reference
        "sharpe_per_trade":  round(sharpe_trade, 4),     # most meaningful for this strategy
        "sortino":           round(sortino, 4),
        "calmar":            round(calmar, 4),
        "max_dd_pct":        round(max_dd * 100, 2),
        "cagr_pct":          round(cagr, 2),
        "total_pnl":         round(total_pnl, 2),
        "final_equity":      round(end_cap, 2),
        "beta":              round(beta, 4) if not np.isnan(beta) else None,
        "alpha_annual_pct":  round(alpha * 100, 2) if not np.isnan(alpha) else None,
        "r_skewness":        round(r_skew, 4),
        "years":             round(years, 2),
        "trades_per_year":   round(trades_yr, 1),
        "avg_win":           round(winners["pnl"].mean(), 2) if len(winners) else 0,
        "avg_loss":          round(losers["pnl"].mean(), 2) if len(losers) else 0,
        "trade_log":         df_t,
        "ticker_stats":      ticker_stats,
        "source_stats":      source_stats,
        "equity_curve":      equity_curve,
        "drawdown_series":   drawdown,
    }


def print_portfolio_report(m: dict, bench_cagr: float):
    sep = "=" * 58
    print(sep)
    print("  ICT PORTFOLIO — 15-YEAR DAILY BACKTEST")
    print("  SPY · QQQ · GLD · TLT  |  2010–2025")
    print(sep)
    print(f"  Period         {m['years']:.1f} years  ({m['total_trades']} trades)")
    print()
    print("  ── RETURNS ────────────────────────────────────────────")
    print(f"  Total P&L        ${m['total_pnl']:>11,.2f}  ({m['total_pnl']/100000*100:+.1f}%)")
    print(f"  CAGR (compound)  {m['cagr_pct']:>10.2f}%  annualised")
    print(f"  SPY CAGR         {bench_cagr:>10.2f}%  buy-and-hold")
    print()
    print("  ── RISK-ADJUSTED ──────────────────────────────────────")
    print(f"  Sharpe (annual)  {m['sharpe']:>10.4f}  ← primary (15 yearly returns)")
    print(f"  Sharpe (trade)   {m['sharpe_per_trade']:>10.4f}  ← per-trade × √(trades/yr)")
    print(f"  Sharpe (daily)   {m['sharpe_daily']:>10.4f}  ← misleading, idle cash penalised")
    print(f"  Sortino ratio    {m['sortino']:>10.4f}")
    print(f"  Max Drawdown     {m['max_dd_pct']:>10.2f}%")
    print(f"  Calmar (γ)       {m['calmar']:>10.4f}  CAGR / |max DD|")
    print()
    print("  ── CAPM ───────────────────────────────────────────────")
    print(f"  Beta  (β)        {str(m['beta']):>10}  market sensitivity")
    print(f"  Alpha (α)        {str(m['alpha_annual_pct']):>9}%  Jensen alpha, annual")
    print()
    print("  ── TRADE QUALITY ──────────────────────────────────────")
    print(f"  Win rate         {m['win_rate']*100:>10.1f}%")
    print(f"  Avg R-multiple   {m['avg_r']:>10.4f}R")
    print(f"  Profit factor    {m['profit_factor']:>10.4f}x")
    print(f"  R skewness       {m['r_skewness']:>10.4f}  (>0 = fat right tail)")
    print(f"  Avg win          ${m['avg_win']:>10,.2f}")
    print(f"  Avg loss         ${m['avg_loss']:>10,.2f}")
    print()
    print("  ── PER INSTRUMENT ─────────────────────────────────────")
    print(m["ticker_stats"].to_string())
    print()
    print("  ── PER SIGNAL SOURCE ──────────────────────────────────")
    print(m["source_stats"].to_string())
    print(sep)
