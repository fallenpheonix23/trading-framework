"""
ICT backtest engine.
- $100,000 starting capital, 1% risk per trade
- Unlimited concurrent positions
- Stop = below/above order block
- Target = 2.5R
- 0.05% transaction cost per side
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


INITIAL_CAPITAL = 100_000.0
RISK_PCT = 0.01           # 1% of current equity per trade
REWARD_R = 2.5            # target in R multiples
TRANSACTION_COST = 0.0005 # 0.05% per side
MAX_ZONE_AGE = 100        # bars; zones older than this are stale and skipped
MSS_EXPIRY_BARS = 250     # bars; an MSS older than this is no longer "fresh"
ATR_STOP_BUFFER = 0.5     # multiply ATR × this for stop cushion beyond zone edge
MIN_STOP_PCT = 0.003      # minimum stop distance as fraction of price (0.3%)


@dataclass
class Trade:
    trade_id: int
    direction: str          # 'long' | 'short'
    entry_bar: int
    entry_price: float
    stop_price: float
    target_price: float
    position_size: float    # shares
    risk_dollar: float
    entry_time: pd.Timestamp
    kill_zone: str
    signal_source: str      # 'fvg_bull', 'fvg_bear', 'ob_bull', 'ob_bear', 'bb_bull', 'bb_bear'
    ote_score: float = 0.0

    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None  # 'target' | 'stop' | 'end_of_data'
    pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    open_equity_peak: float = 0.0


def _entry_signal_generator(
    df: pd.DataFrame,
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    breakers: pd.DataFrame,
    mss: pd.DataFrame,
    daily_bias: pd.Series,
) -> pd.DataFrame:
    """
    Produces a DataFrame of entry signals.

    Key rules applied here:
    - Daily 50/200 EMA bias must align with trade direction
    - MSS must have fired in the last MSS_EXPIRY_BARS bars (not just ever)
    - Zone must be < MAX_ZONE_AGE bars old
    - Stop = zone edge - ATR_STOP_BUFFER * ATR, minimum MIN_STOP_PCT of price
    - One signal per zone type per bar (closest fresh zone wins)
    """
    from ict_structures import _swing_highs_lows, _atr, in_kill_zone, ote_score

    signals = []
    closes = df["close"].values
    idx = df.index
    atr_vals = _atr(df).values

    # MSS recency: for each bar, track how many bars ago each direction last fired
    # mss_last[direction] = bar index of most recent MSS of that type
    mss_last: dict[str, int] = {"bull": -9999, "bear": -9999}
    mss_fire_map: dict[int, str] = {}
    for _, row in mss.iterrows():
        mss_fire_map[int(row["bar_index"])] = row["type"]

    sh, sl = _swing_highs_lows(df)
    swing_lows = df["low"].where(sl).ffill().values
    swing_highs = df["high"].where(sh).ffill().values

    def get_daily_bias(bar_ts: pd.Timestamp) -> Optional[str]:
        date = bar_ts.date()
        return daily_bias.get(date, None) if hasattr(daily_bias, "get") else (
            daily_bias.loc[date] if date in daily_bias.index else None
        )

    def check_zone_entry(bar_i: int, zones_df: pd.DataFrame, direction: str, source_prefix: str):
        if zones_df.empty:
            return
        price = closes[bar_i]
        ts = idx[bar_i]
        atr_i = atr_vals[bar_i]

        kz = in_kill_zone(ts)
        if kz is None:
            return

        # Daily bias gate
        bias = get_daily_bias(ts)
        if bias is not None and bias != direction:
            return

        # MSS recency gate — must have had an MSS of this direction within last MSS_EXPIRY_BARS
        bars_since_mss = bar_i - mss_last[direction]
        if bars_since_mss > MSS_EXPIRY_BARS:
            return

        # Build zone filter
        base_filter = (
            (zones_df["bar_index"] < bar_i) &
            (zones_df["bar_index"] >= bar_i - MAX_ZONE_AGE) &  # staleness gate
            (zones_df["type"] == direction[:4])
        )
        if "mitigated" in zones_df.columns:
            valid_zone = (
                (zones_df["mitigated"] == False) |
                (zones_df.get("mitigated_bar", pd.Series(np.nan, index=zones_df.index)) == bar_i)
            )
            base_filter = base_filter & valid_zone
        active_zones = zones_df[base_filter]
        if active_zones.empty:
            return

        for _, zone in active_zones.iterrows():
            if not (zone["bottom"] <= price <= zone["top"]):
                continue

            # ATR-based stop with minimum floor
            if np.isnan(atr_i) or atr_i == 0:
                continue
            buffer = ATR_STOP_BUFFER * atr_i
            min_buffer = MIN_STOP_PCT * price
            actual_buffer = max(buffer, min_buffer)

            if direction == "bull":
                stop = zone["bottom"] - actual_buffer
            else:
                stop = zone["top"] + actual_buffer

            risk_per_share = abs(price - stop)
            if risk_per_share <= 0:
                continue

            sl_v = swing_lows[bar_i] if not np.isnan(swing_lows[bar_i]) else price * 0.98
            sh_v = swing_highs[bar_i] if not np.isnan(swing_highs[bar_i]) else price * 1.02
            ote = ote_score(price, sl_v, sh_v)

            signals.append({
                "bar_index": bar_i,
                "direction": "long" if direction == "bull" else "short",
                "entry_price": price,
                "stop_price": stop,
                "entry_zone_top": zone["top"],
                "entry_zone_bottom": zone["bottom"],
                "signal_source": f"{source_prefix}_{direction[:4]}",
                "ote_score": ote,
                "kill_zone": kz,
                "timestamp": ts,
            })
            break  # nearest fresh zone wins

    for i in range(30, len(df)):
        # Update MSS recency tracker before processing this bar
        if i in mss_fire_map:
            mss_last[mss_fire_map[i]] = i

        for direction in ("bull", "bear"):
            check_zone_entry(i, fvgs, direction, "fvg")
            check_zone_entry(i, obs, direction, "ob")
            if not breakers.empty:
                check_zone_entry(i, breakers, direction, "bb")

    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["bar_index", "direction", "entry_price", "stop_price",
                 "entry_zone_top", "entry_zone_bottom", "signal_source",
                 "ote_score", "kill_zone", "timestamp"]
    )


def run_backtest(
    df: pd.DataFrame,
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    breakers: pd.DataFrame,
    mss: pd.DataFrame,
    daily_bias: pd.Series,
) -> tuple[list[Trade], pd.Series]:
    """
    Main backtest loop.
    Returns (trades_list, equity_curve_series).
    """
    signals = _entry_signal_generator(df, fvgs, obs, breakers, mss, daily_bias)
    if signals.empty:
        return [], pd.Series([INITIAL_CAPITAL], name="equity")

    signals = signals.sort_values("bar_index").reset_index(drop=True)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index

    equity = INITIAL_CAPITAL
    equity_curve = np.full(len(df), np.nan)
    equity_curve[0] = equity

    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []
    trade_id = 0

    # Index signals by bar for fast lookup
    signal_map: dict[int, list] = {}
    for _, sig in signals.iterrows():
        bi = int(sig["bar_index"])
        signal_map.setdefault(bi, []).append(sig)

    for bar_i in range(1, len(df)):
        high_i = highs[bar_i]
        low_i = lows[bar_i]
        close_i = closes[bar_i]

        # --- Check exits for open trades ---
        still_open = []
        for trade in open_trades:
            hit_stop = (trade.direction == "long" and low_i <= trade.stop_price) or \
                       (trade.direction == "short" and high_i >= trade.stop_price)
            hit_target = (trade.direction == "long" and high_i >= trade.target_price) or \
                         (trade.direction == "short" and low_i <= trade.target_price)

            if hit_stop or hit_target:
                if hit_target and not hit_stop:
                    exit_p = trade.target_price
                    reason = "target"
                elif hit_stop and not hit_target:
                    exit_p = trade.stop_price
                    reason = "stop"
                else:
                    # Both hit same bar — conservative: assume stop hit first (worst case)
                    exit_p = trade.stop_price
                    reason = "stop"

                cost = exit_p * trade.position_size * TRANSACTION_COST
                if trade.direction == "long":
                    raw_pnl = (exit_p - trade.entry_price) * trade.position_size
                else:
                    raw_pnl = (trade.entry_price - exit_p) * trade.position_size

                pnl = raw_pnl - cost
                trade.exit_bar = bar_i
                trade.exit_price = exit_p
                trade.exit_time = idx[bar_i]
                trade.exit_reason = reason
                trade.pnl = pnl
                trade.r_multiple = (exit_p - trade.entry_price) / abs(trade.entry_price - trade.stop_price) \
                    if trade.direction == "long" else \
                    (trade.entry_price - exit_p) / abs(trade.entry_price - trade.stop_price)

                equity += pnl
                closed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # --- Open new trades from signals at this bar ---
        if bar_i in signal_map:
            for sig in signal_map[bar_i]:
                entry_p = float(sig["entry_price"])
                stop_p = float(sig["stop_price"])
                risk_per_share = abs(entry_p - stop_p)

                if risk_per_share <= 0:
                    continue

                risk_dollar = equity * RISK_PCT
                pos_size = risk_dollar / risk_per_share

                if sig["direction"] == "long":
                    target_p = entry_p + REWARD_R * risk_per_share
                else:
                    target_p = entry_p - REWARD_R * risk_per_share

                entry_cost = entry_p * pos_size * TRANSACTION_COST
                equity -= entry_cost  # pay entry commission immediately

                trade_id += 1
                t = Trade(
                    trade_id=trade_id,
                    direction=sig["direction"],
                    entry_bar=bar_i,
                    entry_price=entry_p,
                    stop_price=stop_p,
                    target_price=target_p,
                    position_size=pos_size,
                    risk_dollar=risk_dollar,
                    entry_time=idx[bar_i],
                    kill_zone=str(sig["kill_zone"]),
                    signal_source=str(sig["signal_source"]),
                    ote_score=float(sig["ote_score"]),
                )
                open_trades.append(t)

        equity_curve[bar_i] = equity

    # Close any remaining open trades at last close
    for trade in open_trades:
        exit_p = closes[-1]
        cost = exit_p * trade.position_size * TRANSACTION_COST
        if trade.direction == "long":
            raw_pnl = (exit_p - trade.entry_price) * trade.position_size
        else:
            raw_pnl = (trade.entry_price - exit_p) * trade.position_size
        pnl = raw_pnl - cost
        trade.exit_bar = len(df) - 1
        trade.exit_price = exit_p
        trade.exit_time = idx[-1]
        trade.exit_reason = "end_of_data"
        trade.pnl = pnl
        trade.r_multiple = (exit_p - trade.entry_price) / abs(trade.entry_price - trade.stop_price) \
            if trade.direction == "long" else \
            (trade.entry_price - exit_p) / abs(trade.entry_price - trade.stop_price)
        equity += pnl
        closed_trades.append(trade)

    # Forward-fill NaN in equity curve
    eq_series = pd.Series(equity_curve, index=df.index)
    eq_series = eq_series.ffill().fillna(INITIAL_CAPITAL)

    return closed_trades, eq_series


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade], equity_curve: pd.Series) -> dict:
    if not trades:
        return {"error": "No trades executed"}

    df_t = pd.DataFrame([{
        "direction": t.direction,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "pnl": t.pnl,
        "r_multiple": t.r_multiple,
        "kill_zone": t.kill_zone,
        "signal_source": t.signal_source,
        "ote_score": t.ote_score,
    } for t in trades])

    total_trades = len(df_t)
    winners = df_t[df_t["pnl"] > 0]
    losers = df_t[df_t["pnl"] <= 0]
    win_rate = len(winners) / total_trades if total_trades > 0 else 0

    avg_win = winners["pnl"].mean() if len(winners) else 0
    avg_loss = losers["pnl"].mean() if len(losers) else 0
    avg_r = df_t["r_multiple"].mean()

    total_pnl = df_t["pnl"].sum()
    final_equity = equity_curve.iloc[-1]

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    # Sharpe (annualized, assuming 252 trading days * 6.5 hours = 1638 hourly bars/year)
    returns = equity_curve.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(1638)) if returns.std() > 0 else 0

    # Profit factor
    gross_profit = winners["pnl"].sum() if len(winners) else 0
    gross_loss = abs(losers["pnl"].sum()) if len(losers) else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Kill zone breakdown
    kz_stats = df_t.groupby("kill_zone").agg(
        trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
        total_pnl=("pnl", "sum"),
    ).round(4)

    # Signal source breakdown
    source_stats = df_t.groupby("signal_source").agg(
        trades=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_r=("r_multiple", "mean"),
        total_pnl=("pnl", "sum"),
    ).round(4)

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 4),
        "avg_r_multiple": round(avg_r, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "trade_log": df_t,
        "kill_zone_stats": kz_stats,
        "source_stats": source_stats,
    }
