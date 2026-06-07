"""
ICT structure detection: FVG, Order Blocks, Breaker Blocks, MSS, Kill Zones, OTE.
All functions operate on pandas DataFrames with columns: open, high, low, close, volume.
Index must be a DatetimeTZ-aware index (UTC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([hi - lo, (hi - cl).abs(), (lo - cl).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _swing_highs_lows(df: pd.DataFrame, n: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    Returns (swing_highs, swing_lows) boolean Series.
    A swing high at index i: df.high[i] is the highest in [i-n, i+n].
    Uses a fixed lookback/lookahead of n candles on each side.
    """
    h = df["high"]
    l = df["low"]
    sh = (h == h.rolling(2 * n + 1, center=True).max())
    sl = (l == l.rolling(2 * n + 1, center=True).min())
    return sh, sl


# ---------------------------------------------------------------------------
# Fair Value Gaps
# ---------------------------------------------------------------------------

def detect_fvgs(df: pd.DataFrame, min_gap_atr: float = 0.15) -> pd.DataFrame:
    """
    Bullish FVG: candle[i-2].high < candle[i].low  (gap above i-2, below i)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap below i-2, above i)
    Only keep gaps wider than min_gap_atr * ATR to filter micro-gaps from noise.

    Returns DataFrame with columns:
        type        : 'bull' | 'bear'
        top         : upper boundary of gap
        bottom      : lower boundary of gap
        bar_index   : iloc index of candle i (the candle that completes the gap)
        timestamp   : actual timestamp
        mitigated   : bool (filled after detection, updated by update_fvg_mitigation)
    """
    records = []
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    atr = _atr(df).values

    for i in range(2, len(df)):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        min_size = min_gap_atr * atr[i]
        bull_gap = h[i - 2] < l[i] and (l[i] - h[i - 2]) >= min_size
        bear_gap = l[i - 2] > h[i] and (l[i - 2] - h[i]) >= min_size
        if bull_gap:
            records.append({
                "type": "bull", "top": l[i], "bottom": h[i - 2],
                "bar_index": i, "timestamp": idx[i], "mitigated": False,
            })
        elif bear_gap:
            records.append({
                "type": "bear", "top": l[i - 2], "bottom": h[i],
                "bar_index": i, "timestamp": idx[i], "mitigated": False,
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["type", "top", "bottom", "bar_index", "timestamp", "mitigated"]
    )


def update_fvg_mitigation(fvgs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark an FVG mitigated when a candle CLOSES inside the gap zone.
    Only looks at candles after the FVG formed.
    """
    fvgs = fvgs.copy()
    closes = df["close"].values
    bar_indices = df.index

    for idx_f, row in fvgs.iterrows():
        if row["mitigated"]:
            continue
        start = row["bar_index"] + 1
        for i in range(start, len(closes)):
            c = closes[i]
            if row["bottom"] <= c <= row["top"]:
                fvgs.at[idx_f, "mitigated"] = True
                fvgs.at[idx_f, "mitigated_bar"] = i
                break

    return fvgs


# ---------------------------------------------------------------------------
# Order Blocks
# ---------------------------------------------------------------------------

def detect_order_blocks(df: pd.DataFrame, atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Bullish OB: last bearish candle before a 3-candle bullish displacement.
    Bearish OB: last bullish candle before a 3-candle bearish displacement.
    Displacement: move over 3 candles where total range > atr_multiplier * ATR.
    2.0x ATR threshold filters weak, unreliable displacements.

    Returns DataFrame with columns:
        type, top, bottom, bar_index, timestamp, mitigated, mitigated_bar
    """
    atr = _atr(df).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    idx = df.index
    records = []

    for i in range(5, len(df) - 3):
        # 3-candle displacement window: [i+1, i+2, i+3]
        disp_high = max(h[i + 1], h[i + 2], h[i + 3])
        disp_low = min(l[i + 1], l[i + 2], l[i + 3])
        disp_range = disp_high - disp_low

        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        bullish_displacement = (
            c[i + 1] > o[i + 1] and c[i + 2] > o[i + 2] and c[i + 3] > o[i + 3]
            and disp_range > atr_multiplier * atr[i]
        )
        bearish_displacement = (
            c[i + 1] < o[i + 1] and c[i + 2] < o[i + 2] and c[i + 3] < o[i + 3]
            and disp_range > atr_multiplier * atr[i]
        )

        if bullish_displacement and c[i] < o[i]:  # last bearish candle before bull move
            records.append({
                "type": "bull", "top": max(o[i], c[i]), "bottom": min(o[i], c[i]),
                "bar_index": i, "timestamp": idx[i], "mitigated": False, "mitigated_bar": np.nan,
            })
        elif bearish_displacement and c[i] > o[i]:  # last bullish candle before bear move
            records.append({
                "type": "bear", "top": max(o[i], c[i]), "bottom": min(o[i], c[i]),
                "bar_index": i, "timestamp": idx[i], "mitigated": False, "mitigated_bar": np.nan,
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["type", "top", "bottom", "bar_index", "timestamp", "mitigated", "mitigated_bar"]
    )


def update_ob_mitigation(obs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Mark an OB mitigated when price closes inside its body."""
    obs = obs.copy()
    closes = df["close"].values

    for idx_o, row in obs.iterrows():
        if row["mitigated"]:
            continue
        start = int(row["bar_index"]) + 1
        for i in range(start, len(closes)):
            c = closes[i]
            if row["bottom"] <= c <= row["top"]:
                obs.at[idx_o, "mitigated"] = True
                obs.at[idx_o, "mitigated_bar"] = i
                break

    return obs


# ---------------------------------------------------------------------------
# Breaker Blocks
# ---------------------------------------------------------------------------

def detect_breaker_blocks(obs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    A mitigated OB flips polarity: bull OB -> bearish breaker, bear OB -> bullish breaker.
    Breaker acts as support (bull breaker) or resistance (bear breaker) after price passes through.
    """
    if obs.empty:
        return pd.DataFrame(columns=["type", "top", "bottom", "bar_index", "timestamp", "origin_type"])

    mitigated = obs[obs["mitigated"] == True].copy()
    if mitigated.empty:
        return pd.DataFrame(columns=["type", "top", "bottom", "bar_index", "timestamp", "origin_type"])

    breakers = []
    for _, row in mitigated.iterrows():
        flipped_type = "bear" if row["type"] == "bull" else "bull"
        breakers.append({
            "type": flipped_type,
            "top": row["top"],
            "bottom": row["bottom"],
            "bar_index": row["mitigated_bar"],
            "timestamp": df.index[int(row["mitigated_bar"])] if not np.isnan(row["mitigated_bar"]) else pd.NaT,
            "origin_type": row["type"],
        })

    return pd.DataFrame(breakers)


# ---------------------------------------------------------------------------
# Market Structure Shifts
# ---------------------------------------------------------------------------

def detect_mss(df: pd.DataFrame, swing_n: int = 10) -> pd.DataFrame:
    """
    MSS = break of previous swing high/low with a displacement candle.
    Bullish MSS: close above last swing high with candle body > 1.5 ATR.
    Bearish MSS: close below last swing low with candle body > 1.5 ATR.
    10-candle swings and 1.5x ATR body requirement produce only high-conviction breaks.

    Returns DataFrame: type, bar_index, timestamp, broken_level, displacement_range
    """
    sh, sl = _swing_highs_lows(df, swing_n)
    atr = _atr(df).values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    o = df["open"].values
    idx = df.index

    last_sh_price = np.nan
    last_sl_price = np.nan
    records = []

    for i in range(swing_n * 2 + 1, len(df)):
        # Update last known swing levels from confirmed swings (use i-swing_n to avoid lookahead)
        lookback = max(0, i - swing_n)
        for j in range(lookback, i):
            if sh.iloc[j]:
                last_sh_price = h[j]
            if sl.iloc[j]:
                last_sl_price = l[j]

        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        candle_range = abs(c[i] - o[i])
        is_displacement = candle_range > 1.5 * atr[i]  # body must be 1.5x ATR — strong conviction

        if not np.isnan(last_sh_price) and c[i] > last_sh_price and is_displacement:
            records.append({
                "type": "bull", "bar_index": i, "timestamp": idx[i],
                "broken_level": last_sh_price, "displacement_range": candle_range,
            })
            last_sh_price = np.nan  # reset to avoid duplicate signals until new swing forms

        elif not np.isnan(last_sl_price) and c[i] < last_sl_price and is_displacement:
            records.append({
                "type": "bear", "bar_index": i, "timestamp": idx[i],
                "broken_level": last_sl_price, "displacement_range": candle_range,
            })
            last_sl_price = np.nan

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["type", "bar_index", "timestamp", "broken_level", "displacement_range"]
    )


# ---------------------------------------------------------------------------
# Kill Zones
# ---------------------------------------------------------------------------

def in_kill_zone(ts: pd.Timestamp) -> str | None:
    """
    Returns 'london', 'new_york', or None.
    Kill zones defined in UTC.
    """
    hour = ts.hour
    minute = ts.minute
    total_minutes = hour * 60 + minute

    london_open = (7 * 60, 10 * 60)
    ny_open = (13 * 60 + 30, 16 * 60)

    if london_open[0] <= total_minutes < london_open[1]:
        return "london"
    if ny_open[0] <= total_minutes < ny_open[1]:
        return "new_york"
    return None


def add_kill_zone_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'kill_zone' column to the DataFrame."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["kill_zone"] = df.index.map(in_kill_zone)
    return df


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Standard Wilder ADX.
    Returns a Series of ADX values indexed to df.
    ADX > 20 = trending; ADX < 20 = ranging/choppy.
    """
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]

    up   = hi.diff()
    down = -lo.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr_s = _atr(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    adx.name = "adx"
    return adx


# ---------------------------------------------------------------------------
# OTE Scoring
# ---------------------------------------------------------------------------

def ote_score(price: float, swing_low: float, swing_high: float) -> float:
    """
    Returns a 0-1 score based on how close price is to the OTE zone (61.8-78.6% fib).
    1.0 = perfectly at 70.2% (midpoint of OTE zone), 0 = outside zone but still scored.
    """
    if swing_high == swing_low:
        return 0.0

    fib_618 = swing_high - 0.618 * (swing_high - swing_low)
    fib_786 = swing_high - 0.786 * (swing_high - swing_low)
    ote_low = min(fib_618, fib_786)
    ote_high = max(fib_618, fib_786)
    ote_mid = (ote_low + ote_high) / 2

    distance = abs(price - ote_mid)
    zone_width = ote_high - ote_low
    if zone_width == 0:
        return 0.0

    # Score: 1.0 inside zone center, decays linearly outside
    score = max(0.0, 1.0 - distance / (zone_width * 2))
    return round(score, 4)
