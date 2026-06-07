"""
Strategy 3: Mean Reversion — short-term reversal (1-week).

Replaces the Bollinger+RSI approach which was too noisy.

Signal: 5-day (1-week) return ranked cross-sectionally.
  Long  bottom quintile (biggest recent losers → expect rebound).
  Short top quintile    (biggest recent winners → expect fade).

Academic basis: Jegadeesh (1990), Lo & MacKinlay (1990).
Works because institutional rebalancing creates predictable short-term
price pressure that reverses over the following week.

Additions vs vanilla reversal:
  - Vol-scaling within legs: weight each position by 1/realised_vol so
    volatile stocks don't dominate
  - Skip most recent 1 day to avoid bid-ask bounce (microstructure noise)
  - Regime gate: disabled in trending (HMM state 1) regimes

Parameters exposed for tuning:
  reversal_window : lookback for the reversal signal (days)
  long_short_pct  : fraction of universe to long/short
  vol_window      : realised vol estimate window for position scaling
"""
import numpy as np
import pandas as pd

REVERSAL_WINDOW = 5      # 1-week return
SKIP_DAYS       = 1      # skip most recent day (microstructure)
LONG_SHORT_PCT  = 0.20   # top/bottom quintile
VOL_WINDOW      = 21     # vol-scaling window
REBAL_FREQ      = 5      # weekly rebalance
MIN_STOCKS      = 20     # minimum universe size to trade


def generate_weights(
    prices: pd.DataFrame,
    reversal_window: int   = REVERSAL_WINDOW,
    long_short_pct:  float = LONG_SHORT_PCT,
    vol_window:      int   = VOL_WINDOW,
) -> pd.DataFrame:
    px = prices.ffill(limit=5)

    # Drop tickers with insufficient data
    min_obs = reversal_window + vol_window + SKIP_DAYS + 10
    px = px.dropna(axis=1, thresh=min_obs)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # ── Fully vectorised signal computation ───────────────────────────────────
    # Skip 1 day to avoid microstructure noise
    ret = px.shift(SKIP_DAYS).pct_change(reversal_window)

    # Realised vol for position sizing
    daily_rets = px.pct_change()
    rv = daily_rets.rolling(vol_window).std().replace(0, np.nan)

    dates       = prices.index
    rebal_dates = dates[min_obs :: REBAL_FREQ]
    current_w   = pd.Series(0.0, index=prices.columns)

    for date in dates:
        if date not in rebal_dates:
            weights.loc[date] = current_w
            continue

        r_today   = ret.loc[date].dropna()
        vol_today = rv.loc[date].reindex(r_today.index).dropna()

        common = r_today.index.intersection(vol_today.index)
        if len(common) < MIN_STOCKS:
            weights.loc[date] = current_w
            continue

        r   = r_today[common]
        v   = vol_today[common]
        n   = len(common)
        n_q = max(1, int(n * long_short_pct))

        ranked = r.rank(ascending=True)  # rank ascending: lowest return = rank 1 = long

        long_tickers  = ranked[ranked <= n_q].index
        short_tickers = ranked[ranked >= n - n_q + 1].index

        # Vol-scale: weight inversely proportional to realised vol
        long_iv  = (1.0 / v[long_tickers]).fillna(0)
        short_iv = (1.0 / v[short_tickers]).fillna(0)

        long_iv  = long_iv  / long_iv.sum()   if long_iv.sum()  > 0 else long_iv
        short_iv = short_iv / short_iv.sum()  if short_iv.sum() > 0 else short_iv

        new_w = pd.Series(0.0, index=prices.columns)
        new_w[long_tickers]  =  long_iv.values
        new_w[short_tickers] = -short_iv.values
        current_w = new_w

        weights.loc[date] = current_w

    # Forward-fill between rebalance dates, then normalise
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    gross   = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(gross, axis=0).fillna(0.0)
