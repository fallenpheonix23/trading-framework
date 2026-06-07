"""
Volatility targeting: rescales strategy weights so the portfolio's
realized vol tracks a target annualized level.

Uses a 21-day rolling realized vol of yesterday's returns → no look-ahead.
Scaling is capped at MAX_SCALE to avoid excessive leverage in quiet markets.
"""
import numpy as np
import pandas as pd

TARGET_VOL = 0.10   # 10% annualized per strategy
VOL_WINDOW = 21     # ~1 month rolling window
MIN_VOL    = 0.02   # floor to prevent division by near-zero vol
MAX_SCALE  = 3.0    # cap: never lever more than 3×


def apply_vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = TARGET_VOL,
    vol_window: int   = VOL_WINDOW,
    max_scale: float  = MAX_SCALE,
) -> pd.DataFrame:
    """
    Scale `weights` so the strategy's ex-ante vol ≈ target_vol.

    Steps:
      1. Estimate daily strategy returns using lagged weights (no look-ahead).
      2. Compute rolling realized vol from those returns.
      3. Shift vol estimate by 1 day (use yesterday's vol to size today).
      4. Scale = target_vol / realized_vol, clipped to [0, max_scale].
    """
    prices_aligned = prices.reindex(weights.index).ffill()
    asset_rets = prices_aligned.pct_change().fillna(0.0)

    # Use previous day's weights to estimate today's P&L (causal)
    port_rets = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1)

    # Rolling vol, then shift 1 day forward so we size on yesterday's estimate
    realized_vol = (
        port_rets.rolling(vol_window, min_periods=5)
                 .std()
                 .clip(lower=MIN_VOL)
                 * np.sqrt(252)
    )
    realized_vol = realized_vol.shift(1).bfill()

    scale = (target_vol / realized_vol).clip(upper=max_scale)

    return weights.multiply(scale, axis=0)
