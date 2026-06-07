"""
Strategy 2: Trend Following — enhanced.

Two complementary signals combined:
  1. Cross-sectional momentum (CS): vol-adjusted 12-1 month return ranked
     across the universe.  Dividing raw return by realised vol removes the
     'size × vol' bias and is the standard TSMOM/CSMOM adjustment.
  2. Time-series momentum (TS): 1 if stock is above its 200-day MA, -1 if below.
     Gives an absolute direction signal independent of peer ranking.

Final score = (1 - ts_weight) * CS_rank + ts_weight * TS_direction.
Long top decile, short bottom decile of final score.
Regime filter: disabled in choppy (HMM state 0) markets.
"""
import numpy as np
import pandas as pd

FORMATION_MONTHS = 12
SKIP_MONTHS      = 1
REBAL_FREQ       = 21      # monthly rebalance
TOP_DECILE       = 0.10
BOT_DECILE       = 0.10
MA_WINDOW        = 200     # 200-day MA for time-series signal
TS_WEIGHT        = 0.30    # blend weight for TS vs CS signal


def _cs_momentum_score(prices: pd.DataFrame, end_idx: int) -> pd.Series:
    """Vol-adjusted 12-1 month cross-sectional return."""
    skip  = SKIP_MONTHS * 21
    start = end_idx - FORMATION_MONTHS * 21
    if start < 0:
        return pd.Series(dtype=float)

    ret = prices.iloc[end_idx - skip] / prices.iloc[start] - 1

    # Realised daily vol over the formation window
    window_rets = prices.iloc[start : end_idx - skip].pct_change()
    vol = window_rets.std() * np.sqrt(252)
    vol = vol.replace(0, np.nan)

    return (ret / vol).dropna()


def _ts_signal(prices: pd.DataFrame, end_idx: int, ma_window: int) -> pd.Series:
    """
    Time-series signal: +1 if current price > rolling MA, -1 otherwise.
    Normalised to [-1, +1] so it can be blended directly with CS ranks.
    """
    if end_idx < ma_window:
        return pd.Series(dtype=float)
    ma  = prices.iloc[max(0, end_idx - ma_window) : end_idx].mean()
    cur = prices.iloc[end_idx]
    return ((cur > ma).astype(float) * 2 - 1).dropna()


def generate_weights(
    prices: pd.DataFrame,
    ts_weight: float = TS_WEIGHT,
    ma_window: int   = MA_WINDOW,
) -> pd.DataFrame:
    weights       = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    dates         = prices.index
    rebal_dates   = dates[FORMATION_MONTHS * 21 :: REBAL_FREQ]
    current_w     = pd.Series(0.0, index=prices.columns)

    for date in dates:
        if date in rebal_dates:
            end_idx = prices.index.get_loc(date)

            cs_raw = _cs_momentum_score(prices, end_idx)
            ts_raw = _ts_signal(prices, end_idx, ma_window)

            if cs_raw.empty:
                current_w = pd.Series(0.0, index=prices.columns)
                weights.loc[date] = current_w
                continue

            # Rank CS scores to [-1, +1]
            cs_ranked = cs_raw.rank(pct=True) * 2 - 1  # uniform [-1, 1]

            # Align TS to same tickers
            ts_aligned = ts_raw.reindex(cs_ranked.index).fillna(0.0)

            # Blend
            combined = (1 - ts_weight) * cs_ranked + ts_weight * ts_aligned

            n        = len(combined)
            n_long   = max(1, int(n * TOP_DECILE))
            n_short  = max(1, int(n * BOT_DECILE))
            ranked   = combined.rank(ascending=False)

            new_w    = pd.Series(0.0, index=prices.columns)
            new_w[ranked[ranked <= n_long].index]           = 1.0 / n_long
            new_w[ranked[ranked >= n - n_short + 1].index]  = -1.0 / n_short
            current_w = new_w

        weights.loc[date] = current_w

    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(gross, axis=0).fillna(0.0)
