"""
Half-Kelly capital allocation with VIX regime scaling.

Rolling 252-day window estimates mu and sigma^2 per strategy.
f_i = mu_i / sigma_i^2, normalise, apply half-Kelly.

VIX scaling: when VIX is elevated, total portfolio allocation is
reduced linearly from 1.0 (VIX <= vix_lo) to 0.5 (VIX >= vix_hi).
This systematically reduces exposure during regime uncertainty.
"""
from typing import Optional
import numpy as np
import pandas as pd

KELLY_WINDOW = 252
MIN_HISTORY  = 126
HALF_KELLY   = 0.5
VIX_LO       = 18
VIX_HI       = 28


def compute_kelly_fractions(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(
        index=strategy_returns.index,
        columns=strategy_returns.columns,
        dtype=float,
    )
    for col in strategy_returns.columns:
        r      = strategy_returns[col]
        mu     = r.rolling(KELLY_WINDOW, min_periods=MIN_HISTORY).mean() * 252
        var    = r.rolling(KELLY_WINDOW, min_periods=MIN_HISTORY).var()  * 252
        kelly  = (mu / var.replace(0, np.nan)).clip(lower=0)
        raw[col] = kelly

    row_sum   = raw.sum(axis=1).replace(0, np.nan)
    normalised = raw.div(row_sum, axis=0).fillna(0.0)
    return normalised * HALF_KELLY


def combine_equity_curves(
    strategy_returns: pd.DataFrame,
    vix_series:       Optional[pd.Series] = None,
    vix_lo:           float = VIX_LO,
    vix_hi:           float = VIX_HI,
    initial_capital:  float = 1_000_000.0,
) -> pd.DataFrame:
    fractions    = compute_kelly_fractions(strategy_returns)
    fractions_lag = fractions.shift(1).fillna(0.0)

    # VIX regime scaling: scale total allocated weight down in crises
    if vix_series is not None:
        vix_aligned = vix_series.reindex(strategy_returns.index).ffill().fillna(vix_lo)
        # 1.0 when calm, 0.5 when crisis — shift 1 day to avoid look-ahead
        risk_scale  = (
            1.0 - 0.5 * ((vix_aligned - vix_lo) / (vix_hi - vix_lo)).clip(0, 1)
        ).shift(1).fillna(1.0)
        fractions_lag = fractions_lag.multiply(risk_scale, axis=0)

    portfolio_returns = (strategy_returns * fractions_lag).sum(axis=1)
    equity = initial_capital * (1 + portfolio_returns).cumprod()

    result = pd.DataFrame({"equity_curve": equity, "daily_returns": portfolio_returns})
    for col in fractions.columns:
        result[f"kelly_{col}"] = fractions[col]
    return result
