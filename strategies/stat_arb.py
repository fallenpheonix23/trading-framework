"""
Strategy 1: Statistical Arbitrage — Engle-Granger cointegration pairs trading.

Walk-forward design:
  - Pairs + hedge ratios fitted on TRAIN data only (no look-ahead).

Enhancements vs baseline:
  - Ornstein-Uhlenbeck half-life filter: only trade pairs whose spread
    mean-reverts within max_halflife days.  Slow pairs tie up capital
    in positions that take months to resolve → consistent drag.
  - Position sizing by 1/half_life: faster-reverting pairs get larger
    allocations (they generate more round-trips per unit of risk).
  - Same vectorised state-machine signal (no inner loops).

Signal logic:
  Long spread  (long A, short B) when z < -Z_ENTRY
  Short spread (short A, long B) when z >  Z_ENTRY
  Exit when |z| < Z_EXIT
"""
import itertools
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

Z_ENTRY        = 2.5
Z_EXIT         = 0.5
SPREAD_WINDOW  = 126
COINT_PVAL     = 0.01
MIN_TRAIN_OBS  = 200
MAX_PAIRS      = 20
MIN_HALFLIFE   = 5     # days — below this: noise, not signal
MAX_HALFLIFE   = 30    # days — above this: too slow to trade profitably


# ── Pair selection (train only) ───────────────────────────────────────────────

def _ou_halflife(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life via OLS on lagged spread."""
    s    = spread.dropna()
    lag  = s.shift(1).dropna()
    delta = s.diff().dropna()
    idx  = lag.index.intersection(delta.index)
    if len(idx) < 30:
        return np.inf
    try:
        mdl  = OLS(delta.loc[idx], add_constant(lag.loc[idx])).fit()
        beta = float(mdl.params.iloc[1])
        if beta >= 0:
            return np.inf
        return float(-np.log(2) / beta)
    except Exception:
        return np.inf


def find_cointegrated_pairs(
    train_prices: pd.DataFrame,
    pval_thresh:  float = COINT_PVAL,
    max_pairs:    int   = MAX_PAIRS,
    min_hl:       float = MIN_HALFLIFE,
    max_hl:       float = MAX_HALFLIFE,
) -> List[Tuple[str, str, float, float, float]]:
    """
    Returns list of (ticker_a, ticker_b, beta, pval, halflife).
    Only pairs passing the half-life filter are returned.
    """
    px   = train_prices.dropna(axis=1, thresh=int(len(train_prices) * 0.8))
    cols = px.columns.tolist()
    results = []

    for a, b in itertools.combinations(cols, 2):
        ab = px[[a, b]].dropna()
        if len(ab) < MIN_TRAIN_OBS:
            continue
        _, pval, _ = coint(ab[a], ab[b])
        if pval >= pval_thresh:
            continue

        mdl  = OLS(ab[a], add_constant(ab[b])).fit()
        beta = float(mdl.params[b])
        spread = ab[a] - beta * ab[b]
        hl   = _ou_halflife(spread)

        if min_hl <= hl <= max_hl:
            results.append((a, b, beta, pval, hl))

    results.sort(key=lambda x: x[3])   # sort by p-value
    return results[:max_pairs]


# ── Vectorised signal generation ──────────────────────────────────────────────

def _zscore_series(spread: pd.Series, window: int = SPREAD_WINDOW) -> pd.Series:
    mu    = spread.rolling(window).mean()
    sigma = spread.rolling(window).std().replace(0, np.nan)
    return (spread - mu) / sigma


def _state_machine(z_arr: np.ndarray, entry: float, exit_thresh: float) -> np.ndarray:
    n, pos, state = len(z_arr), np.zeros(len(z_arr)), 0.0
    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            pos[i] = state; continue
        if state == 0.0:
            if z < -entry:  state =  1.0
            elif z > entry: state = -1.0
        elif state == 1.0:
            if z > -exit_thresh: state = 0.0
        elif state == -1.0:
            if z < exit_thresh:  state = 0.0
        pos[i] = state
    return pos


def generate_weights(
    full_prices:   pd.DataFrame,
    train_prices:  pd.DataFrame,
    z_entry:       float = Z_ENTRY,
    z_exit:        float = Z_EXIT,
    coint_pval:    float = COINT_PVAL,
    spread_window: int   = SPREAD_WINDOW,
    max_halflife:  float = MAX_HALFLIFE,
) -> pd.DataFrame:
    pairs = find_cointegrated_pairs(
        train_prices, pval_thresh=coint_pval, max_hl=max_halflife
    )
    if not pairs:
        return pd.DataFrame(0.0, index=full_prices.index, columns=full_prices.columns)

    weights  = pd.DataFrame(0.0, index=full_prices.index, columns=full_prices.columns)

    # Inverse-half-life weights: faster pairs get more capital
    half_lives  = np.array([hl for *_, hl in pairs])
    inv_hl      = 1.0 / half_lives
    pair_weights = inv_hl / inv_hl.sum()

    for (a, b, beta, _, hl), pw in zip(pairs, pair_weights):
        if a not in full_prices.columns or b not in full_prices.columns:
            continue
        spread = full_prices[a] - beta * full_prices[b]
        z      = _zscore_series(spread, window=spread_window).dropna()
        if len(z) < spread_window + 10:
            continue
        pos    = _state_machine(z.values, entry=z_entry, exit_thresh=z_exit)
        ps     = pd.Series(pos, index=z.index)
        weights.loc[ps.index, a] += ps * pw
        weights.loc[ps.index, b] -= ps * pw

    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(gross, axis=0).fillna(0.0)
