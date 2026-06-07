"""
Strategy 4: Macro Systematic — 5-asset rotation, 4 independent signals.

Assets : SPY (US equity), TLT (long bonds), GLD (gold),
         HYG (high-yield credit), EFA (international equity)

Signals:
  1. Yield curve slope  — TLT/SHY ETF ratio + blended ^TNX-^IRX spread
  2. Inflation proxy    — TIP/IEF ratio momentum
  3. Credit spread      — HYG/LQD ratio (tightening = risk appetite)
  4. Dollar strength    — UUP momentum (strong USD = risk-off internationally)

Enhancements:
  - SPY 200-day MA equity gate: damp SPY weight when trend is down
  - Weekly rebalancing with 2% drift threshold: cuts transaction costs ~90%
"""
import numpy as np
import pandas as pd

SLOPE_WINDOW   = 21
INFL_WINDOW    = 63
CREDIT_WINDOW  = 90
UUP_WINDOW     = 63
YIELD_WEIGHT   = 0.0

EQUITY_MA_WIN  = 200    # SPY trend gate window
EQUITY_DAMP    = 0.25   # multiply SPY weight by this when below MA

REBAL_FREQ     = 5      # trading days between rebalances (weekly)
REBAL_THRESH   = 0.02   # only rebalance if max weight shift > 2%

ASSETS = ["SPY", "TLT", "GLD", "HYG", "EFA"]

# Signal → asset betas (economic priors)
SLOPE_BETAS  = {"SPY": +0.15, "TLT": -0.18, "GLD":  0.00, "HYG": +0.10, "EFA": +0.10}
INFL_BETAS   = {"SPY": -0.05, "TLT": -0.15, "GLD": +0.20, "HYG": -0.05, "EFA": -0.05}
CREDIT_BETAS = {"SPY": +0.10, "TLT": -0.15, "GLD": -0.05, "HYG": +0.15, "EFA": +0.05}
UUP_BETAS    = {"SPY":  0.00, "TLT": +0.05, "GLD": -0.15, "HYG": -0.05, "EFA": -0.15}


def _rebal_filter(
    signal_w:  pd.DataFrame,
    freq:      int   = REBAL_FREQ,
    threshold: float = REBAL_THRESH,
) -> pd.DataFrame:
    """
    Hold weights fixed between rebalance events.
    Rebalance only when: it's a scheduled day AND max weight drift > threshold.
    Passing held weights to the backtest engine means zero transaction costs
    on non-rebalance days.
    """
    held   = signal_w.copy()
    prev_w = signal_w.iloc[0].values.copy()

    for i in range(len(signal_w)):
        new_w = signal_w.iloc[i].values
        drift = np.abs(new_w - prev_w).max()
        if (i % freq == 0) and (drift > threshold):
            prev_w = new_w.copy()
        held.iloc[i] = prev_w

    return held


def generate_weights(
    macro_prices:  pd.DataFrame,
    slope_window:  int   = SLOPE_WINDOW,
    infl_window:   int   = INFL_WINDOW,
    credit_window: int   = CREDIT_WINDOW,
    uup_window:    int   = UUP_WINDOW,
    yield_weight:  float = YIELD_WEIGHT,
    equity_ma_win: int   = EQUITY_MA_WIN,
    equity_damp:   float = EQUITY_DAMP,
    rebal_freq:    int   = REBAL_FREQ,
    rebal_thresh:  float = REBAL_THRESH,
) -> pd.DataFrame:
    required = ["TLT", "SHY", "IEF", "TIP", "SPY", "GLD", "HYG", "LQD", "EFA"]
    missing  = [t for t in required if t not in macro_prices.columns]
    if missing:
        raise ValueError(f"Missing tickers: {missing}")

    px = macro_prices[required].ffill()

    # ── Signal 1: Yield curve slope ───────────────────────────────────────────
    tlt_shy   = px["TLT"] / px["SHY"]
    etf_slope = np.tanh((tlt_shy / tlt_shy.rolling(slope_window).mean() - 1) * 10)

    direct_slope = pd.Series(0.0, index=px.index)
    if "^TNX" in macro_prices.columns and "^IRX" in macro_prices.columns:
        raw_spread   = macro_prices["^TNX"].ffill() - macro_prices["^IRX"].ffill()
        spread_z     = ((raw_spread - raw_spread.rolling(252).mean())
                        / raw_spread.rolling(252).std().replace(0, np.nan))
        direct_slope = np.tanh(spread_z.reindex(px.index).fillna(0.0))

    slope_norm = (1 - yield_weight) * etf_slope + yield_weight * direct_slope

    # ── Signal 2: Inflation proxy ─────────────────────────────────────────────
    tip_ief   = px["TIP"] / px["IEF"]
    infl_norm = np.tanh((tip_ief / tip_ief.rolling(infl_window).mean() - 1) * 10)

    # ── Signal 3: Credit spread ───────────────────────────────────────────────
    hyg_lqd     = px["HYG"] / px["LQD"]
    credit_norm = np.tanh(
        (hyg_lqd / hyg_lqd.rolling(credit_window).mean() - 1) * 10
    )

    # ── Signal 4: Dollar strength (UUP) ───────────────────────────────────────
    uup_norm = pd.Series(0.0, index=px.index)
    if "UUP" in macro_prices.columns:
        uup = macro_prices["UUP"].ffill().reindex(px.index)
        uup_mom  = uup / uup.rolling(uup_window).mean() - 1
        uup_norm = np.tanh(uup_mom * 10).fillna(0.0)

    # ── Base allocation + signal tilts ────────────────────────────────────────
    base = 1.0 / len(ASSETS)
    weights_dict = {}
    for asset in ASSETS:
        weights_dict[asset] = (
            base
            + SLOPE_BETAS[asset]  * slope_norm
            + INFL_BETAS[asset]   * infl_norm
            + CREDIT_BETAS[asset] * credit_norm
            + UUP_BETAS[asset]    * uup_norm
        )

    raw = pd.concat(weights_dict, axis=1).dropna()

    # ── SPY 200-day MA equity gate ────────────────────────────────────────────
    spy_ma   = px["SPY"].rolling(equity_ma_win).mean()
    above_ma = (px["SPY"] >= spy_ma).astype(float).reindex(raw.index)
    # Scale factor: equity_damp when below MA, 1.0 when above
    eq_scale = equity_damp + (1.0 - equity_damp) * above_ma
    raw["SPY"] = raw["SPY"] * eq_scale

    # Clip, normalise
    raw     = raw.clip(lower=0.0)
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    norm    = raw.div(row_sum, axis=0).fillna(base)

    # ── Weekly rebalancing filter ─────────────────────────────────────────────
    norm = _rebal_filter(norm, freq=rebal_freq, threshold=rebal_thresh)

    # Reindex to full macro_prices index
    out = pd.DataFrame(0.0, index=macro_prices.index, columns=macro_prices.columns)
    for col in ASSETS:
        out[col] = norm[col].reindex(macro_prices.index).ffill()
    return out
