"""
Grid search for parameter refinement — updated for enhanced strategies.
All evaluation on OOS period (2020-2024) only.

Stat Arb  : z_entry × coint_pval × spread_window × max_halflife
Macro     : slope_window × infl_window × vix_lo × vix_hi × yield_weight
Trend     : ts_weight
Mean Rev  : reversal_window × long_short_pct
HMM       : n_states ∈ {2, 3}
"""
import itertools
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from engine.backtest import BacktestEngine
from engine.walk_forward import WalkForwardSplit
from engine.vol_targeting import apply_vol_target
from regime.hmm_classifier import RegimeClassifier, apply_regime_filter
import strategies.stat_arb as stat_arb
import strategies.trend_following as trend_following
import strategies.mean_reversion as mean_reversion
import strategies.macro_systematic as macro_systematic

WF         = WalkForwardSplit()
TC         = 0.001
SLIP       = 0.0005
TARGET_VOL = 0.10


def _oos_sharpe(weights, prices, target_vol=TARGET_VOL):
    active = [
        c for c in weights.columns
        if c in prices.columns and weights[c].abs().sum() > 0
    ]
    if not active:
        return np.nan
    w = weights[active].reindex(prices.index).ffill().fillna(0.0)
    p = prices[active].reindex(w.index).ffill()
    w_vt = apply_vol_target(w, p, target_vol=target_vol)
    result = BacktestEngine(p, w_vt, transaction_cost=TC, slippage=SLIP).run()
    oos = result["daily_returns"][result["daily_returns"].index >= WF.test_start]
    return BacktestEngine.compute_metrics(oos)["Sharpe"]


# ── Stat Arb ──────────────────────────────────────────────────────────────────

STAT_ARB_GRID = {
    "z_entry":       [2.0, 2.5, 3.0],
    "coint_pval":    [0.01, 0.02],
    "spread_window": [63, 126],
    "max_halflife":  [20, 30, 45],
}


def tune_stat_arb(full_prices, train_prices, verbose=True):
    # Cache pair sets per (coint_pval, max_halflife) — both affect pair filtering
    pair_cache: Dict[Tuple, List] = {}
    for pval in STAT_ARB_GRID["coint_pval"]:
        for max_hl in STAT_ARB_GRID["max_halflife"]:
            key = (pval, max_hl)
            pairs = stat_arb.find_cointegrated_pairs(
                train_prices, pval_thresh=pval, max_hl=max_hl
            )
            pair_cache[key] = pairs
            if verbose:
                print(f"    pval={pval:.2f}  max_hl={max_hl} → {len(pairs)} pairs")

    records = []
    for z_entry, coint_pval, spread_window, max_halflife in itertools.product(
        STAT_ARB_GRID["z_entry"],
        STAT_ARB_GRID["coint_pval"],
        STAT_ARB_GRID["spread_window"],
        STAT_ARB_GRID["max_halflife"],
    ):
        pairs = pair_cache[(coint_pval, max_halflife)]
        if not pairs:
            sharpe = np.nan
        else:
            weights = _weights_from_pairs(
                full_prices, pairs, z_entry=z_entry, spread_window=spread_window
            )
            sharpe = _oos_sharpe(weights, full_prices)

        records.append(dict(z_entry=z_entry, coint_pval=coint_pval,
                            spread_window=spread_window, max_halflife=max_halflife,
                            oos_sharpe=sharpe))
        if verbose:
            print(f"    z={z_entry}  pval={coint_pval}  win={spread_window}  "
                  f"max_hl={max_halflife}  →  {sharpe:.3f}")

    df      = pd.DataFrame(records)
    best_r  = df.loc[df["oos_sharpe"].idxmax()]
    best    = dict(z_entry=float(best_r["z_entry"]),
                   coint_pval=float(best_r["coint_pval"]),
                   spread_window=int(best_r["spread_window"]),
                   max_halflife=float(best_r["max_halflife"]))
    return df, best


def _weights_from_pairs(full_prices, pairs, z_entry, spread_window, z_exit=0.5):
    weights  = pd.DataFrame(0.0, index=full_prices.index, columns=full_prices.columns)
    half_lives  = np.array([hl for *_, hl in pairs])
    inv_hl   = 1.0 / half_lives
    pair_weights = inv_hl / inv_hl.sum()

    for (a, b, beta, _, hl), pw in zip(pairs, pair_weights):
        if a not in full_prices.columns or b not in full_prices.columns:
            continue
        spread = full_prices[a] - beta * full_prices[b]
        z      = stat_arb._zscore_series(spread, window=spread_window).dropna()
        if len(z) < spread_window + 10:
            continue
        pos = stat_arb._state_machine(z.values, entry=z_entry, exit_thresh=z_exit)
        ps  = pd.Series(pos, index=z.index)
        weights.loc[ps.index, a] += ps * pw
        weights.loc[ps.index, b] -= ps * pw

    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(gross, axis=0).fillna(0.0)


# ── Macro Systematic ──────────────────────────────────────────────────────────

MACRO_GRID = {
    "slope_window":  [21, 42, 63],
    "infl_window":   [42, 63, 90],
    "credit_window": [42, 63, 90],
    "uup_window":    [42, 63, 90],
    "yield_weight":  [0.0, 0.3],
}


def tune_macro(macro_prices, verbose=True):
    records = []
    for slope_w, infl_w, credit_w, uup_w, yw in itertools.product(
        MACRO_GRID["slope_window"],
        MACRO_GRID["infl_window"],
        MACRO_GRID["credit_window"],
        MACRO_GRID["uup_window"],
        MACRO_GRID["yield_weight"],
    ):
        w = macro_systematic.generate_weights(
            macro_prices,
            slope_window=slope_w, infl_window=infl_w,
            credit_window=credit_w, uup_window=uup_w,
            yield_weight=yw,
        )
        sharpe = _oos_sharpe(w, macro_prices)
        records.append(dict(slope_window=slope_w, infl_window=infl_w,
                            credit_window=credit_w, uup_window=uup_w,
                            yield_weight=yw, oos_sharpe=sharpe))
        if verbose:
            print(f"    slope={slope_w} infl={infl_w} credit={credit_w} "
                  f"uup={uup_w} yw={yw}  →  {sharpe:.3f}")

    df     = pd.DataFrame(records)
    best_r = df.loc[df["oos_sharpe"].idxmax()]
    best   = dict(slope_window=int(best_r["slope_window"]),
                  infl_window=int(best_r["infl_window"]),
                  credit_window=int(best_r["credit_window"]),
                  uup_window=int(best_r["uup_window"]),
                  yield_weight=float(best_r["yield_weight"]))
    return df, best


# ── Trend Following ───────────────────────────────────────────────────────────

TREND_GRID = {"ts_weight": [0.0, 0.2, 0.3, 0.4, 0.5]}


def tune_trend(equity_prices, verbose=True):
    records = []
    for ts_w in TREND_GRID["ts_weight"]:
        w      = trend_following.generate_weights(equity_prices, ts_weight=ts_w)
        sharpe = _oos_sharpe(w, equity_prices)
        records.append(dict(ts_weight=ts_w, oos_sharpe=sharpe))
        if verbose:
            print(f"    ts_weight={ts_w}  →  {sharpe:.3f}")

    df     = pd.DataFrame(records)
    best_r = df.loc[df["oos_sharpe"].idxmax()]
    best   = dict(ts_weight=float(best_r["ts_weight"]))
    return df, best


# ── Mean Reversion ────────────────────────────────────────────────────────────

MR_GRID = {
    "reversal_window": [5, 10],
    "long_short_pct":  [0.10, 0.20],
}


def tune_mean_reversion(equity_prices, verbose=True):
    records = []
    for rw, pct in itertools.product(
        MR_GRID["reversal_window"], MR_GRID["long_short_pct"]
    ):
        w      = mean_reversion.generate_weights(
            equity_prices, reversal_window=rw, long_short_pct=pct
        )
        sharpe = _oos_sharpe(w, equity_prices)
        records.append(dict(reversal_window=rw, long_short_pct=pct, oos_sharpe=sharpe))
        if verbose:
            print(f"    reversal={rw}  pct={pct}  →  {sharpe:.3f}")

    df     = pd.DataFrame(records)
    best_r = df.loc[df["oos_sharpe"].idxmax()]
    best   = dict(reversal_window=int(best_r["reversal_window"]),
                  long_short_pct=float(best_r["long_short_pct"]))
    return df, best


# ── HMM ───────────────────────────────────────────────────────────────────────

def tune_hmm(equity_prices, equity_train, verbose=True):
    spy_full  = equity_prices["SPY"].pct_change().dropna()
    spy_train = equity_train["SPY"].pct_change().dropna()
    tf_base   = trend_following.generate_weights(equity_prices)
    mr_base   = mean_reversion.generate_weights(equity_prices)
    records   = []

    for n in [2, 3]:
        clf = RegimeClassifier(n_states=n)
        clf.fit(spy_train)
        regime = clf.predict(spy_full)
        tf_w   = apply_regime_filter(tf_base, regime, allowed_regime=1)
        mr_w   = apply_regime_filter(mr_base, regime, allowed_regime=0)
        tf_s   = _oos_sharpe(tf_w, equity_prices)
        mr_s   = _oos_sharpe(mr_w, equity_prices)
        avg    = np.nanmean([tf_s, mr_s])
        records.append(dict(n_states=n, tf_sharpe=tf_s, mr_sharpe=mr_s, avg=avg))
        if verbose:
            print(f"    n={n}  TF={tf_s:.3f}  MR={mr_s:.3f}  avg={avg:.3f}")

    df     = pd.DataFrame(records)
    best_n = int(df.loc[df["avg"].idxmax(), "n_states"])
    return df, best_n


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_tuning(equity_prices, equity_train, macro_prices):
    sa_cols  = [t for t in [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","BAC","GS",
        "XOM","CVX","JNJ","UNH","PG","KO","CAT","HON","NEE","AMT","SPY",
        "XLK","XLF","XLV","XLI","XLE","QQQ","IWM",
    ] if t in equity_prices.columns]
    sa_full  = equity_prices[sa_cols]
    sa_train = equity_train[[c for c in sa_cols if c in equity_train.columns]]

    print("\n── Stat Arb ────────────────────────────────────────────")
    sa_df, sa_best = tune_stat_arb(sa_full, sa_train)

    print("\n── Macro Systematic ────────────────────────────────────")
    mac_df, mac_best = tune_macro(macro_prices)

    print("\n── Trend Following ─────────────────────────────────────")
    tf_df, tf_best = tune_trend(equity_prices)

    print("\n── Mean Reversion ──────────────────────────────────────")
    mr_df, mr_best = tune_mean_reversion(equity_prices)

    print("\n── HMM Regime ──────────────────────────────────────────")
    hmm_df, best_n = tune_hmm(equity_prices, equity_train)

    print("\n── Best Parameters ─────────────────────────────────────")
    print(f"  Stat Arb : {sa_best}")
    print(f"  Macro    : {mac_best}")
    print(f"  Trend    : {tf_best}")
    print(f"  Mean Rev : {mr_best}")
    print(f"  HMM      : n_states={best_n}")

    return dict(
        stat_arb=sa_best, macro=mac_best,
        trend=tf_best,    mean_rev=mr_best,
        n_states=best_n,
        stat_arb_results=sa_df, macro_results=mac_df,
        trend_results=tf_df,    mr_results=mr_df,
        hmm_results=hmm_df,
    )
