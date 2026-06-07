"""
Main runner — enhanced strategies + VIX-scaled Kelly.
"""
import warnings
import sys
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.pipeline import load_equity_prices, load_macro_data
from engine.backtest import BacktestEngine
from engine.walk_forward import WalkForwardSplit
from engine.vol_targeting import apply_vol_target
from regime.hmm_classifier import RegimeClassifier, apply_regime_filter
from portfolio.kelly_allocator import combine_equity_curves
from analytics.performance import full_report, monthly_returns, correlation_matrix
from visualization.dashboard import build_full_dashboard
from tuning.grid_search import run_all_tuning

import strategies.stat_arb as stat_arb
import strategies.trend_following as trend_following
import strategies.mean_reversion as mean_reversion
import strategies.macro_systematic as macro_systematic
import strategies.market_making as market_making

# ── Configuration ──────────────────────────────────────────────────────────────
ACTIVE_STRATEGIES = {
    "Stat Arb":         False,
    "Trend Following":  False,
    "Mean Reversion":   False,
    "Macro Systematic": True,
    "Market Making":    False,
}

TC         = 0.001
SLIP       = 0.0005
CAPITAL    = 1_000_000.0
TARGET_VOL = 0.10

SA_UNIVERSE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","BAC","GS",
    "XOM","CVX","JNJ","UNH","PG","KO","CAT","HON","NEE","AMT","SPY",
    "XLK","XLF","XLV","XLI","XLE","QQQ","IWM",
]

# Sensible defaults (replaced by --tune results when used)
DEFAULT_PARAMS = {
    "stat_arb":  {"z_entry": 2.5, "coint_pval": 0.01,
                  "spread_window": 126, "max_halflife": 30.0},
    "macro":     {"slope_window": 21, "infl_window": 63,
                  "credit_window": 90, "uup_window": 63,
                  "yield_weight": 0.0},
    "trend":     {"ts_weight": 0.3},
    "mean_rev":  {"reversal_window": 5, "long_short_pct": 0.20},
    "n_states":  3,
}


def main(force_refresh: bool = False, tune: bool = False):
    print("=" * 65)
    print("  Trading Framework  |  Enhanced Strategies v2")
    print("=" * 65)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    equity_prices = load_equity_prices(force_refresh=force_refresh)
    macro_prices  = load_macro_data(force_refresh=force_refresh)

    equity_prices = equity_prices.dropna(axis=1, thresh=int(len(equity_prices) * 0.7))
    equity_prices = equity_prices.ffill(limit=5)

    for ticker in ["GLD", "SPY"]:
        if ticker not in macro_prices.columns:
            if ticker in equity_prices.columns:
                macro_prices[ticker] = equity_prices[ticker]
            else:
                import yfinance as yf
                dl = yf.download(ticker, start="2015-01-01", end="2024-12-31",
                                 auto_adjust=True, progress=False)["Close"]
                dl.index = pd.to_datetime(dl.index).tz_localize(None)
                macro_prices[ticker] = dl

    # VIX series for portfolio-level scaling
    vix_series = macro_prices["^VIX"].ffill() if "^VIX" in macro_prices.columns else None

    # ── 2. Walk-Forward Split ──────────────────────────────────────────────────
    wf = WalkForwardSplit()
    equity_train, _ = wf.split(equity_prices)
    spy_full  = equity_prices["SPY"].pct_change().dropna()
    spy_train = equity_train["SPY"].pct_change().dropna()

    print(f"      Train: {equity_train.index[0].date()} → {equity_train.index[-1].date()}")
    print(f"      Test:  {wf.test_start.date()} → {equity_prices.index[-1].date()}")

    # ── 3. Optional tuning ────────────────────────────────────────────────────
    best = DEFAULT_PARAMS.copy()
    if tune:
        print("\n[TUNING] Running parameter grid search...")
        tuned = run_all_tuning(equity_prices, equity_train, macro_prices)
        best["stat_arb"] = tuned["stat_arb"]
        best["macro"]    = tuned["macro"]
        best["trend"]    = tuned["trend"]
        best["mean_rev"] = tuned["mean_rev"]
        best["n_states"] = tuned["n_states"]
        out = Path(__file__).parent / "output"
        for key, fname in [("stat_arb_results","tune_stat_arb.csv"),
                            ("macro_results","tune_macro.csv"),
                            ("trend_results","tune_trend.csv"),
                            ("mr_results","tune_mean_rev.csv"),
                            ("hmm_results","tune_hmm.csv")]:
            if key in tuned:
                tuned[key].to_csv(out / fname, index=False)
        print("      CSVs saved to ./output/")

    # ── 4. HMM Regime ─────────────────────────────────────────────────────────
    print("\n[2/6] Fitting HMM regime classifier...")
    n_states   = best["n_states"]
    classifier = RegimeClassifier(n_states=n_states)
    classifier.fit(spy_train)
    regime = classifier.predict(spy_full)
    print(f"      n_states={n_states}  "
          f"Train: {regime[regime.index <= wf.train_end].value_counts().to_dict()}  "
          f"Test: {regime[regime.index >= wf.test_start].value_counts().to_dict()}")

    # ── 5. Strategy Weights ────────────────────────────────────────────────────
    print("\n[3/6] Generating strategy weights...")
    strategy_weights = {}
    strategy_prices  = {}

    if ACTIVE_STRATEGIES.get("Stat Arb"):
        sa_p = best["stat_arb"]
        print(f"      → Stat Arb  {sa_p}")
        sa_cols  = [t for t in SA_UNIVERSE if t in equity_prices.columns]
        sa_full  = equity_prices[sa_cols]
        sa_train = equity_train[[c for c in sa_cols if c in equity_train.columns]]
        w = stat_arb.generate_weights(sa_full, sa_train, **sa_p)
        strategy_weights["Stat Arb"] = w
        strategy_prices["Stat Arb"]  = sa_full

    if ACTIVE_STRATEGIES.get("Trend Following"):
        tf_p = best["trend"]
        print(f"      → Trend Following  ts_weight={tf_p['ts_weight']}")
        w = trend_following.generate_weights(equity_prices, **tf_p)
        w = apply_regime_filter(w, regime, allowed_regime=1)
        strategy_weights["Trend Following"] = w
        strategy_prices["Trend Following"]  = equity_prices

    if ACTIVE_STRATEGIES.get("Mean Reversion"):
        mr_p = best["mean_rev"]
        print(f"      → Mean Reversion  {mr_p}")
        w = mean_reversion.generate_weights(equity_prices, **mr_p)
        w = apply_regime_filter(w, regime, allowed_regime=0)
        strategy_weights["Mean Reversion"] = w
        strategy_prices["Mean Reversion"]  = equity_prices

    if ACTIVE_STRATEGIES.get("Macro Systematic"):
        mp = best["macro"]
        print(f"      → Macro Systematic  {mp}")
        w = macro_systematic.generate_weights(macro_prices, **mp)
        strategy_weights["Macro Systematic"] = w
        strategy_prices["Macro Systematic"]  = macro_prices

    if ACTIVE_STRATEGIES.get("Market Making"):
        mm_px = macro_prices if "SPY" in macro_prices.columns else equity_prices
        w = market_making.generate_weights(mm_px)
        strategy_weights["Market Making"] = w
        strategy_prices["Market Making"]  = mm_px

    # ── 6. Vol-Target + Backtest ───────────────────────────────────────────────
    print("\n[4/6] Applying vol-targeting and running backtests...")
    all_returns   = {}
    equity_curves = {}

    for name, raw_w in strategy_weights.items():
        px = strategy_prices[name]
        active = [c for c in raw_w.columns
                  if c in px.columns and raw_w[c].abs().sum() > 0]
        if not active:
            print(f"      WARNING: no active columns for {name}")
            continue

        w = raw_w[active].reindex(px.index).ffill().fillna(0.0)
        p = px[active].reindex(w.index).ffill()
        w_vt = apply_vol_target(w, p, target_vol=TARGET_VOL)

        result   = BacktestEngine(p, w_vt, TC, SLIP, CAPITAL).run()
        full_r   = result["daily_returns"]
        oos_r    = full_r[full_r.index >= wf.test_start]
        full_m   = BacktestEngine.compute_metrics(full_r)
        oos_m    = BacktestEngine.compute_metrics(oos_r)

        print(f"      {name}")
        print(f"        Full  Sharpe={full_m['Sharpe']:>7}  MaxDD={full_m['Max Drawdown']:>7}%")
        print(f"        OOS   Sharpe={oos_m['Sharpe']:>7}  MaxDD={oos_m['Max Drawdown']:>7}%")

        all_returns[name]   = full_r
        equity_curves[name] = result["equity_curve"]

    if not all_returns:
        print("No strategies produced results.")
        return

    # ── 7. Kelly + VIX-Scaled Portfolio ───────────────────────────────────────
    print("\n[5/6] Kelly + VIX-scaled portfolio...")
    returns_df     = pd.DataFrame(all_returns).dropna(how="all").fillna(0.0)
    combined_result = combine_equity_curves(
        returns_df,
        initial_capital=CAPITAL,
    )
    comb_eq  = combined_result["equity_curve"]
    comb_ret = combined_result["daily_returns"]

    full_cm = BacktestEngine.compute_metrics(comb_ret)
    oos_cm  = BacktestEngine.compute_metrics(
        comb_ret[comb_ret.index >= wf.test_start]
    )
    print(f"      Combined  Full  Sharpe={full_cm['Sharpe']:>7}  "
          f"MaxDD={full_cm['Max Drawdown']:>7}%")
    print(f"      Combined  OOS   Sharpe={oos_cm['Sharpe']:>7}  "
          f"MaxDD={oos_cm['Max Drawdown']:>7}%")

    # ── 8. Analytics & Dashboard ───────────────────────────────────────────────
    print("\n[6/6] Building analytics and dashboard...")
    oos_strat = {n: r[r.index >= wf.test_start] for n, r in all_returns.items()}
    oos_comb  = comb_ret[comb_ret.index >= wf.test_start]

    perf_full = full_report(all_returns, comb_ret)
    perf_oos  = full_report(oos_strat,   oos_comb)
    monthly_df = monthly_returns(comb_ret)
    corr_df    = correlation_matrix(all_returns)

    print("\n── Full-Period Performance ──────────────────────────────")
    print(perf_full.to_string())
    print("\n── OOS Performance (2020-2024) ─────────────────────────")
    print(perf_oos.to_string())
    print("\n── Strategy Correlation Matrix ─────────────────────────")
    print(corr_df.round(3).to_string())

    build_full_dashboard(
        strategy_equity=equity_curves,
        combined_equity=comb_eq,
        combined_result=combined_result,
        monthly_df=monthly_df,
        corr_df=corr_df,
        perf_df=perf_oos,
    )
    perf_oos.to_csv(Path(__file__).parent / "output" / "performance_oos.csv")

    print("\n" + "=" * 65)
    print("  Run complete.  Charts and metrics saved to ./output/")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tune",    action="store_true")
    args = parser.parse_args()
    main(force_refresh=args.refresh, tune=args.tune)
