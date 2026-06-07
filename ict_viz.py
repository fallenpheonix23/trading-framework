"""
Visualization: annotated price chart, equity curve, drawdown, performance tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


COLORS = {
    "bull_fvg": "#00cc66",
    "bear_fvg": "#ff4444",
    "bull_ob": "#0066ff",
    "bear_ob": "#ff6600",
    "bull_bb": "#00ffcc",
    "bear_bb": "#ff00aa",
    "mss_bull": "#00ff00",
    "mss_bear": "#ff0000",
    "long_entry": "#00cc66",
    "short_entry": "#ff4444",
    "target": "#ffffff",
    "stop": "#ff8800",
    "bg": "#0d0d0d",
    "grid": "#1a1a1a",
    "text": "#cccccc",
    "price": "#e0e0e0",
}


def _setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": "#333333",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def plot_price_with_structures(
    df: pd.DataFrame,
    fvgs: pd.DataFrame,
    obs: pd.DataFrame,
    breakers: pd.DataFrame,
    mss: pd.DataFrame,
    trades: list,
    title: str = "ICT Structure Analysis",
    max_bars: int = 500,
    start_bar: int = 0,
) -> plt.Figure:
    """
    Plot annotated price chart showing FVGs, OBs, Breaker Blocks, MSS signals, and trades.
    max_bars limits display window for readability.
    """
    _setup_dark_style()

    end_bar = min(start_bar + max_bars, len(df))
    df_slice = df.iloc[start_bar:end_bar]
    x_dates = df_slice.index

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [4, 1]},
        facecolor=COLORS["bg"]
    )

    # --- Candlestick (simplified: line chart for speed; OHLC bars via fill_between) ---
    # Use OHLC bars
    for i, (ts, row) in enumerate(df_slice.iterrows()):
        color = COLORS["bull_fvg"] if row["close"] >= row["open"] else COLORS["bear_fvg"]
        alpha = 0.8
        x_pos = i
        ax_price.plot([x_pos, x_pos], [row["low"], row["high"]], color=color, linewidth=0.8, alpha=alpha)
        ax_price.plot([x_pos - 0.3, x_pos], [row["open"], row["open"]], color=color, linewidth=1.5, alpha=alpha)
        ax_price.plot([x_pos, x_pos + 0.3], [row["close"], row["close"]], color=color, linewidth=1.5, alpha=alpha)

    n = len(df_slice)

    # --- FVGs ---
    for _, fvg in fvgs.iterrows():
        bi = int(fvg["bar_index"]) - start_bar
        if not (0 <= bi < n):
            continue
        color = COLORS["bull_fvg"] if fvg["type"] == "bull" else COLORS["bear_fvg"]
        alpha = 0.15 if fvg["mitigated"] else 0.3
        rect = mpatches.FancyArrowPatch
        ax_price.axhspan(fvg["bottom"], fvg["top"],
                         xmin=bi / n, xmax=min((bi + 60) / n, 1.0),
                         color=color, alpha=alpha, zorder=1)

    # --- Order Blocks ---
    for _, ob in obs.iterrows():
        bi = int(ob["bar_index"]) - start_bar
        if not (0 <= bi < n):
            continue
        color = COLORS["bull_ob"] if ob["type"] == "bull" else COLORS["bear_ob"]
        alpha = 0.12 if ob["mitigated"] else 0.25
        ax_price.axhspan(ob["bottom"], ob["top"],
                         xmin=bi / n, xmax=min((bi + 80) / n, 1.0),
                         color=color, alpha=alpha, zorder=1)
        ax_price.text(bi + 1, (ob["top"] + ob["bottom"]) / 2,
                      f"OB {'▲' if ob['type'] == 'bull' else '▼'}",
                      color=color, fontsize=6, alpha=0.7, va="center")

    # --- Breaker Blocks ---
    if not breakers.empty:
        for _, bb in breakers.iterrows():
            bi = int(bb["bar_index"]) - start_bar if not np.isnan(bb["bar_index"]) else -1
            if not (0 <= bi < n):
                continue
            color = COLORS["bull_bb"] if bb["type"] == "bull" else COLORS["bear_bb"]
            ax_price.axhspan(bb["bottom"], bb["top"],
                             xmin=bi / n, xmax=min((bi + 60) / n, 1.0),
                             color=color, alpha=0.2, zorder=1, linestyle="--")
            ax_price.text(bi + 1, (bb["top"] + bb["bottom"]) / 2,
                          f"BB {'▲' if bb['type'] == 'bull' else '▼'}",
                          color=color, fontsize=6, alpha=0.7, va="center")

    # --- MSS arrows ---
    for _, m in mss.iterrows():
        bi = int(m["bar_index"]) - start_bar
        if not (0 <= bi < n):
            continue
        price_level = df_slice.iloc[bi]["high"] if m["type"] == "bull" else df_slice.iloc[bi]["low"]
        color = COLORS["mss_bull"] if m["type"] == "bull" else COLORS["mss_bear"]
        arrow_dy = 1.5 if m["type"] == "bull" else -1.5
        ax_price.annotate(
            f"MSS {'▲' if m['type'] == 'bull' else '▼'}",
            xy=(bi, price_level),
            xytext=(bi, price_level + arrow_dy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            fontsize=7, color=color, ha="center",
        )

    # --- Trades ---
    for trade in trades:
        ei = trade.entry_bar - start_bar
        xi = trade.exit_bar - start_bar if trade.exit_bar is not None else None
        if not (0 <= ei < n):
            continue
        e_color = COLORS["long_entry"] if trade.direction == "long" else COLORS["short_entry"]

        ax_price.axhline(y=trade.entry_price, xmin=ei / n,
                         xmax=min((xi or ei + 40) / n, 1.0),
                         color=e_color, linewidth=0.8, linestyle="-", alpha=0.6)
        ax_price.axhline(y=trade.stop_price, xmin=ei / n,
                         xmax=min((xi or ei + 40) / n, 1.0),
                         color=COLORS["stop"], linewidth=0.6, linestyle=":", alpha=0.5)
        ax_price.axhline(y=trade.target_price, xmin=ei / n,
                         xmax=min((xi or ei + 40) / n, 1.0),
                         color=COLORS["target"], linewidth=0.6, linestyle=":", alpha=0.4)

        ax_price.scatter(ei, trade.entry_price, marker="^" if trade.direction == "long" else "v",
                         color=e_color, s=60, zorder=5)

        if xi is not None and 0 <= xi < n:
            exit_color = COLORS["long_entry"] if trade.exit_reason == "target" else COLORS["stop"]
            ax_price.scatter(xi, trade.exit_price, marker="x", color=exit_color, s=50, zorder=5)

    # Volume
    vol_colors = [
        COLORS["bull_fvg"] if df_slice["close"].iloc[i] >= df_slice["open"].iloc[i]
        else COLORS["bear_fvg"]
        for i in range(len(df_slice))
    ]
    ax_vol.bar(range(len(df_slice)), df_slice["volume"].values, color=vol_colors, alpha=0.5, width=0.8)
    ax_vol.set_ylabel("Volume", fontsize=8)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    # X-axis labels: sample every ~50 bars
    tick_positions = list(range(0, n, max(1, n // 10)))
    tick_labels = [x_dates[i].strftime("%Y-%m-%d\n%H:%M") for i in tick_positions]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, fontsize=6, rotation=30)
    ax_vol.set_xticks(tick_positions)
    ax_vol.set_xticklabels(tick_labels, fontsize=6, rotation=30)

    ax_price.set_title(title, fontsize=14, color=COLORS["text"], pad=10)
    ax_price.set_ylabel("Price", fontsize=9)
    ax_price.grid(True, alpha=0.3)
    ax_vol.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["bull_fvg"], alpha=0.5, label="Bullish FVG"),
        mpatches.Patch(color=COLORS["bear_fvg"], alpha=0.5, label="Bearish FVG"),
        mpatches.Patch(color=COLORS["bull_ob"], alpha=0.5, label="Bull Order Block"),
        mpatches.Patch(color=COLORS["bear_ob"], alpha=0.5, label="Bear Order Block"),
        mpatches.Patch(color=COLORS["bull_bb"], alpha=0.5, label="Bull Breaker"),
        mpatches.Patch(color=COLORS["bear_bb"], alpha=0.5, label="Bear Breaker"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["long_entry"],
                   markersize=8, label="Long Entry"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor=COLORS["short_entry"],
                   markersize=8, label="Short Entry"),
    ]
    ax_price.legend(handles=legend_elements, loc="upper left", fontsize=7,
                    framealpha=0.3, facecolor=COLORS["bg"])

    plt.tight_layout()
    return fig


def plot_equity_and_drawdown(equity_curve: pd.Series, trades: list) -> plt.Figure:
    """Equity curve with drawdown subplot and trade markers."""
    _setup_dark_style()

    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(16, 8),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        facecolor=COLORS["bg"])

    # Equity
    ax_eq.plot(equity_curve.index, equity_curve.values,
               color="#4da6ff", linewidth=1.5, label="Equity")
    ax_eq.axhline(y=100_000, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_eq.fill_between(equity_curve.index, 100_000, equity_curve.values,
                       where=equity_curve.values >= 100_000,
                       color="#00cc66", alpha=0.15, label="Profit")
    ax_eq.fill_between(equity_curve.index, 100_000, equity_curve.values,
                       where=equity_curve.values < 100_000,
                       color="#ff4444", alpha=0.15, label="Loss")

    # Trade exit markers on equity
    for trade in trades:
        if trade.exit_time is None or trade.pnl is None:
            continue
        eq_val = equity_curve.asof(trade.exit_time) if trade.exit_time in equity_curve.index \
            else equity_curve.iloc[min(trade.exit_bar, len(equity_curve) - 1)]
        color = COLORS["long_entry"] if trade.pnl > 0 else COLORS["bear_fvg"]
        ax_eq.scatter(trade.exit_time, eq_val, color=color, s=20, zorder=4, alpha=0.7)

    ax_eq.set_title("Equity Curve", fontsize=12, color=COLORS["text"])
    ax_eq.set_ylabel("Portfolio Value ($)", fontsize=9)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(fontsize=8, framealpha=0.3, facecolor=COLORS["bg"])

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       color="#ff4444", alpha=0.5, label="Drawdown")
    ax_dd.plot(drawdown.index, drawdown.values, color="#ff4444", linewidth=0.8)
    ax_dd.set_title("Drawdown %", fontsize=10, color=COLORS["text"])
    ax_dd.set_ylabel("DD %", fontsize=8)
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_metrics_table(metrics: dict):
    """Pretty-print the performance metrics."""
    print("\n" + "=" * 55)
    print("  ICT BACKTEST RESULTS — SPY 1H (2 Years)")
    print("=" * 55)

    summary_keys = [
        ("total_trades", "Total Trades", ""),
        ("win_rate", "Win Rate", "%", 100),
        ("avg_r_multiple", "Avg R-Multiple", "R"),
        ("profit_factor", "Profit Factor", "x"),
        ("sharpe_ratio", "Sharpe Ratio (Ann.)", ""),
        ("max_drawdown_pct", "Max Drawdown", "%"),
        ("total_pnl", "Total P&L", "$"),
        ("final_equity", "Final Equity", "$"),
        ("avg_win", "Avg Win", "$"),
        ("avg_loss", "Avg Loss", "$"),
    ]

    for key, label, unit, *mult in summary_keys:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            if mult:
                val = val * mult[0]
            print(f"  {label:<28} {val:>10.2f} {unit}")
        else:
            print(f"  {label:<28} {str(val):>10} {unit}")

    print("\n--- Kill Zone Breakdown ---")
    print(metrics["kill_zone_stats"].to_string())

    print("\n--- Signal Source Breakdown ---")
    print(metrics["source_stats"].to_string())
    print("=" * 55 + "\n")
