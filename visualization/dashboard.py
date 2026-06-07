"""
Matplotlib dashboard: equity curves, monthly heatmap, correlation matrix, Kelly fractions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STRATEGY_COLORS = {
    "Stat Arb": "#2196F3",
    "Trend Following": "#4CAF50",
    "Mean Reversion": "#FF9800",
    "Macro Systematic": "#9C27B0",
    "Market Making": "#F44336",
    "Combined Portfolio": "#212121",
}


def plot_equity_curves(
    strategy_equity: dict[str, pd.Series],
    combined_equity: pd.Series,
    save: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, curve in strategy_equity.items():
        color = STRATEGY_COLORS.get(name, None)
        ax.plot(curve.index, curve / curve.iloc[0], label=name, alpha=0.75, linewidth=1.2, color=color)
    ax.plot(
        combined_equity.index,
        combined_equity / combined_equity.iloc[0],
        label="Combined Portfolio",
        color=STRATEGY_COLORS["Combined Portfolio"],
        linewidth=2.5,
        linestyle="--",
    )
    ax.set_title("Strategy Equity Curves (Normalised to 1)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "equity_curves.png", dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_heatmap(
    monthly_df: pd.DataFrame,
    strategy_name: str = "Combined Portfolio",
    save: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, max(4, len(monthly_df) * 0.5)))
    sns.heatmap(
        monthly_df * 100,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Return (%)"},
    )
    ax.set_title(f"Monthly Returns (%) — {strategy_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Year")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "monthly_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(corr: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        mask=False,
    )
    ax.set_title("Strategy Return Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    return fig


def plot_kelly_fractions(combined_result: pd.DataFrame, save: bool = True) -> plt.Figure:
    kelly_cols = [c for c in combined_result.columns if c.startswith("kelly_")]
    if not kelly_cols:
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in kelly_cols:
        name = col.replace("kelly_", "").replace("_", " ").title()
        color = STRATEGY_COLORS.get(name, None)
        ax.plot(combined_result.index, combined_result[col] * 100, label=name, alpha=0.8, color=color)
    ax.set_title("Rolling Half-Kelly Allocation per Strategy (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Allocation (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "kelly_fractions.png", dpi=150, bbox_inches="tight")
    return fig


def plot_performance_table(perf_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, max(3, len(perf_df) * 0.6)))
    ax.axis("off")
    tbl = ax.table(
        cellText=perf_df.values,
        colLabels=perf_df.columns,
        rowLabels=perf_df.index,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ECEFF1" if r % 2 == 0 else "white")
    ax.set_title("Performance Attribution Table", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "performance_table.png", dpi=150, bbox_inches="tight")
    return fig


def build_full_dashboard(
    strategy_equity: dict[str, pd.Series],
    combined_equity: pd.Series,
    combined_result: pd.DataFrame,
    monthly_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    perf_df: pd.DataFrame,
) -> None:
    plot_equity_curves(strategy_equity, combined_equity)
    plot_monthly_heatmap(monthly_df)
    plot_correlation_matrix(corr_df)
    plot_kelly_fractions(combined_result)
    plot_performance_table(perf_df)
    print(f"\nAll charts saved to: {OUTPUT_DIR}")
