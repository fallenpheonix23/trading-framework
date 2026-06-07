"""
Vectorized backtest engine shared by all strategies.
Handles transaction costs, slippage, and position sizing.
"""
import numpy as np
import pandas as pd


class BacktestEngine:
    """
    Converts a DataFrame of target weights into a daily P&L series.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (dates, assets)
    weights : pd.DataFrame
        Target portfolio weights, same shape as prices.
        Weights are applied at close of the signal day and filled forward
        until changed (i.e. weights are end-of-day targets).
    transaction_cost : float
        One-way cost as a fraction (e.g. 0.001 = 10 bps).
    slippage : float
        Additional one-way slippage as a fraction of price.
    initial_capital : float
        Starting NAV in dollars.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        initial_capital: float = 1_000_000.0,
    ):
        self.prices = prices
        self.weights = weights.reindex(prices.index).ffill().fillna(0.0)
        self.tc = transaction_cost
        self.slip = slippage
        self.capital = initial_capital

    def run(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            equity_curve, daily_returns, turnover
        """
        prices = self.prices
        weights = self.weights

        # align
        common_idx = prices.index.intersection(weights.index)
        prices = prices.loc[common_idx]
        weights = weights.loc[common_idx]

        # asset returns (arithmetic for P&L calculation)
        asset_rets = prices.pct_change().fillna(0.0)

        # gross strategy return before costs
        gross = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1)

        # turnover = sum of absolute weight changes each day
        weight_changes = weights.diff().abs().sum(axis=1)
        total_cost = weight_changes * (self.tc + self.slip)

        net_returns = gross - total_cost

        equity = self.capital * (1 + net_returns).cumprod()

        return pd.DataFrame(
            {
                "equity_curve": equity,
                "daily_returns": net_returns,
                "gross_returns": gross,
                "turnover": weight_changes,
            },
            index=common_idx,
        )

    @staticmethod
    def compute_metrics(daily_returns: pd.Series, rf: float = 0.0) -> dict:
        """Compute Sharpe, Sortino, max drawdown, Calmar."""
        r = daily_returns.dropna()
        ann = 252

        mean_r = r.mean() * ann
        std_r = r.std() * np.sqrt(ann)
        sharpe = (mean_r - rf) / std_r if std_r > 0 else np.nan

        downside = r[r < 0].std() * np.sqrt(ann)
        sortino = (mean_r - rf) / downside if downside > 0 else np.nan

        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()

        calmar = mean_r / abs(max_dd) if max_dd != 0 else np.nan

        return {
            "Ann. Return": round(mean_r * 100, 2),
            "Ann. Volatility": round(std_r * 100, 2),
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Max Drawdown": round(max_dd * 100, 2),
            "Calmar": round(calmar, 3),
        }
