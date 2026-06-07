"""
Performance analytics and reporting.
"""
from typing import Dict
import numpy as np
import pandas as pd
from engine.backtest import BacktestEngine

MONTH_MAP = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}


def full_report(
    strategy_returns: Dict[str, pd.Series],
    combined_returns: pd.Series,
) -> pd.DataFrame:
    rows = {}
    for name, rets in strategy_returns.items():
        rows[name] = BacktestEngine.compute_metrics(rets)
    rows["Combined Portfolio"] = BacktestEngine.compute_metrics(combined_returns)
    return pd.DataFrame(rows).T


def monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    df = monthly.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = [MONTH_MAP[c] for c in pivot.columns]
    return pivot


def correlation_matrix(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(strategy_returns)
    return df.corr()
