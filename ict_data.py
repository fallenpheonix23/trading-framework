"""
Data download and preprocessing for ICT bot.
Downloads 1h data for backtesting and daily data for bias detection.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


def download_hourly(ticker: str = "SPY", period: str = "2y") -> pd.DataFrame:
    """
    Download 1-hour OHLCV data, ensure UTC timezone, clean nulls.
    yfinance max for 1h = 730 days.
    """
    raw = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No hourly data returned for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "datetime"

    # Ensure UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df = df.sort_index()

    print(f"[Data] {ticker} hourly: {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def download_daily(ticker: str = "SPY", period: str = "3y") -> pd.DataFrame:
    """Daily OHLCV for bias detection (need extra history to compute structure on daily)."""
    raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No daily data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).normalize()

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0]
    df = df.sort_index()

    print(f"[Data] {ticker} daily: {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def compute_daily_bias(df_daily: pd.DataFrame, ema_fast: int = 50, ema_slow: int = 200) -> pd.Series:
    """
    Daily bias: 'bull' if 50 EMA > 200 EMA (golden cross), 'bear' otherwise.
    50/200 is far more stable than 20/50 — filters trend regime, not micro-oscillations.
    Returns a Series indexed by date (not datetime), values 'bull' | 'bear'.
    """
    close = df_daily["close"]
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    bias = pd.Series(
        np.where(ema_f > ema_s, "bull", "bear"),
        index=df_daily.index.date,
        name="daily_bias",
    )
    return bias


def download_portfolio_daily(
    tickers: list,
    start: str = "2010-01-01",
    end: str = "2025-01-01",
) -> dict:
    """
    Download 15 years of daily OHLCV for a list of tickers.
    Returns {ticker: df} where each df has lowercase ohlcv columns and a tz-naive DatetimeIndex.
    """
    result = {}
    for ticker in tickers:
        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            progress=False, auto_adjust=True
        )
        if raw.empty:
            print(f"[Data] WARNING: no data for {ticker}")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df["close"] > 0].sort_index()
        result[ticker] = df
        print(f"[Data] {ticker:<10}: {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}")
    return result


def align_bias_to_hourly(df_hourly: pd.DataFrame, daily_bias: pd.Series) -> pd.Series:
    """Align daily bias Series (date index) to the hourly DataFrame."""
    dates = pd.Series(
        [ts.date() for ts in df_hourly.index],
        index=df_hourly.index,
    )
    aligned = dates.map(lambda d: daily_bias.get(d, None))
    aligned.name = "daily_bias"
    return aligned
