"""
Shared data pipeline — all data sourced from yfinance only.
Downloads are cached to parquet to avoid repeated network calls.
Equity universe: full S&P 500 current constituents (scraped from Wikipedia).
"""
import warnings
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from pathlib import Path
from io import StringIO
from tqdm import tqdm

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

START = "2010-01-01"
END   = "2024-12-31"

MACRO_TICKERS = ["TLT", "SHY", "IEF", "TIP", "GLD", "SPY",
                 "HYG", "LQD", "EFA", "UUP",
                 "^VIX", "^TNX", "^IRX"]

# Always-include tickers needed by specific strategies
REQUIRED_TICKERS = ["SPY", "QQQ", "IWM"]


def fetch_sp500_tickers() -> List[str]:
    """
    Scrape current S&P 500 constituents from Wikipedia.
    Falls back to the cached list in sp500_tickers.txt if unavailable.
    """
    txt_cache = Path(__file__).parent.parent / "sp500_tickers.txt"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=10,
        )
        tbl = pd.read_html(StringIO(r.text))[0]
        tickers = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()
        # Cache to file for offline use
        txt_cache.write_text("\n".join(tickers))
        return tickers
    except Exception:
        if txt_cache.exists():
            return txt_cache.read_text().strip().splitlines()
        raise RuntimeError("Cannot fetch S&P 500 tickers and no local cache found.")


def get_equity_universe() -> List[str]:
    tickers = fetch_sp500_tickers()
    # Merge with always-required tickers
    for t in REQUIRED_TICKERS:
        if t not in tickers:
            tickers.append(t)
    return tickers


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _download(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices, return wide DataFrame."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices


def load_equity_prices(
    tickers: Optional[List[str]] = None,
    start: str = START,
    end: str = END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache = _cache_path("sp500_equity_prices")

    if cache.exists() and not force_refresh:
        df = pd.read_parquet(cache)
        if tickers:
            available = [t for t in tickers if t in df.columns]
            return df[available]
        return df

    universe = tickers or get_equity_universe()
    print(f"Downloading S&P 500 universe ({len(universe)} tickers)...")

    chunk_size = 50
    chunks = []
    for i in tqdm(range(0, len(universe), chunk_size)):
        batch = universe[i : i + chunk_size]
        try:
            chunk = _download(batch, start, end)
            chunks.append(chunk)
        except Exception as e:
            print(f"  Warning: batch {i//chunk_size+1} failed ({e}), skipping")

    prices = pd.concat(chunks, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.to_parquet(cache)
    print(f"  Downloaded {prices.shape[1]} tickers with data.")
    return prices


def load_macro_data(
    start: str = START,
    end: str = END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache = _cache_path("macro_data")
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)
    print("Downloading macro/fixed-income data...")
    prices = _download(MACRO_TICKERS, start, end)
    prices.to_parquet(cache)
    return prices


def compute_returns(prices: pd.DataFrame, fill_limit: int = 5) -> pd.DataFrame:
    filled = prices.ffill(limit=fill_limit)
    rets = np.log(filled / filled.shift(1))
    return rets.dropna(how="all")


def align_dates(*frames: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    return tuple(f.loc[idx] for f in frames)
