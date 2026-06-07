"""
Walk-forward validation engine.
Train: 2015-2019. Test (OOS): 2020-2024.

Prevents look-ahead by ensuring:
  - HMM is fitted on train returns, then predicted over full period
  - Stat arb pairs and hedge ratios are selected on train only
  - All other strategies use rolling lookbacks that are naturally causal
"""
import pandas as pd
import numpy as np

TRAIN_END  = "2015-12-31"
TEST_START = "2016-01-01"


class WalkForwardSplit:
    def __init__(self, train_end: str = TRAIN_END, test_start: str = TEST_START):
        self.train_end  = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)

    def split(self, df: pd.DataFrame):
        """Return (train_df, test_df) with no overlap."""
        train = df[df.index <= self.train_end]
        test  = df[df.index >= self.test_start]
        return train, test

    def is_test(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(index >= self.test_start, index=index)

    def metrics_on_test(
        self,
        daily_returns: pd.Series,
        compute_fn,
    ) -> dict:
        """Slice to test period and compute metrics."""
        test_rets = daily_returns[daily_returns.index >= self.test_start]
        return compute_fn(test_rets)
