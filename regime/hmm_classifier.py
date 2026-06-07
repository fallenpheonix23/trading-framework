"""
HMM-based market regime classifier.
Fits a 2-state Gaussian HMM on rolling realized volatility and returns.
State 0 = low-vol / choppy (mean reversion favoured, trend disabled).
State 1 = trending / high-vol  (trend following favoured, mean reversion disabled).
"""
from typing import Optional, Dict
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeClassifier:
    def __init__(self, n_states: int = 2, n_iter: int = 200, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self._state_map: Dict[int, str] = {}

    def _build_features(self, returns: pd.Series) -> np.ndarray:
        vol = returns.rolling(21).std().bfill()
        return np.column_stack([returns.values, vol.values])

    def fit(self, train_returns: pd.Series) -> "RegimeClassifier":
        """Fit HMM on train data only. Call predict() separately for full period."""
        r = train_returns.dropna()
        features = self._build_features(r)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(features)
        self.model = model

        # Identify which state is "trending" on train data
        states = model.predict(features)
        state_abs_ret = {
            s: np.abs(r.values[states == s]).mean() for s in range(self.n_states)
        }
        trending_state = max(state_abs_ret, key=state_abs_ret.get)
        self._state_map = {
            s: ("trending" if s == trending_state else "choppy")
            for s in range(self.n_states)
        }
        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        """Apply fitted HMM to any period (train or full). Must call fit() first."""
        r = returns.dropna()
        features = self._build_features(r)
        states = self.model.predict(features)
        regime = pd.Series(states, index=r.index, name="regime")
        return regime.map(lambda x: 1 if self._state_map.get(x, "choppy") == "trending" else 0)

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Convenience: fit and predict on same series (legacy / in-sample use)."""
        return self.fit(returns).predict(returns)

    @property
    def state_labels(self) -> Dict:
        return self._state_map


def apply_regime_filter(
    weights: pd.DataFrame,
    regime: pd.Series,
    allowed_regime: int,
) -> pd.DataFrame:
    """Zero-out weights on days where regime != allowed_regime."""
    mask = regime.reindex(weights.index).fillna(allowed_regime)
    weights = weights.copy()
    weights.loc[mask != allowed_regime] = 0.0
    return weights
