"""
Strategy 5: Market-Making Skeleton.
Simulates a single-asset market maker on SPY using:
  - Mid-price from close
  - Fixed half-spread around mid
  - Inventory limit: flip sign when inventory exceeds threshold
  - P&L from captured spreads minus inventory risk
This is a simulation skeleton — real MM requires tick data.
"""
import numpy as np
import pandas as pd

HALF_SPREAD = 0.0005      # 5 bps half-spread
INVENTORY_LIMIT = 10      # max units of inventory before forced unwind
FILL_PROB_BASE = 0.3      # base probability a quote is filled each period
ASSET = "SPY"


def generate_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a weight DataFrame with only SPY column populated.
    Position is ±INVENTORY_LIMIT normalised to [-1, 1].
    Simulates a mean-reverting inventory process.
    """
    if ASSET not in prices.columns:
        raise ValueError(f"{ASSET} not in prices")

    px = prices[ASSET].dropna()
    n = len(px)
    rng = np.random.default_rng(42)

    inventory = 0
    positions = []

    for i in range(n):
        # Quote both sides; fills are stochastic
        buy_fill = rng.random() < FILL_PROB_BASE
        sell_fill = rng.random() < FILL_PROB_BASE

        if buy_fill:
            inventory += 1
        if sell_fill:
            inventory -= 1

        # Inventory risk management: unwind if limit hit
        if inventory > INVENTORY_LIMIT:
            inventory = INVENTORY_LIMIT
        elif inventory < -INVENTORY_LIMIT:
            inventory = -INVENTORY_LIMIT

        # Lean quotes: if long inventory, skew bid down to attract sells
        lean = -0.3 * (inventory / INVENTORY_LIMIT)
        inventory_adj = max(-INVENTORY_LIMIT, min(INVENTORY_LIMIT, inventory + lean * 2))
        positions.append(inventory_adj / INVENTORY_LIMIT)

    pos_series = pd.Series(positions, index=px.index, name=ASSET)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[pos_series.index, ASSET] = pos_series.values
    return weights
