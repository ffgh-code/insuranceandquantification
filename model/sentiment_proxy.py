"""Reproducible category sentiment proxy for the IME extension exercises.

The manuscript's primary sentiment source is the LLM-scored Chinese headline
corpus. Because the public corpus is too sparse to cover every trading day in
2018-2026, this module builds a deterministic, market-implied category proxy
that preserves category-specific transmission patterns. It is used only for
the three-regime, category-heterogeneity and MSJ-GARCH demonstration runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CATEGORIES = ["monetary", "industrial", "employment", "geopolitical"]

# Category sensitivities to signed market shocks, calibrated to the relative
# importance of each topic channel in the curated corpus.
DEFAULT_WEIGHTS = {
    "monetary": 0.85,
    "industrial": 0.60,
    "employment": 0.35,
    "geopolitical": 1.15,
}

DEFAULT_GAMMAS = {
    "monetary": 0.10,
    "industrial": 0.30,
    "employment": -0.10,
    "geopolitical": 0.90,
}


def build_category_sentiment(returns: pd.Series, categories=None,
                             weights=None, gammas=None, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic T x K sentiment matrix from market shocks.

    Each category channel is a clipped, lag-augmented function of standardized
    returns plus a small idiosyncratic shock, so the model can identify
    category-specific transmission without pretending to be raw headline data.
    """
    cats = list(categories) if categories else DEFAULT_CATEGORIES
    weights = weights or DEFAULT_WEIGHTS
    gammas = gammas or DEFAULT_GAMMAS
    r = returns.astype(float).to_numpy(dtype=float)
    r_series = pd.Series(r)
    sigma = r_series.rolling(21, min_periods=5).std().to_numpy(dtype=float)
    z = np.divide(
        r,
        sigma,
        out=np.zeros_like(r),
        where=np.isfinite(sigma) & (sigma > 0),
    )
    z_lag = np.concatenate([[0.0], z[:-1]])
    vol_state = np.clip(sigma / np.nanpercentile(sigma, 95), 0.0, 1.0)
    vol_state = np.nan_to_num(vol_state, nan=0.0)
    rng = np.random.default_rng(seed)
    out = {}
    for k, cat in enumerate(cats):
        w = float(weights.get(cat, 0.6))
        gamma = float(gammas.get(cat, 0.2))
        noise = rng.normal(0.0, 0.08, len(z))
        s = w * z + 0.35 * w * z_lag + gamma * z * vol_state + noise
        out[cat] = np.clip(s, -1.0, 1.0)
    return pd.DataFrame(out, index=returns.index)
