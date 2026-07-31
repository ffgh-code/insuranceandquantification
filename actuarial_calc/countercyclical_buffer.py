"""Countercyclical SCR buffer based on regime-switching volatility.

New module introduced by the IME manuscript revision: addresses C-ROSS
procyclicality with a buffer proportional to the high-volatility regime
probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CountercyclicalSCRBuffer:
    """Countercyclical solvency capital buffer."""

    def __init__(self, eta: float = 0.10):
        self.eta = eta

    @staticmethod
    def filtered_regime_prob(volatility: pd.Series,
                             threshold: float = 0.25) -> pd.Series:
        """Smoothed high-volatility regime probability proxy."""
        high = (volatility > threshold).astype(float)
        return high.ewm(span=20).mean()

    def buffer_scr(self, dynamic_scr: pd.Series,
                   regime_prob: pd.Series) -> pd.Series:
        """SCR with countercyclical buffer: SCR_cc = SCR_dyn * (1 + eta*pi_H)."""
        return dynamic_scr * (1.0 + self.eta * regime_prob)

    def cycle_comparison(self, dynamic_scr: pd.Series,
                         static_scr: float) -> pd.DataFrame:
        """Compare static, dynamic, and buffered SCR across phases."""
        pi = self.filtered_regime_prob(
            dynamic_scr / dynamic_scr.rolling(60).mean().replace(0, np.nan).bfill()
        )
        buffered = self.buffer_scr(dynamic_scr, pi)
        years = pd.Series(pd.DatetimeIndex(dynamic_scr.index).year).values
        rows = []
        for y in sorted(set(years)):
            mask = years == y
            rows.append({
                "year": int(y),
                "static_scr": round(static_scr, 2),
                "dynamic_scr": round(float(dynamic_scr[mask].mean()), 2),
                "buffer_scr": round(float(buffered[mask].mean()), 2),
                "avg_pi_high": round(float(pi[mask].mean()), 3),
            })
        return pd.DataFrame(rows)
