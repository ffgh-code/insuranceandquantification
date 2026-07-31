"""Sentiment shock transmission: theoretical multipliers and impulse responses.

New module introduced by the IME manuscript: implements Proposition 1 and 2,
the sentiment multiplier M = lambda/(1-beta), and regime impulse responses.
"""

from __future__ import annotations

import numpy as np


class SentimentShockTransmission:
    """Structural sentiment-to-volatility transmission."""

    @staticmethod
    def multiplier(lam: float, beta: float) -> float:
        """Equilibrium sentiment multiplier M = lambda / (1 - beta)."""
        return lam / (1.0 - beta) if beta < 1.0 else np.inf

    @staticmethod
    def impulse_response(lam: float, beta: float, horizon: int = 21) -> np.ndarray:
        """Cumulative log-variance response to a unit sentiment shock."""
        return np.array([lam * sum(beta**j for j in range(t + 1)) for t in range(horizon)])

    @staticmethod
    def regime_response(lam: np.ndarray, beta: np.ndarray, horizon: int = 21) -> dict:
        """Impulse responses by regime."""
        return {
            i: SentimentShockTransmission.impulse_response(lam[i], beta[i], horizon)
            for i in range(len(lam))
        }

    @staticmethod
    def transmission_difference(lam, beta, horizon=21):
        """Difference in impulse responses between regimes (t-statistic style)."""
        r0 = SentimentShockTransmission.impulse_response(lam[0], beta[0], horizon)
        r1 = SentimentShockTransmission.impulse_response(lam[1], beta[1], horizon)
        diff = r1 - r0
        return diff, diff / (np.std(diff) + 1e-8)

    @staticmethod
    def category_multipliers(lam, beta, categories=None):
        """Multiplier table for regime x sentiment-category channels.

        ``lam`` is a (n_regimes, n_categories) matrix and ``beta`` a vector of
        regime persistence parameters. Returns a pandas DataFrame.
        """
        import pandas as pd
        lam = np.asarray(lam, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if categories is None:
            categories = [f"cat{k + 1}" for k in range(lam.shape[1])]
        m = lam / (1.0 - beta[:, None])
        return pd.DataFrame(m, columns=list(categories))

    @staticmethod
    def pairwise_differences(lam, beta, horizon=21, regime_names=None):
        """Pairwise impulse-response differences for two or three regimes."""
        import pandas as pd
        lam = np.asarray(lam, dtype=float)
        beta = np.asarray(beta, dtype=float)
        n = len(lam)
        if regime_names is None:
            if n == 2:
                regime_names = ["low_vol", "high_vol"]
            elif n == 3:
                regime_names = ["calm", "turbulent", "crisis"]
            else:
                regime_names = [f"regime_{i + 1}" for i in range(n)]
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                ri = SentimentShockTransmission.impulse_response(lam[i], beta[i], horizon)
                rj = SentimentShockTransmission.impulse_response(lam[j], beta[j], horizon)
                d = rj - ri
                rows.append({
                    "pair": f"{regime_names[j]}-{regime_names[i]}",
                    "horizon_1": d[0],
                    "horizon_5": d[4] if horizon >= 5 else np.nan,
                    "horizon_10": d[9] if horizon >= 10 else np.nan,
                    "horizon_21": d[-1],
                    "std": float(np.std(d)),
                })
        return pd.DataFrame(rows)
