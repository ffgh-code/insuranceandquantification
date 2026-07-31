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
