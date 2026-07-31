"""Endogenous sentiment weight estimation.

New module introduced by the IME manuscript revision: jointly estimates
source weights with volatility parameters, testing fixed vs endogenous weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SentimentWeightEstimator:
    """Estimate source weights jointly with the volatility model."""

    def __init__(self, n_sources: int = 3):
        self.n_sources = n_sources
        self.weights = None
        self.fixed_ll = None
        self.endogenous_ll = None

    @staticmethod
    def aggregate(scores: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Weighted sentiment index from source-specific score columns."""
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return scores.values @ w

    def _log_likelihood(self, returns: np.ndarray, sentiment: np.ndarray,
                        params: np.ndarray) -> float:
        omega, alpha, beta, lam = params
        T = len(returns)
        h = np.zeros(T)
        h[0] = np.var(returns)
        for t in range(1, T):
            h[t] = np.exp(
                omega + alpha * (abs(returns[t - 1] / np.sqrt(max(h[t - 1], 1e-8))) - np.sqrt(2 / np.pi))
                + beta * np.log(max(h[t - 1], 1e-8))
                + lam * sentiment[t - 1]
            )
        return -0.5 * np.sum(np.log(2 * np.pi * h) + returns**2 / np.maximum(h, 1e-8))

    def estimate(self, scores: pd.DataFrame, returns: pd.Series,
                 fixed_weights: np.ndarray, n_iter: int = 100) -> dict:
        """Estimate fixed-weight and endogenous-weight models."""
        ret = returns.values
        T = len(ret)

        # Fixed weights
        sent_fixed = self.aggregate(scores, fixed_weights)
        rng = np.random.default_rng(42)
        best_fixed = (-np.inf, None)
        for _ in range(n_iter):
            p = np.array([rng.uniform(0.05, 0.25), rng.uniform(0.10, 0.25),
                          rng.uniform(0.55, 0.80), rng.uniform(0.01, 0.08)])
            ll = self._log_likelihood(ret, sent_fixed, p)
            if ll > best_fixed[0]:
                best_fixed = (ll, p)
        self.fixed_ll = best_fixed[0]

        # Endogenous weights (simplex: w1, w2, w3 = 1 - w1 - w2)
        best_end = (-np.inf, None, None)
        for _ in range(n_iter):
            w1 = rng.uniform(0.2, 0.6)
            w2 = rng.uniform(0.1, 0.5)
            w3 = max(1.0 - w1 - w2, 0.05)
            w = np.array([w1, w2, w3])
            sent_end = self.aggregate(scores, w)
            p = np.array([rng.uniform(0.05, 0.25), rng.uniform(0.10, 0.25),
                          rng.uniform(0.55, 0.80), rng.uniform(0.01, 0.08)])
            ll = self._log_likelihood(ret, sent_end, p)
            if ll > best_end[0]:
                best_end = (ll, w, p)

        self.endogenous_ll = best_end[0]
        self.weights = best_end[1]
        lr = 2.0 * (self.endogenous_ll - self.fixed_ll)

        return {
            "fixed_weights": fixed_weights / fixed_weights.sum(),
            "endogenous_weights": best_end[1],
            "fixed_log_likelihood": self.fixed_ll,
            "endogenous_log_likelihood": self.endogenous_ll,
            "likelihood_ratio": lr,
            "aic_improvement": 2.0 * lr - 2.0,
        }
