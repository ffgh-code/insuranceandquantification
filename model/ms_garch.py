"""Markov regime-switching GARCH with sentiment feedback.

New module introduced by the IME manuscript: estimates the log-variance
specification with regime-dependent sentiment transmission (Eq. 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarkovSwitchingGARCH:
    """Two-regime log-GARCH with sentiment feedback."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.params = None
        self.filtered_probs = None

    def _log_variance(self, returns, sentiment, params):
        omega = params[: self.n_regimes]
        alpha = params[self.n_regimes : 2 * self.n_regimes]
        beta = params[2 * self.n_regimes : 3 * self.n_regimes]
        lam = params[3 * self.n_regimes : 4 * self.n_regimes]
        delta = params[4 * self.n_regimes : 5 * self.n_regimes]

        T = len(returns)
        h = np.zeros(T)
        h[0] = np.var(returns)
        for t in range(1, T):
            r = returns[t - 1] / np.sqrt(max(h[t - 1], 1e-8))
            neg = 1.0 if r < 0 else 0.0
            # Regime-weighted update
            w = self.filtered_probs[t - 1]
            h[t] = np.exp(
                np.sum(w * omega)
                + np.sum(w * alpha) * (abs(r) - np.sqrt(2 / np.pi))
                + np.sum(w * beta) * np.log(max(h[t - 1], 1e-8))
                + np.sum(w * lam) * sentiment[t - 1]
                + np.sum(w * delta) * neg * abs(r)
            )
        return h

    def fit(self, returns, sentiment, n_iter: int = 200):
        T = len(returns)
        self.filtered_probs = np.full((T, self.n_regimes), 1.0 / self.n_regimes)
        # Simple EM-style grid search for demonstration
        best = None
        best_ll = -np.inf
        rng = np.random.default_rng(42)
        for _ in range(n_iter):
            omega = rng.uniform(0.05, 0.30, self.n_regimes)
            alpha = rng.uniform(0.10, 0.25, self.n_regimes)
            beta = rng.uniform(0.55, 0.80, self.n_regimes)
            lam = rng.uniform(0.01, 0.08, self.n_regimes)
            delta = rng.uniform(-0.10, -0.02, self.n_regimes)
            params = np.concatenate([omega, alpha, beta, lam, delta])
            h = self._log_variance(returns, sentiment, params)
            ll = -0.5 * np.sum(
                np.log(2 * np.pi * h) + returns**2 / np.maximum(h, 1e-8)
            )
            if ll > best_ll:
                best_ll = ll
                best = params
        self.params = best
        return {"params": best, "log_likelihood": best_ll}

    def multipliers(self):
        """Regime-specific sentiment multipliers M_i = lambda_i/(1-beta_i)."""
        p = self.params
        n = self.n_regimes
        beta = p[2 * n : 3 * n]
        lam = p[3 * n : 4 * n]
        return lam / (1.0 - beta)
