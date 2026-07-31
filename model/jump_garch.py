"""Jump-GARCH comparison model.

New module introduced by the IME manuscript revision: compound Poisson
jump component versus regime-switching GARCH, for model completeness tests.
"""

from __future__ import annotations

import numpy as np


class JumpGARCH:
    """GARCH with compound Poisson jumps."""

    def __init__(self, intensity: float = 0.05):
        self.intensity = intensity

    def simulate_jumps(self, n: int, seed: int = 42) -> np.ndarray:
        """Simulate compound Poisson jump realizations."""
        rng = np.random.default_rng(seed)
        n_jumps = rng.poisson(self.intensity, n)
        jumps = np.zeros(n)
        for t, nj in enumerate(n_jumps):
            if nj > 0:
                jumps[t] = rng.normal(0.0, 0.02, nj).sum()
        return jumps

    @staticmethod
    def log_likelihood(returns: np.ndarray, params: np.ndarray) -> float:
        """Gaussian log-likelihood with jump-augmented variance."""
        omega, alpha, beta, jump_var = params
        T = len(returns)
        h = np.var(returns)
        ll = 0.0
        for t in range(1, T):
            h = np.exp(omega + alpha * (returns[t - 1] ** 2 / max(h, 1e-8)) + beta * np.log(max(h, 1e-8)))
            h += jump_var
            ll += -0.5 * (np.log(2 * np.pi * h) + returns[t] ** 2 / max(h, 1e-8))
        return ll

    def estimate(self, returns: np.ndarray, n_iter: int = 100) -> dict:
        """Estimate jump-GARCH parameters by random search."""
        rng = np.random.default_rng(42)
        best = (-np.inf, None)
        for _ in range(n_iter):
            p = np.array([rng.uniform(0.05, 0.25), rng.uniform(0.05, 0.30),
                          rng.uniform(0.50, 0.85), rng.uniform(1e-4, 1e-3)])
            ll = self.log_likelihood(returns, p)
            if ll > best[0]:
                best = (ll, p)
        aic = -2.0 * best[0] + 2.0 * 4
        bic = -2.0 * best[0] + 4 * np.log(len(returns))
        return {"log_likelihood": best[0], "aic": aic, "bic": bic, "params": best[1]}
