"""Monte Carlo stress testing of volatility under adverse sentiment scenarios.

New module introduced by the IME manuscript: simulates volatility paths
under exponential sentiment shocks and computes tail VaR depletion.
"""

from __future__ import annotations

import numpy as np


class MonteCarloStressTest:
    """Monte Carlo stress testing for sentiment-driven volatility."""

    def __init__(self, n_paths: int = 100_000, horizon: int = 63, seed: int = 42):
        self.n_paths = n_paths
        self.horizon = horizon
        self.seed = seed

    def simulate(
        self, omega: float, alpha: float, beta: float, lam: float,
        psi: float = 0.0, base_vol: float = 0.20
    ):
        """Simulate volatility paths under stressed sentiment shocks.

        Args:
            omega, alpha, beta, lam: log-GARCH parameters.
            psi: stress intensity (0 = baseline).
            base_vol: initial annualized volatility.
        """
        rng = np.random.default_rng(self.seed)
        T = self.horizon
        N = self.n_paths
        log_h = np.full((N, T), np.log(base_vol**2 / 252))
        for t in range(1, T):
            shock = rng.standard_normal(N)
            sent_shock = psi * rng.exponential(1.0, N)
            log_h[:, t] = (
                omega
                + alpha * (abs(shock) - np.sqrt(2 / np.pi))
                + beta * log_h[:, t - 1]
                + lam * sent_shock
            )
        h = np.exp(log_h)
        # Cumulative quarterly variance and VaR
        cum_var = h.sum(axis=1)
        var_995 = 2.576 * np.sqrt(cum_var)
        var_99 = 2.326 * np.sqrt(cum_var)
        return {
            "median_var_995": float(np.median(var_995)),
            "p95_var_995": float(np.percentile(var_995, 95)),
            "mean_depletion_ratio": float(var_995.mean() / np.median(var_995)),
            "median_var_99": float(np.median(var_99)),
        }

    def stress_scenarios(self, params: dict, psi_list=(0.0, 1.0, 2.0)):
        """Run stress scenarios across shock intensities."""
        out = {}
        base = None
        for psi in psi_list:
            r = self.simulate(
                params["omega"], params["alpha"], params["beta"],
                params["lambda"], psi=psi
            )
            if psi == 0.0:
                base = r["median_var_995"]
            r["depletion_ratio"] = r["median_var_995"] / base
            out[psi] = r
        return out
