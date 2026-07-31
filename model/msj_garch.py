"""Unified regime-switching jump-GARCH model.

New module introduced by the IME revision: estimates regime-specific GARCH
recursions, regime-dependent compound Poisson jump intensities, and a Markov
transition matrix in one joint likelihood, rather than comparing separate
regime-switching and jump-GARCH specifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson

from model.ms_garch import _sentiment_matrix, _stationary_distribution


class RegimeSwitchingJumpGARCH:
    """MS-GARCH-X augmented with regime-dependent compound Poisson jumps.

    Parameter layout:
      omega (n), alpha (n), beta (n), delta (n), lambda (n x K),
      nu (n), muJ (1), sigmaJ (1), transition raw (n x n).
    The conditional density in regime i at time t is a mixture over the number
    of jump arrivals k = 0,...,max_jumps:
      f_i(r_t) = sum_k Poisson(k; nu_i) * N(r_t; k*muJ, h_i,t + k*sigmaJ^2).
    """

    def __init__(self, n_regimes: int = 2, category_names=None, max_jumps: int = 3):
        if n_regimes not in (2, 3):
            raise ValueError("n_regimes must be 2 or 3 in the current implementation.")
        self.n_regimes = n_regimes
        self.category_names = list(category_names) if category_names else None
        self.n_categories = len(self.category_names) if self.category_names else 1
        self.max_jumps = max_jumps
        self.params = None
        self.filtered_probs = None
        self.h = None
        self.transition = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.categories = self.category_names or ["overall"]

    @property
    def regime_names(self):
        if self.n_regimes == 2:
            return ["low_vol", "high_vol"]
        if self.n_regimes == 3:
            return ["calm", "turbulent", "crisis"]
        return [f"regime_{i + 1}" for i in range(self.n_regimes)]

    def _param_slices(self, params):
        n = self.n_regimes
        K = self.n_categories
        omega = params[0:n]
        alpha = params[n:2 * n]
        beta = params[2 * n:3 * n]
        delta = params[3 * n:4 * n]
        lam = params[4 * n:4 * n + n * K].reshape(n, K)
        nu = params[4 * n + n * K:5 * n + n * K]
        muJ = params[5 * n + n * K]
        sigmaJ = params[5 * n + n * K + 1]
        raw = params[5 * n + n * K + 2:].reshape(n, n)
        return omega, alpha, beta, delta, lam, nu, muJ, sigmaJ, raw

    def _transition(self, params):
        _, _, _, _, _, _, _, _, raw = self._param_slices(params)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _variance_paths(self, returns, sentiment, params):
        n = self.n_regimes
        omega, alpha, beta, delta, lam, _, _, _, _ = self._param_slices(params)
        T = len(returns)
        h = np.full((T, n), float(np.var(returns)))
        r = np.asarray(returns, dtype=float)
        S = np.asarray(sentiment, dtype=float)
        for t in range(1, T):
            prev = np.maximum(h[t - 1], 1e-8)
            z = r[t - 1] / np.sqrt(prev)
            neg = np.where(z < 0, np.abs(z), 0.0)
            h[t] = np.exp(
                omega
                + alpha * (np.abs(z) - np.sqrt(2.0 / np.pi))
                + beta * np.log(prev)
                + delta * neg
                + lam @ S[t - 1]
            )
        return np.maximum(h, 1e-8)

    def _jump_density(self, r, h, nu, muJ, sigmaJ):
        """Regime-specific mixture density over jump arrivals."""
        ks = np.arange(self.max_jumps + 1)
        pmf = poisson.pmf(ks, nu)
        var = h + ks * (sigmaJ ** 2)
        sd = np.sqrt(np.maximum(var, 1e-10))
        dens = pmf * norm.pdf(r, loc=ks * muJ, scale=sd)
        return np.maximum(dens.sum(), 1e-300)

    def _filter(self, returns, sentiment, params):
        h = self._variance_paths(returns, sentiment, params)
        _, _, _, _, _, nu, muJ, sigmaJ, _ = self._param_slices(params)
        P = self._transition(params)
        pi0 = _stationary_distribution(P)
        T = len(returns)
        xi = np.zeros((T, self.n_regimes))
        ll = 0.0
        pred = pi0.copy()
        for t in range(T):
            eta = np.array([
                self._jump_density(returns[t], h[t, i], nu[i], muJ, sigmaJ)
                for i in range(self.n_regimes)
            ])
            weighted = pred * eta
            s = weighted.sum()
            if not np.isfinite(s) or s <= 0:
                s = 1e-300
            ll += np.log(s)
            xi[t] = weighted / s
            pred = xi[t] @ P
        return xi, ll, h, P, pi0

    def _log_likelihood(self, returns, sentiment, params):
        _, ll, _, _, _ = self._filter(returns, sentiment, params)
        return ll if np.isfinite(ll) else -np.inf

    def _param_bounds(self):
        n = self.n_regimes
        K = self.n_categories
        bounds = [(-1.0, 1.0)] * n
        bounds += [(1e-4, 0.60)] * n
        bounds += [(1e-4, 0.999)] * n
        bounds += [(-0.60, 0.10)] * n
        bounds += [(-0.30, 0.60)] * (n * K)
        bounds += [(1e-4, 0.60)] * n
        bounds += [(-0.03, 0.03)]
        bounds += [(1e-4, 0.05)]
        bounds += [(-3.0, 3.0)] * (n * n)
        return bounds

    def _random_start(self, rng):
        n = self.n_regimes
        K = self.n_categories
        omega = rng.uniform(-0.30, 0.30, n)
        alpha = rng.uniform(0.05, 0.25, n)
        beta = rng.uniform(0.50, 0.85, n)
        delta = rng.uniform(-0.15, -0.01, n)
        lam = rng.uniform(-0.02, 0.12, n * K)
        nu = rng.uniform(0.01, 0.20, n)
        muJ = rng.uniform(-0.01, 0.01)
        sigmaJ = rng.uniform(0.002, 0.015)
        raw = rng.normal(0.0, 1.0, (n, n))
        raw[np.diag_indices(n)] += 2.5
        return np.concatenate([omega, alpha, beta, delta, lam, nu, [muJ, sigmaJ], raw.ravel()])

    def fit(self, returns, sentiment=None, n_iter: int = 200, max_iter: int = 250,
            seed: int = 42, use_optimizer: bool = False) -> dict:
        """Estimate the unified regime-switching jump-GARCH model.

        By default a structured random search is used; set ``use_optimizer=True``
        for L-BFGS-B restarts on top of the random search.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        sentiment, cats = _sentiment_matrix(sentiment, self.category_names, len(returns))
        self.categories = cats
        self.n_categories = len(cats)
        bounds = self._param_bounds()
        rng = np.random.default_rng(seed)
        best_ll = -np.inf
        best_params = None
        for i in range(n_iter):
            cand = self._random_start(rng)
            if use_optimizer:
                try:
                    res = minimize(
                        lambda x: -self._log_likelihood(returns, sentiment, x),
                        cand,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": max_iter},
                    )
                    cand = res.x
                except Exception:
                    pass
            ll = self._log_likelihood(returns, sentiment, cand)
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_params = cand.copy()
        if best_params is None:
            raise RuntimeError("MSJ-GARCH estimation failed: no finite likelihood found.")
        self.params = best_params
        self.filtered_probs, self.log_likelihood, self.h, self.transition, self.initial_probs = \
            self._filter(returns, sentiment, best_params)
        self.n_params = len(best_params)
        self.aic = -2.0 * self.log_likelihood + 2.0 * self.n_params
        self.bic = -2.0 * self.log_likelihood + self.n_params * np.log(len(returns))
        return {
            "params": best_params,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "filtered_probs": self.filtered_probs,
            "transition": self.transition,
            "jump_intensities": self.jump_intensities(),
            "categories": self.categories,
            "regime_names": self.regime_names,
        }

    def jump_intensities(self) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Fit the model before requesting jump intensities.")
        n = self.n_regimes
        K = self.n_categories
        nu = self.params[4 * n + n * K:5 * n + n * K]
        muJ = self.params[5 * n + n * K]
        sigmaJ = self.params[5 * n + n * K + 1]
        return pd.DataFrame({
            "regime": self.regime_names,
            "jump_intensity": nu,
            "jump_mean": muJ,
            "jump_sd": sigmaJ,
        })
