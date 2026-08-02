"""Markov regime-switching GARCH with sentiment feedback.

Implements the IME manuscript volatility engine:
- two- or three-regime log-GARCH with regime-specific variance recursions;
- category-specific sentiment channels (monetary, industrial, employment,
  geopolitical or any user-supplied columns);
- Hamilton-filtered regime probabilities and a softmax transition matrix;
- regime-specific sentiment multipliers M_{i,k} = lambda_{i,k}/(1-beta_i).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def _sentiment_matrix(sentiment, categories, length):
    """Normalize a Series/DataFrame/array to a T x K sentiment matrix."""
    if sentiment is None:
        mat = np.zeros((length, 1), dtype=float)
        cats = ["overall"]
    elif isinstance(sentiment, pd.DataFrame):
        cats = [str(c) for c in sentiment.columns]
        mat = sentiment.fillna(0.0).to_numpy(dtype=float)
    else:
        arr = np.asarray(sentiment, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.ndim == 1:
            mat = arr.reshape(-1, 1)
            cats = categories or ["overall"]
        else:
            mat = arr
            cats = categories or [f"cat{k + 1}" for k in range(arr.shape[1])]
    if mat.shape[0] < length:
        pad = np.zeros((length - mat.shape[0], mat.shape[1]), dtype=float)
        mat = np.vstack([mat, pad])
    return mat[:length], cats


def _stationary_distribution(P):
    """Stationary vector pi such that pi = pi @ P."""
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


class MarkovSwitchingGARCH:
    """Regime-switching log-GARCH with sentiment feedback.

    Parameters are ordered as:
      omega (n), alpha (n), beta (n), delta (n),
      lambda (n x K, flattened), transition raw matrix (n x n).
    The transition matrix is the row-softmax of the raw matrix, so every row
    is a valid probability vector without additional constraints.
    """

    def __init__(self, n_regimes: int = 2, category_names=None):
        if n_regimes not in (2, 3):
            raise ValueError("n_regimes must be 2 or 3 in the current implementation.")
        self.n_regimes = n_regimes
        self.category_names = list(category_names) if category_names else None
        self.n_categories = len(self.category_names) if self.category_names else 1
        self.params = None
        self.filtered_probs = None
        self.h = None
        self.transition = None
        self.initial_probs = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.n_params = None
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
        raw = params[4 * n + n * K:].reshape(n, n)
        return omega, alpha, beta, delta, lam, raw

    def _transition(self, params):
        _, _, _, _, _, raw = self._param_slices(params)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _variance_paths(self, returns, sentiment, params):
        n = self.n_regimes
        omega, alpha, beta, delta, lam, _ = self._param_slices(params)
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

    def _filter(self, returns, sentiment, params):
        """Hamilton filter over regime-specific variance paths."""
        h = self._variance_paths(returns, sentiment, params)
        P = self._transition(params)
        pi0 = _stationary_distribution(P)
        T = len(returns)
        xi = np.zeros((T, self.n_regimes))
        ll = 0.0
        pred = pi0.copy()
        sd = np.sqrt(np.maximum(h, 1e-8))
        for t in range(T):
            eta = norm.pdf(returns[t], loc=0.0, scale=sd[t])
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
        raw = rng.normal(0.0, 1.0, (n, n))
        raw[np.diag_indices(n)] += 2.5
        return np.concatenate([omega, alpha, beta, delta, lam, raw.ravel()])

    def fit(self, returns, sentiment=None, n_iter: int = 200, max_iter: int = 300,
            seed: int = 42, use_optimizer: bool = False, polish: bool = False) -> dict:
        """Estimate the regime-switching GARCH.

        By default a structured random search is used, matching the lightweight
        estimator pattern of the project. Set ``use_optimizer=True`` to run
        L-BFGS-B restarts on top of the random search (slower but more precise).
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
        if polish and best_params is not None:
            try:
                res = minimize(
                    lambda x: -self._log_likelihood(returns, sentiment, x),
                    best_params,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter},
                )
                cand = res.x
                ll = self._log_likelihood(returns, sentiment, cand)
                if np.isfinite(ll) and ll > best_ll:
                    best_ll = ll
                    best_params = cand
            except Exception:
                pass
        if best_params is None:
            raise RuntimeError("MS-GARCH estimation failed: no finite likelihood found.")
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
            "categories": self.categories,
            "regime_names": self.regime_names,
        }

    def multipliers(self):
        """Regime/category sentiment multipliers M = lambda/(1-beta)."""
        if self.params is None:
            raise RuntimeError("Fit the model before requesting multipliers.")
        n = self.n_regimes
        K = self.n_categories
        beta = self.params[2 * n:3 * n]
        lam = self.params[4 * n:4 * n + n * K].reshape(n, K)
        m = lam / (1.0 - beta[:, None])
        if K == 1:
            return m.ravel()
        return pd.DataFrame(m, index=self.regime_names, columns=self.categories)

    def transition_matrix(self) -> pd.DataFrame:
        if self.transition is None:
            raise RuntimeError("Fit the model before requesting the transition matrix.")
        return pd.DataFrame(self.transition, index=self.regime_names, columns=self.regime_names)

    def filtered_probabilities(self, index=None) -> pd.DataFrame:
        if self.filtered_probs is None:
            raise RuntimeError("Fit the model before requesting filtered probabilities.")
        return pd.DataFrame(self.filtered_probs, index=index, columns=self.regime_names)

    def classify_regimes(self, threshold: float = 0.5) -> np.ndarray:
        if self.filtered_probs is None:
            raise RuntimeError("Fit the model before classifying regimes.")
        idx = np.argmax(self.filtered_probs, axis=1)
        if threshold < 1.0:
            maxp = self.filtered_probs.max(axis=1)
            idx = np.where(maxp < threshold, -1, idx)
        return idx

    def forecast_variance(self, returns, sentiment=None, horizon: int = 1) -> float:
        """One-day-ahead regime-weighted variance forecast."""
        if self.params is None:
            raise RuntimeError("Fit the model before forecasting.")
        returns = np.asarray(returns, dtype=float).ravel()
        sentiment, _ = _sentiment_matrix(sentiment, self.categories, len(returns))
        h = self._variance_paths(returns, sentiment, self.params)
        omega, alpha, beta, delta, lam, _ = self._param_slices(self.params)
        probs = self.filtered_probs[-1]
        prev = np.maximum(h[-1], 1e-8)
        z = returns[-1] / np.sqrt(prev)
        neg = np.where(z < 0, np.abs(z), 0.0)
        h_next = np.exp(
            omega
            + alpha * (np.abs(z) - np.sqrt(2.0 / np.pi))
            + beta * np.log(prev)
            + delta * neg
            + lam @ sentiment[-1]
        )
        var = float(probs @ h_next)
        for _ in range(1, horizon):
            prev = np.maximum(h_next, 1e-8)
            h_next = np.exp(omega + beta * np.log(prev) + lam @ sentiment[-1])
            var = float(probs @ h_next)
        return var
