"""Category-specific sentiment shock heterogeneity tests.

New module introduced by the IME revision: estimates the transmission of
monetary, industrial, employment and geopolitical news categories separately,
and tests the null hypothesis that all categories share one common sentiment
coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _ols_ll(y: np.ndarray, x: np.ndarray) -> tuple:
    """Gaussian OLS log-likelihood and coefficients."""
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sigma2 = np.mean(resid ** 2) + 1e-12
    ll = -0.5 * len(y) * (np.log(2 * np.pi * sigma2) + 1.0)
    return ll, coef, resid, sigma2


class CategoryHeterogeneityTest:
    """LR test for equality of category-specific sentiment coefficients."""

    def __init__(self, categories=None, lag: int = 1):
        self.categories = list(categories) if categories else [
            "monetary", "industrial", "employment", "geopolitical"
        ]
        self.lag = lag

    def _design(self, returns: pd.Series, sentiment: pd.DataFrame) -> tuple:
        y = returns.abs().to_numpy(dtype=float)
        s = sentiment[self.categories].fillna(0.0).shift(self.lag).to_numpy(dtype=float)
        n = min(len(y), len(s))
        y = y[:n]
        s = s[:n]
        mask = np.isfinite(y) & np.all(np.isfinite(s), axis=1)
        y = y[mask]
        s = s[mask]
        x = np.column_stack([np.ones(len(s)), s])
        return y, x, s

    def estimate(self, returns: pd.Series, sentiment: pd.DataFrame) -> pd.DataFrame:
        """Per-category sentiment coefficients with OLS standard errors."""
        y, x, s = self._design(returns, sentiment)
        _, coef, resid, sigma2 = _ols_ll(y, x)
        xtx_inv = np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        rows = []
        for j, cat in enumerate(self.categories, start=1):
            b = coef[j]
            rows.append({
                "category": cat,
                "coefficient": b,
                "std_error": se[j],
                "t_stat": b / (se[j] + 1e-12),
                "p_value": 1.0 - chi2.cdf((b / (se[j] + 1e-12)) ** 2, 1),
            })
        return pd.DataFrame(rows)

    def likelihood_ratio(self, returns: pd.Series, sentiment: pd.DataFrame) -> dict:
        """LR test of equal category coefficients against a common index."""
        y, x, s = self._design(returns, sentiment)
        ll_unrestricted, _, _, _ = _ols_ll(y, x)
        common = np.mean(s, axis=1)
        x_r = np.column_stack([np.ones(len(common)), common])
        ll_restricted, _, _, _ = _ols_ll(y, x_r)
        df = len(self.categories) - 1
        lr = 2.0 * (ll_unrestricted - ll_restricted)
        pval = 1.0 - chi2.cdf(max(lr, 0.0), df)
        return {
            "restricted_ll": ll_restricted,
            "unrestricted_ll": ll_unrestricted,
            "lr_stat": lr,
            "df": df,
            "p_value": pval,
            "reject_homogeneity": pval < 0.05,
        }

    def single_category_regressions(self, returns: pd.Series,
                                    sentiment: pd.DataFrame) -> pd.DataFrame:
        """Univariate OLS of volatility on each sentiment category separately."""
        y, _, s = self._design(returns, sentiment)
        rows = []
        for j, cat in enumerate(self.categories):
            x = np.column_stack([np.ones(len(s)), s[:, j]])
            _, coef, resid, sigma2 = _ols_ll(y, x)
            xtx_inv = np.linalg.pinv(x.T @ x)
            se = np.sqrt(np.diag(xtx_inv) * sigma2)
            tstat = coef[1] / (se[1] + 1e-12)
            r2 = 1.0 - float(np.var(resid) / (np.var(y) + 1e-12))
            rows.append({
                "category": cat,
                "coefficient": float(coef[1]),
                "std_error": float(se[1]),
                "t_stat": float(tstat),
                "p_value": float(1.0 - chi2.cdf(tstat ** 2, 1)),
                "r_squared": r2,
            })
        return pd.DataFrame(rows)
