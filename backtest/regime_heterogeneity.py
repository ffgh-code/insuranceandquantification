"""Bull-bear heterogeneity tests for sentiment transmission.

New module introduced by the IME manuscript: estimates state-dependent
sentiment coefficients and tests H0: lambda_B = lambda_R = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeHeterogeneityTest:
    """State-dependent sentiment transmission and LR test."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def classify(self, prices: pd.Series) -> pd.Series:
        """Classify bull/bear/range states by 60-day return."""
        ret = prices.pct_change(self.lookback)
        state = pd.Series("range", index=prices.index)
        state[ret > 0.08] = "bull"
        state[ret < -0.05] = "bear"
        return state

    def estimate_state_coefficients(self, returns: pd.Series, sentiment: pd.Series,
                                    state: pd.Series) -> pd.DataFrame:
        """OLS-style state-dependent sentiment coefficients."""
        r = returns.values
        s = sentiment.values
        st = state.values
        rows = []
        for name in ["bull", "bear", "range"]:
            mask = st == name
            if mask.sum() < 20:
                continue
            x = s[mask]
            y = np.abs(r[mask])
            if x.std() > 0:
                coef = np.cov(x, y)[0, 1] / np.var(x)
                se = np.std(y - coef * x) / (np.std(x) * np.sqrt(len(x)))
            else:
                coef, se = 0.0, 0.0
            rows.append({"state": name, "lambda": coef, "se": se})
        return pd.DataFrame(rows)

    def likelihood_ratio(self, restricted_ll: float, unrestricted_ll: float,
                         n_restrictions: int = 2) -> dict:
        """LR test for homogeneous transmission."""
        lr = 2.0 * (unrestricted_ll - restricted_ll)
        # Asymptotic chi-square with n_restrictions degrees of freedom
        from scipy.stats import chi2
        pval = 1.0 - chi2.cdf(lr, n_restrictions)
        return {"lr_stat": lr, "df": n_restrictions, "p_value": pval,
                "reject_homogeneity": pval < 0.05}
