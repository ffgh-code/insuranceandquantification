"""ARIMA model for volatility forecasting baseline."""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ARIMAModel:
    """ARIMA(p,d,q) for volatility forecasting as baseline."""

    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        self.p, self.d, self.q = p, d, q
        self._fitted = None
        self._aic = None
        self._bic = None

    def fit(self, series: pd.Series) -> dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series.dropna(), order=(self.p, self.d, self.q))
            fitted = model.fit()
            self._fitted = fitted
            self._aic = fitted.aic
            self._bic = fitted.bic
            resid = pd.Series(fitted.resid, index=series.dropna().index)
            return {
                "aic": fitted.aic, "bic": fitted.bic,
                "params": dict(fitted.params),
                "residuals": resid,
                "forecast": fitted.forecast(steps=5).values,
                "model": fitted,
            }
        except Exception as e:
            logger.warning("ARIMA fit failed: %s", e)
            return {"aic": np.nan, "bic": np.nan, "params": {}, "residuals": pd.Series(dtype=float)}

    @staticmethod
    def residual_tests(residuals: pd.Series) -> dict:
        """Ljung-Box test for residual autocorrelation."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        try:
            lb = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
            pval = lb["lb_pvalue"].iloc[0] if len(lb) > 0 else 0
            return {"ljung_box_pvalue": float(pval), "white_noise": pval > 0.05}
        except Exception as e:
            return {"ljung_box_pvalue": 0.0, "error": str(e)}
