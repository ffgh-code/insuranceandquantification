"""GARCH family models for volatility forecasting.

Implements standard GARCH, EGARCH, and GJR-GARCH models using
the `arch` library, with support for exogenous variables (sentiment).
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GARCHModel:
    """GARCH family volatility models with exogenous variable support.

    Supports standard GARCH(p,q), EGARCH, and GJR-GARCH specifications,
    with optional exogenous regressors (e.g., sentiment scores).
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model_type: Literal["GARCH", "EGARCH", "GJR-GARCH"] = "GARCH",
        distribution: Literal["normal", "studentst", "skewstudent"] = "studentst",
    ):
        self.p = p
        self.q = q
        self.model_type = model_type
        self.distribution = distribution
        self._fitted_model = None
        self._residuals: Optional[pd.Series] = None

    def _map_distribution(self):
        """Map distribution string to arch library parameter."""
        mapping = {
            "normal": "Normal",
            "studentst": "StudentsT",
            "skewstudent": "SkewStudent",
        }
        return mapping.get(self.distribution, "StudentsT")

    def fit(
        self,
        returns: pd.Series,
        exogenous: Optional[pd.DataFrame] = None,
        forecast_horizon: int = 5,
    ) -> dict:
        """Fit the GARCH model to return series with optional exogenous variables.

        Args:
            returns: Asset return series (daily).
            exogenous: DataFrame of exogenous variables indexed by date.
            forecast_horizon: Number of steps for out-of-sample forecast.

        Returns:
            Dict with fitted model results.
        """
        try:
            from arch import arch_model
        except ImportError:
            logger.error("arch package not installed. Run: pip install arch")
            raise

        # Align returns and exogenous
        if exogenous is not None:
            combined = pd.concat([returns, exogenous], axis=1).dropna()
            ret = combined.iloc[:, 0]
            exog = combined.iloc[:, 1:]
        else:
            ret = returns.dropna()
            exog = None

        # Build model
        model = arch_model(
            ret * 100,  # Scale to percentage for numerical stability
            p=self.p,
            q=self.q,
            o=(1 if self.model_type == "GJR-GARCH" else 0),
            power=(1.0 if self.model_type == "EGARCH" else 2.0),
            dist=self._map_distribution(),
            x=exog,
            mean="zero",
            vol=self.model_type if self.model_type == "EGARCH" else "GARCH",
        )

        try:
            fitted = model.fit(update_freq=0, disp="off")
            self._fitted_model = fitted
            self._residuals = pd.Series(
                fitted.resid / 100, index=ret.index, name="residuals"
            )

            # Generate conditional volatility (already annualized by arch)
            cond_vol = fitted.conditional_volatility / 100

            # Forecast
            forecasts = fitted.forecast(horizon=forecast_horizon)
            forecast_vol = np.sqrt(forecasts.variance.iloc[-1] / 10000)

            return {
                "model": fitted,
                "conditional_volatility": pd.Series(np.asarray(cond_vol).ravel(), index=ret.index),
                "forecast_volatility": forecast_vol.values,
                "params": dict(fitted.params),
                "aic": fitted.aic,
                "bic": fitted.bic,
                "loglikelihood": fitted.loglikelihood,
            }
        except Exception as e:
            logger.warning("GARCH model fitting failed: %s", e)
            return {
                "model": None,
                "conditional_volatility": pd.Series(dtype=float),
                "forecast_volatility": np.array([]),
                "params": {},
                "aic": np.nan,
                "bic": np.nan,
                "loglikelihood": np.nan,
            }

    def fit_with_sentiment(
        self,
        returns: pd.Series,
        sentiment_scores: pd.Series,
        forecast_horizon: int = 5,
    ) -> dict:
        """Fit GARCH-X model with sentiment as exogenous variable.

        This is the key method that combines sentiment and volatility modeling.

        Args:
            returns: Asset return series.
            sentiment_scores: Sentiment score series (aligned by date).
            forecast_horizon: Forecast horizon.

        Returns:
            Dict with fitted model results.
        """
        exog = sentiment_scores.to_frame("sentiment")
        result = self.fit(returns, exogenous=exog, forecast_horizon=forecast_horizon)

        # Extract sentiment coefficient significance
        if result["model"] is not None:
            try:
                # The exogenous variable coefficient
                exog_coef = result["params"].get("beta[1]", None)
                if exog_coef is None:
                    # Try alternative naming in arch library
                    exog_coef = result["params"].get("alpha[1]", None)
                result["sentiment_coefficient"] = exog_coef
            except Exception:
                result["sentiment_coefficient"] = None

        return result

    def summary(self) -> Optional[pd.DataFrame]:
        """Return model coefficients as a DataFrame for display.

        Returns:
            DataFrame of model coefficients or None if not fitted.
        """
        if self._fitted_model is None:
            return None
        try:
            return pd.DataFrame(
                {
                    "coefficient": self._fitted_model.params,
                    "std_error": self._fitted_model.std_err,
                    "p_value": self._fitted_model.pvalues,
                }
            )
        except Exception:
            return None

    @staticmethod
    def compare_models(
        returns: pd.Series,
        sentiment: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compare GARCH, EGARCH, and GJR-GARCH models.

        Args:
            returns: Return series.
            sentiment: Optional sentiment series for GARCH-X comparison.

        Returns:
            DataFrame comparing model AIC, BIC, and log-likelihood.
        """
        results = []
        for model_type in ["GARCH", "EGARCH", "GJR-GARCH"]:
            for use_sentiment in [False, True] if sentiment is not None else [False]:
                model = GARCHModel(p=1, q=1, model_type=model_type)
                if use_sentiment and sentiment is not None:
                    result = model.fit_with_sentiment(returns, sentiment)
                else:
                    result = model.fit(returns)

                results.append(
                    {
                        "model": f"{model_type}{'-X' if use_sentiment else ''}",
                        "aic": result["aic"],
                        "bic": result["bic"],
                        "loglikelihood": result["loglikelihood"],
                    }
                )
        return pd.DataFrame(results).sort_values("aic")



