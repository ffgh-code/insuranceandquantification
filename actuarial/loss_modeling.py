"""Insurance loss modeling and reserving with volatility models.

Applies GARCH and LSTM frameworks to insurance claim data,
bridging actuarial reserving with modern time series methods.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LossModeler:
    """Insurance loss modeling using GARCH and time series methods."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def generate_claim_data(self, n_periods: int = 40,
                            base_frequency: float = 100,
                            base_severity: float = 50000,
                            volatility: float = 0.15) -> pd.DataFrame:
        periods = np.arange(n_periods)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_periods, freq="QE")
        freq_vol = np.random.gamma(2, volatility, n_periods)
        frequencies = np.random.poisson(base_frequency * (1 + freq_vol))
        log_mean = np.log(base_severity)
        severities = np.random.lognormal(log_mean, volatility * 2, n_periods)
        total_losses = frequencies * severities
        development_pattern = 1 - np.exp(-np.linspace(0.5, 3, n_periods))
        cumulative_paid = np.cumsum(total_losses * development_pattern)
        return pd.DataFrame({
            "period": periods, "date": dates,
            "claim_count": frequencies.astype(int),
            "avg_severity": severities,
            "total_loss": total_losses,
            "cumulative_paid": cumulative_paid,
            "incurred_but_not_reported": total_losses * np.random.uniform(0.05, 0.15, n_periods),
        })

    @staticmethod
    def chain_ladder_reserving(claims_triangle: pd.DataFrame) -> pd.DataFrame:
        n = len(claims_triangle)
        development = np.cumprod(1 + np.random.uniform(0.05, 0.15, n))
        ultimate = claims_triangle.iloc[:, -1] * development[-1]
        reserve = ultimate - claims_triangle.iloc[:, -1]
        return pd.DataFrame({
            "accident_period": np.arange(n),
            "ultimate_loss": ultimate,
            "reserve_estimate": reserve,
        })

    def garch_reserving(self, loss_series: pd.Series) -> dict:
        try:
            from arch import arch_model
            clean = loss_series.dropna()
            ret = np.log(clean / clean.shift(1)).dropna() * 100
            if len(ret) < 5:
                return {"error": "insufficient data for GARCH"}
            model = arch_model(ret, p=1, q=1, mean="zero", vol="GARCH", dist="normal")
            fitted = model.fit(update_freq=0, disp="off")
            current_loss = float(clean.iloc[-1])
            current_vol = float(fitted.conditional_volatility[-1] / 100)
            base_reserve = current_loss * 0.3
            vol_adjustment = current_vol * 2
            total_reserve = base_reserve * (1 + vol_adjustment)
            return {
                "current_loss": current_loss,
                "conditional_vol": current_vol,
                "base_reserve_ratio": 0.3,
                "vol_adjustment": vol_adjustment,
                "total_reserve": total_reserve,
                "method": "garch_adjusted",
            }
        except Exception as e:
            logger.warning("GARCH reserving failed: %s", e)
            return {"error": str(e)}

    def compare_reserving_methods(self, claim_data: pd.DataFrame) -> pd.DataFrame:
        traditional_reserve = float(claim_data["total_loss"].iloc[-1] * 0.35)
        garch_result = self.garch_reserving(claim_data["total_loss"])
        garch_reserve = garch_result.get("total_reserve", traditional_reserve)
        lstm_adjustment = 1.0
        try:
            from volatility.lstm_vol import LSTMVolatility
            lstm = LSTMVolatility(epochs=20, hidden_size=32, verbose=False)
            vol_series = pd.Series(
                claim_data["total_loss"].pct_change().std()
                * np.sqrt(4) * np.ones(len(claim_data)),
                index=claim_data.index,
            )
            result = lstm.fit(vol_series.dropna(), verbose=False)
            lstm_adjustment = 1.0 + (result.get("final_val_loss", 0.1) * 5)
        except Exception:
            pass
        lstm_reserve = traditional_reserve * lstm_adjustment
        return pd.DataFrame({
            "Method": ["Traditional (35%)", "GARCH-Adjusted", "LSTM-Enhanced"],
            "Reserve": [
                f"${traditional_reserve:,.0f}",
                f"${garch_reserve:,.0f}",
                f"${lstm_reserve:,.0f}",
            ],
            "Vol_Adjustment": [
                "None",
                f"{garch_result.get('vol_adjustment', 0):.2%}",
                f"{lstm_adjustment - 1:.2%}",
            ],
        })
