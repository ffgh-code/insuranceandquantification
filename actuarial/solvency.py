"""Solvency II / C-ROSS capital calculation. Uses GARCH-X vol input."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class SolvencyCalculator:
    def __init__(self, capital_base=100_000_000):
        self.capital_base = capital_base
        self._volatility_series = None
    def set_volatility_input(self, vol_series):
        self._volatility_series = vol_series.copy()
    @staticmethod
    def calculate_var(vol, confidence=0.995, horizon_days=1, trading_days=242):
        from scipy.stats import norm
        if vol <= 0: return 0.0
        hv = vol * np.sqrt(horizon_days / trading_days)
        return float(max(-norm.ppf(1 - confidence) * hv, 0.0))
    @staticmethod
    def calculate_cvar(vol, confidence=0.995, horizon_days=1, trading_days=242):
        from scipy.stats import norm
        if vol <= 0: return 0.0
        hv = vol * np.sqrt(horizon_days / trading_days)
        alpha = 1 - confidence
        return float(max(hv * norm.pdf(norm.ppf(confidence)) / alpha, 0.0))
    def compute_market_risk_scr(self):
        if self._volatility_series is None or self._volatility_series.empty:
            return pd.DataFrame()
        vol = self._volatility_series.dropna()
        v995 = vol.apply(lambda v: self.calculate_var(v, 0.995))
        v99 = vol.apply(lambda v: self.calculate_var(v, 0.99))
        return pd.DataFrame({
            "date": vol.index, "conditional_vol": vol.values,
            "var_99.5": v995.values, "var_99": v99.values,
            "scr_solvency_ii": v995.values * self.capital_base,
            "mcr_solvency_ii": v995.values * self.capital_base * 0.3,
            "scr_cross": v99.values * self.capital_base * 0.8,
        })
    def stress_test_extreme_sentiment(self, shock=1.5):
        if self._volatility_series is None: return pd.DataFrame()
        orig = float(self._volatility_series.iloc[-1])
        return pd.DataFrame([{"scenario": "extreme negative shock",
            "shock": shock,
            "original_scr": orig * 0.0325 * self.capital_base,
            "stressed_scr": orig * shock * 0.0325 * self.capital_base}])
    def prepare_chart_data(self, scr_df):
        cols = ["date", "scr_solvency_ii", "mcr_solvency_ii", "scr_cross"]
        return scr_df[[c for c in cols if c in scr_df.columns]].copy()
