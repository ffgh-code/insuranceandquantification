"""Solvency II / C-ROSS Solvency Capital Requirement calculation."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
logger = logging.getLogger(__name__)

class SolvencyCalculator:
    """Solvency capital requirement calculator using volatility models."""

    SCR_CONFIDENCE = 0.995
    MCR_CONFIDENCE = 0.85
    CROSS_MCR_MULTIPLIER = 0.3

    def __init__(self, capital_base: float = 100_000_000):
        self.capital_base = capital_base

    @staticmethod
    def calculate_var(volatility: float, confidence: float = 0.995,
                      horizon_days: int = 1, trading_days: int = 252) -> float:
        if volatility <= 0:
            return 0.0
        horizon_vol = volatility * np.sqrt(horizon_days / trading_days)
        var = -norm.ppf(1 - confidence) * horizon_vol
        return float(max(var, 0.0))

    @staticmethod
    def calculate_cvar(volatility: float, confidence: float = 0.995,
                       horizon_days: int = 1, trading_days: int = 252) -> float:
        if volatility <= 0:
            return 0.0
        horizon_vol = volatility * np.sqrt(horizon_days / trading_days)
        alpha = 1 - confidence
        cvar = horizon_vol * norm.pdf(norm.ppf(confidence)) / alpha
        return float(max(cvar, 0.0))

    def scratch_market_risk(self, volatility_series: pd.Series,
                            confidence: float = 0.995) -> pd.DataFrame:
        var_series = volatility_series.apply(
            lambda v: self.calculate_var(v, confidence))
        cvar_series = volatility_series.apply(
            lambda v: self.calculate_cvar(v, confidence))
        scr = var_series * self.capital_base
        mcr = scr * 0.3
        return pd.DataFrame({
            "volatility": volatility_series,
            "var_99.5": var_series,
            "cvar_99.5": cvar_series,
            "scr_market": scr,
            "mcr_market": mcr,
        })

    def scratch_cross(self, volatility_series: pd.Series) -> pd.DataFrame:
        var_90 = volatility_series.apply(lambda v: self.calculate_var(v, 0.90))
        var_95 = volatility_series.apply(lambda v: self.calculate_var(v, 0.95))
        var_99 = volatility_series.apply(lambda v: self.calculate_var(v, 0.99))
        scr = var_99 * self.capital_base * 0.8
        mcr = scr * 0.3
        return pd.DataFrame({
            "volatility": volatility_series,
            "var_90": var_90, "var_95": var_95, "var_99": var_99,
            "scr_cross": scr, "mcr_cross": mcr,
        })

    @staticmethod
    def capital_adequacy_ratio(available_capital: float, scr: float) -> float:
        return (available_capital / scr) * 100 if scr > 0 else float("inf")

    def generate_report(self, volatility: pd.Series,
                        available_capital: float = None) -> dict:
        market_risk = self.scratch_market_risk(volatility)
        cross = self.scratch_cross(volatility)
        latest_scr = float(market_risk["scr_market"].iloc[-1])
        report = {
            "current_vol": float(volatility.iloc[-1]),
            "latest_scr": latest_scr,
            "latest_mcr": float(market_risk["mcr_market"].iloc[-1]),
            "peak_scr": float(market_risk["scr_market"].max()),
            "mean_scr": float(market_risk["scr_market"].mean()),
            "solvency_ii_var": float(market_risk["var_99.5"].iloc[-1]),
            "cross_scr": float(cross["scr_cross"].iloc[-1]),
        }
        if available_capital is not None:
            report["capital_adequacy"] = self.capital_adequacy_ratio(
                available_capital, latest_scr)
        return report
