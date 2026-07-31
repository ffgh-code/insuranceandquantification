"""Dynamic loss reserving with volatility adjustment."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class LossReserving:
    def __init__(self):
        self._quarterly_data = None
    def _gen_synthetic(self):
        np.random.seed(42); n = 40
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="QE")
        trend = np.linspace(500, 800, n) * 1e8
        cycle = 50 * np.sin(np.linspace(0, 4*np.pi, n)) * 1e8
        total = trend + cycle + np.random.randn(n) * 30 * 1e8
        return pd.DataFrame({"date": dates, "total_loss": np.abs(total),
            "cumulative_paid": np.abs(np.cumsum(total * 0.7 / n))})
    def load(self):
        if self._quarterly_data is None: self._quarterly_data = self._gen_synthetic()
        return self._quarterly_data
    def compare(self, data=None, volatility=None):
        if data is None: data = self.load()
        if data.empty: return pd.DataFrame()
        last = data["total_loss"].iloc[-1]
        static = last * 0.35
        if volatility is not None and not volatility.empty:
            r = volatility.iloc[-1] / max(volatility.median(), 1e-6)
            dr = max(0.25, min(0.50, 0.35 * r))
            dynamic = last * dr
        else:
            dynamic = static
        return pd.DataFrame({"Plan": ["Static 35%", "Dynamic Vol-Adjusted"],
            "Reserve": [f"CNY {static:,.0f}", f"CNY {dynamic:,.0f}"]})
    def diff_chart(self, volatility=None):
        data = self.load()
        if data.empty: return pd.DataFrame()
        y = data.groupby(data["date"].dt.year).agg(total=("total_loss","sum")).reset_index()
        if volatility is not None and not volatility.empty:
            b = volatility.median()
            y["dr"] = y.year.apply(lambda yr: max(0.25, min(0.50, 0.35 * (
                volatility.loc[volatility.index.year == yr].mean() / b
                if not volatility.loc[volatility.index.year == yr].empty else 1.0))))
            y["static"] = y.total * 0.35
            y["dynamic"] = y.total * y.dr
            y["diff"] = y.static - y.dynamic
        return y
