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
            if isinstance(volatility, pd.DataFrame):
                vol_series = volatility.iloc[:, 0]
            else:
                vol_series = volatility
            r = float(np.asarray(vol_series.iloc[-1]).ravel()[0]) / max(
                float(np.asarray(vol_series.median()).ravel()[0]), 1e-6)
            dr = max(0.25, min(0.50, 0.35 * r))
            dynamic = last * dr
        else:
            dynamic = static
        return pd.DataFrame({"Plan": ["Static 35%", "Dynamic Vol-Adjusted"],
            "Reserve": [f"CNY {static:,.0f}", f"CNY {dynamic:,.0f}"]})
    def diff_chart(self, volatility=None):
        data = self.load()
        if data.empty: return pd.DataFrame()
        y = data.copy()
        y["year"] = y["date"].dt.year
        y = y.groupby("year").agg(total=("total_loss","sum")).reset_index()
        if volatility is not None and not volatility.empty:
            # Handle both Series and DataFrame volatility input
            if isinstance(volatility, pd.DataFrame):
                vol_series = volatility.iloc[:, 0]
            else:
                vol_series = volatility
            b = float(np.asarray(vol_series.median()).ravel()[0])
            # Convert volatility to DataFrame with year column for safe lookup
            vol_df = pd.DataFrame({"date": pd.to_datetime(vol_series.index)})
            vol_df["vol"] = np.asarray(vol_series.values).ravel()
            vol_df["year"] = vol_df["date"].dt.year
            yearly_vol = vol_df.groupby("year")["vol"].mean()

            def _dyn_ratio(yr):
                if yr in yearly_vol.index:
                    ratio = float(np.asarray(yearly_vol[yr]).ravel()[0]) / b
                    return max(0.25, min(0.50, 0.35 * ratio))
                return 0.35

            y["dr"] = y["year"].apply(_dyn_ratio)
            y["static"] = y.total * 0.35
            y["dynamic"] = y.total * y.dr
            y["diff"] = y.static - y.dynamic
        return y
