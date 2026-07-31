"""Mortality forecasting: Naive, Lee-Carter, GARCH-enhanced."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class MortalityForecaster:
    def __init__(self):
        self._data = None
    def _gen(self):
        np.random.seed(42); n = 50
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="YE")
        vol = np.zeros(n); vol[0] = 0.15
        for t in range(1, n): vol[t] = 0.05 + 0.3*vol[t-1] + 0.2*abs(np.random.randn())*0.1
        imp = -0.02 + vol * np.random.randn(n)
        return pd.DataFrame({"year": range(n), "date": dates,
            "mx": 0.008 * np.exp(np.cumsum(imp)), "imp": imp, "lm": np.log(0.008) + np.cumsum(imp)})
    def load(self):
        if self._data is None: self._data = self._gen()
        return self._data
    def naive(self, data=None, fy=10):
        if data is None: data = self.load()
        ai = data["imp"].mean()
        last = data["mx"].iloc[-1]
        return [last * np.exp(ai * t) for t in range(1, fy+1)]
    def lee_carter(self, data=None, fy=10):
        if data is None: data = self.load()
        kt = data["lm"] - data["lm"].mean()
        drift = np.diff(kt).mean()
        fc = np.exp(float(kt.iloc[-1]) + drift * np.arange(1, fy+1))
        rs = float(np.diff(kt).std())
        return fc, rs
    def lc_garch(self, data=None, fy=10):
        if data is None: data = self.load()
        kt = data["lm"] - data["lm"].mean()
        drift = float(np.diff(kt).mean())
        try:
            from arch import arch_model
            res = np.diff(kt) - drift
            m = arch_model(res*100, 1, 1, mean="zero", vol="GARCH", dist="normal").fit(update_freq=0, disp="off")
            fv = np.sqrt(m.forecast(horizon=fy).variance.iloc[-1].values) / 100
            np.random.seed(42); kfc = [float(kt.iloc[-1])]
            for vt in fv: kfc.append(kfc[-1] + drift + vt*float(np.random.randn()))
            return {"method": "GARCH-LC", "fc": np.exp(kfc[1:])}
        except Exception as e:
            logger.warning("GARCH failed: %s", e)
            fc,_ = self.lee_carter(data, fy)
            return {"method": "LC-fallback", "fc": fc}
    def compare(self, fy=10):
        data = self.load()
        n = self.naive(data, fy)
        lc, rs = self.lee_carter(data, fy)
        lcg = self.lc_garch(data, fy)
        return pd.DataFrame({"Model": ["Naive","Lee-Carter","GARCH-LC"],
            "10yr": [f"{n[-1]:.6f}", f"{lc[-1]:.6f}", f"{lcg['fc'][-1]:.6f}"],
            "Method": ["Constant","ARIMA(0,1,0)","GARCH(1,1)"]})
