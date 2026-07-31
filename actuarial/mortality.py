"""Mortality forecasting with real China Life Insurance Mortality Tables.

The module loads the official CL(2000-2003) and CL(2010-2013) industry life
tables from the bundled panel in ``data/raw/cl_mortality_panel.csv``. The
panel is disaggregated by gender and business line (non-pension type I/II and
pension), replacing the synthetic Lee-Carter-structured series used earlier.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CL_SOURCES = {
    2000: "China Life Insurance Mortality Table CL(2000-2003), CIRC Notice 保监发〔2005〕133号",
    2010: "China Life Insurance Mortality Table CL(2010-2013), CIRC Notice 保监发〔2016〕107号",
}


class MortalityForecaster:
    def __init__(self, panel_path=None, vintages=(2000, 2010),
                 line="pension", gender="M"):
        default_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "cl_mortality_panel.csv"
        self.panel_path = Path(panel_path) if panel_path else default_path
        self.vintages = tuple(vintages)
        self.line = line
        self.gender = gender
        self._panel = None

    def load(self) -> pd.DataFrame:
        """Load the real CL mortality panel by vintage/gender/business line."""
        if self._panel is None:
            if not self.panel_path.exists():
                raise FileNotFoundError(
                    f"Real CL mortality panel not found at {self.panel_path}. "
                    "Re-run data preparation or restore data/raw/cl_mortality_panel.csv."
                )
            self._panel = pd.read_csv(self.panel_path)
        return self._panel

    def load_qx(self, vintage=None, line=None, gender=None) -> pd.Series:
        panel = self.load()
        vintage = self.vintages[1] if vintage is None else vintage
        line = self.line if line is None else line
        gender = self.gender if gender is None else gender
        sub = panel[
            (panel["vintage"] == vintage)
            & (panel["line"] == line)
            & (panel["gender"] == gender)
        ]
        sub = sub.sort_values("age")
        return pd.Series(sub["qx"].to_numpy(dtype=float), index=sub["age"].astype(int))

    def improvement(self, line=None, gender=None) -> pd.DataFrame:
        """Annualized log-mortality improvement between the two CL vintages."""
        old = self.load_qx(self.vintages[0], line, gender)
        new = self.load_qx(self.vintages[1], line, gender)
        df = pd.DataFrame({
            "age": old.index,
            "qx_old": old.to_numpy(dtype=float),
            "qx_new": new.to_numpy(dtype=float),
        })
        df = df[(df["qx_old"] > 0) & (df["qx_new"] > 0) & (df["qx_old"] < 1) & (df["qx_new"] < 1)]
        years = self.vintages[1] - self.vintages[0]
        df["annual_improvement"] = (
            np.log(df["qx_old"]) - np.log(df["qx_new"])
        ) / years
        df["vintage_old"] = self.vintages[0]
        df["vintage_new"] = self.vintages[1]
        df["line"] = self.line if line is None else line
        df["gender"] = self.gender if gender is None else gender
        return df.reset_index(drop=True)

    def naive(self, data=None, fy=10, age=60):
        """Constant-improvement forecast from the latest real CL vintage."""
        if data is not None and isinstance(data, pd.Series):
            qx = data
        else:
            qx = self.load_qx()
        q0 = float(qx.get(age, qx.median()))
        g = float(self.improvement()["annual_improvement"].mean())
        return [q0 * np.exp(-g * t) for t in range(1, fy + 1)]

    def lee_carter(self, data=None, fy=10, age=60):
        """Two-vintage Lee-Carter-style forecast with drift improvement."""
        imp = self.improvement()
        qx = self.load_qx()
        q0 = float(qx.get(age, qx.median()))
        drift = -float(imp["annual_improvement"].mean())
        rs = float(imp["annual_improvement"].std())
        fc = np.array([q0 * np.exp(drift * t) for t in range(1, fy + 1)])
        return fc, rs

    def lc_garch(self, data=None, fy=10, age=60):
        """GARCH-enhanced mortality improvement forecast on real CL vintages."""
        imp = self.improvement()
        g = imp["annual_improvement"].to_numpy(dtype=float)
        drift = -float(g.mean())
        resid = g - g.mean()
        try:
            from arch import arch_model
            m = arch_model(resid * 1000.0, mean="zero", vol="GARCH",
                           p=1, q=1, dist="normal").fit(update_freq=0, disp="off")
            fv = np.sqrt(m.forecast(horizon=fy).variance.iloc[-1].to_numpy()) / 1000.0
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("GARCH mortality fallback: %s", exc)
            fv = np.full(fy, float(np.std(resid)))
        q0 = float(self.load_qx().get(age, self.load_qx().median()))
        rng = np.random.default_rng(42)
        kfc = [np.log(q0)]
        for vt in fv:
            kfc.append(kfc[-1] + drift + float(vt) * float(rng.normal()))
        return {"method": "GARCH-LC-CL", "fc": np.exp(kfc[1:]), "vol_path": fv}

    def compare(self, fy=10, age=60):
        data = self.load_qx()
        naive = self.naive(data, fy=fy, age=age)
        lc, rs = self.lee_carter(data, fy=fy, age=age)
        lcg = self.lc_garch(data, fy=fy, age=age)
        return pd.DataFrame({
            "Model": ["Naive", "Lee-Carter", "GARCH-LC"],
            "10yr": [f"{naive[-1]:.6f}", f"{lc[-1]:.6f}", f"{lcg['fc'][-1]:.6f}"],
            "Method": [
                "Constant improvement",
                "Two-vintage LC drift",
                "GARCH(1,1) improvement volatility",
            ],
        })

    def panel_summary(self) -> pd.DataFrame:
        panel = self.load()
        return panel.groupby(["vintage", "line", "gender"]).agg(
            n_ages=("age", "count"),
            mean_qx=("qx", "mean"),
            max_qx=("qx", "max"),
        ).reset_index()
