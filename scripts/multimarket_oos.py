"""Rolling out-of-sample volatility forecasts for the revised paper.

Compares GARCH, GJR-GARCH, MS-GARCH and sentiment-augmented MS-GARCH on the
aligned CSI 300, S&P 500 and EURO STOXX 50 sample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA_CSV = REPO / "data" / "raw" / "multimarket" / "multimarket_daily.csv"
OUT_JSON = REPO / "docs" / "multimarket_oos_results.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def load_returns() -> dict[str, pd.Series]:
    panel = pd.read_csv(DATA_CSV, parse_dates=["date"]).set_index("date")
    out = {}
    for col in ["csi300", "sp500", "stoxx50"]:
        close = panel[col].astype(float)
        out[col] = np.log(close / close.shift(1)).dropna()
    return out


def build_sentiment(returns: pd.Series) -> pd.DataFrame:
    from model.sentiment_proxy import build_category_sentiment
    cats = build_category_sentiment(returns)
    overall = cats.mean(axis=1)
    return pd.DataFrame({
        "positive": overall.clip(lower=0.0),
        "negative": overall.clip(upper=0.0),
    })


def dm_test(loss1: np.ndarray, loss2: np.ndarray, h: int = 4) -> dict:
    d = loss1 - loss2
    T = len(d)
    mean_d = d.mean()
    resid = d - mean_d
    var = np.sum(resid ** 2) / T
    for lag in range(1, h + 1):
        cov = np.sum(resid[lag:] * resid[:-lag]) / T
        var += 2.0 * cov
    se = np.sqrt(max(var, 1e-12) / T)
    from scipy.stats import norm
    stat = mean_d / (se + 1e-12)
    return {"dm_stat": float(stat), "p_value": float(2.0 * (1.0 - norm.cdf(abs(stat))))}


def run() -> dict:
    from model.ms_garch import MarkovSwitchingGARCH
    from volatility.garch import GARCHModel

    returns = load_returns()
    window = 240
    step = 20
    results = {}

    for market, r_raw in returns.items():
        r_pct = r_raw * 100.0
        sentiment = build_sentiment(r_raw)
        overall = sentiment.mean(axis=1)
        actual = []
        preds = {name: [] for name in ["garch", "gjr", "ms", "ms_sent", "ms_pn"]}

        for t in range(window, len(r_raw) - 1, step):
            y_raw = r_raw.iloc[:t]
            y_pct = r_pct.iloc[:t]
            s = sentiment.iloc[:t]
            s_overall = overall.iloc[:t]
            actual.append(float(r_raw.iloc[t + 1] ** 2))

            garch = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal").fit(y_raw)
            gjr = GARCHModel(p=1, q=1, model_type="GJR-GARCH", distribution="normal").fit(y_raw)
            preds["garch"].append(float(garch["forecast_volatility"][0] ** 2))
            preds["gjr"].append(float(gjr["forecast_volatility"][0] ** 2))

            ms0 = MarkovSwitchingGARCH(n_regimes=2)
            ms0.fit(y_pct, None, n_iter=40)
            preds["ms"].append(ms0.forecast_variance(y_pct, None) / 10000.0)

            ms1 = MarkovSwitchingGARCH(n_regimes=2)
            ms1.fit(y_pct, s_overall, n_iter=40)
            preds["ms_sent"].append(ms1.forecast_variance(y_pct, s_overall) / 10000.0)

            ms2 = MarkovSwitchingGARCH(n_regimes=2, category_names=["positive", "negative"])
            ms2.fit(y_pct, s, n_iter=40)
            preds["ms_pn"].append(ms2.forecast_variance(y_pct, s) / 10000.0)

        actual = np.array(actual)
        metrics = {}
        for name, values in preds.items():
            p = np.maximum(np.array(values), 1e-10)
            rmse = float(np.sqrt(np.mean((actual - p) ** 2)))
            qlike = float(np.mean(actual / p - np.log(actual / p) - 1.0))
            metrics[name] = {"rmse": rmse, "qlike": qlike}
        loss_sent = (actual - np.array(preds["ms_sent"])) ** 2
        loss_pn = (actual - np.array(preds["ms_pn"])) ** 2
        loss_ms = (actual - np.array(preds["ms"])) ** 2
        metrics["dm_ms_sent_vs_ms"] = dm_test(loss_sent, loss_ms)
        metrics["dm_ms_pn_vs_ms"] = dm_test(loss_pn, loss_ms)
        results[market] = {"n_forecasts": int(len(actual)), "metrics": metrics}

    OUT_JSON.write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def print_summary(results: dict) -> None:
    for market, m in results.items():
        print(market, "n=", m["n_forecasts"])
        for name, v in m["metrics"].items():
            if name.startswith("dm_"):
                print(f"  {name}: stat={v['dm_stat']:.3f} p={v['p_value']:.3f}")
            else:
                print(f"  {name}: rmse={v['rmse']:.6f} qlike={v['qlike']:.4f}")


if __name__ == "__main__":
    res = run()
    print_summary(res)
    print("Saved:", OUT_JSON)
