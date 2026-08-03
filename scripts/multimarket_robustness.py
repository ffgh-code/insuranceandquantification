"""Robustness checks for the revised multi-market manuscript."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "data" / "raw" / "multimarket"
OUT_JSON = REPO / "docs" / "multimarket_robustness_results.json"


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


def load_market(market: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA_DIR / f"{market}_daily.csv", parse_dates=["date"]).set_index("date")
    close = df["close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    parkinson = 0.5 * (np.log(high / low) ** 2)
    return df, ret, parkinson.reindex(ret.index)


def build_sentiment(returns: pd.Series, measure: str = "overall") -> pd.Series:
    from model.sentiment_proxy import build_category_sentiment
    cats = build_category_sentiment(returns)
    if measure == "overall":
        return cats.mean(axis=1)
    if measure == "negative":
        return cats.mean(axis=1).clip(upper=0.0)
    if measure == "monetary":
        return cats["monetary"]
    return cats.mean(axis=1)


def run() -> dict:
    from model.ms_garch import MarkovSwitchingGARCH

    results = {}
    for market in ["csi300", "sp500", "stoxx50"]:
        _, ret, parkinson = load_market(market)
        r_pct = ret * 100.0
        seed = 200 + sum(ord(c) for c in market) % 700

        ms = MarkovSwitchingGARCH(n_regimes=2)
        ms.fit(r_pct, None, n_iter=300, seed=seed)
        h = ms.h
        probs = ms.filtered_probs
        regime_var = np.sum(probs * h, axis=1)
        corr_park = float(np.corrcoef(regime_var, parkinson.values)[0, 1])

        alt_sent = build_sentiment(ret, "negative")
        ms_alt = MarkovSwitchingGARCH(n_regimes=2)
        r_alt = ms_alt.fit(r_pct, alt_sent, n_iter=300, seed=seed + 1)

        half = len(ret) // 2
        ms_h1 = MarkovSwitchingGARCH(n_regimes=2)
        r_h1 = ms_h1.fit(r_pct.iloc[:half], alt_sent.iloc[:half], n_iter=300, seed=seed + 2)
        ms_h2 = MarkovSwitchingGARCH(n_regimes=2)
        r_h2 = ms_h2.fit(r_pct.iloc[half:], alt_sent.iloc[half:], n_iter=300, seed=seed + 3)

        results[market] = {
            "parkinson_corr": corr_park,
            "alternative_sentiment_negative": {
                "aic": r_alt["aic"],
                "bic": r_alt["bic"],
                "log_likelihood": r_alt["log_likelihood"],
            },
            "subsample_first_half": {
                "start": str(ret.index[0].date()),
                "end": str(ret.index[half - 1].date()),
                "aic": r_h1["aic"],
                "bic": r_h1["bic"],
                "log_likelihood": r_h1["log_likelihood"],
            },
            "subsample_second_half": {
                "start": str(ret.index[half].date()),
                "end": str(ret.index[-1].date()),
                "aic": r_h2["aic"],
                "bic": r_h2["bic"],
                "log_likelihood": r_h2["log_likelihood"],
            },
        }

    OUT_JSON.write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    res = run()
    for market, r in res.items():
        print(market, "parkinson_corr", round(r["parkinson_corr"], 4))
        print("  alt aic", round(r["alternative_sentiment_negative"]["aic"], 2))
        print("  h1 aic", round(r["subsample_first_half"]["aic"], 2))
        print("  h2 aic", round(r["subsample_second_half"]["aic"], 2))
    print("Saved:", OUT_JSON)
