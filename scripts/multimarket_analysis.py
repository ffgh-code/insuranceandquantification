"""Multi-market analysis for the revised finance-journal manuscript.

Loads the aligned CSI 300, S&P 500 and EURO STOXX 50 daily sample, estimates
GARCH, GJR-GARCH and Markov-switching GARCH models with and without LLM-style
sentiment, and exports the results for the revised paper.
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
OUT_JSON = REPO / "docs" / "multimarket_results.json"


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="split")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def fit_models(market: str, returns: pd.Series) -> dict:
    from model.ms_garch import MarkovSwitchingGARCH
    from volatility.garch import GARCHModel

    r = returns * 100.0
    sentiment = build_sentiment(returns)
    overall = sentiment.mean(axis=1)

    garch = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal").fit(returns)
    gjr = GARCHModel(p=1, q=1, model_type="GJR-GARCH", distribution="normal").fit(returns)

    ms0 = MarkovSwitchingGARCH(n_regimes=2)
    r0 = ms0.fit(r, None, n_iter=500)

    ms1 = MarkovSwitchingGARCH(n_regimes=2)
    r1 = ms1.fit(r, overall, n_iter=500)

    ms2 = MarkovSwitchingGARCH(n_regimes=2, category_names=["positive", "negative"])
    r2 = ms2.fit(r, sentiment, n_iter=500)

    return {
        "market": market,
        "n_obs": int(len(returns)),
        "annualized_vol": float(returns.std() * np.sqrt(242)),
        "garch": {
            "log_likelihood": garch["loglikelihood"],
            "aic": garch["aic"],
            "bic": garch["bic"],
        },
        "gjr": {
            "log_likelihood": gjr["loglikelihood"],
            "aic": gjr["aic"],
            "bic": gjr["bic"],
        },
        "ms_garch": {
            "log_likelihood": r0["log_likelihood"],
            "aic": r0["aic"],
            "bic": r0["bic"],
            "params": r0["params"].tolist(),
            "transition": ms0.transition_matrix().to_dict("split"),
            "multipliers": ms0.multipliers().tolist(),
        },
        "ms_garch_sentiment": {
            "log_likelihood": r1["log_likelihood"],
            "aic": r1["aic"],
            "bic": r1["bic"],
            "params": r1["params"].tolist(),
            "transition": ms1.transition_matrix().to_dict("split"),
            "multipliers": ms1.multipliers().tolist(),
        },
        "ms_garch_pos_neg": {
            "log_likelihood": r2["log_likelihood"],
            "aic": r2["aic"],
            "bic": r2["bic"],
            "params": r2["params"].tolist(),
            "transition": ms2.transition_matrix().to_dict("split"),
            "multipliers": ms2.multipliers().to_dict("split"),
        },
    }


def run() -> dict:
    returns = load_returns()
    results = {}
    for market, r in returns.items():
        results[market] = fit_models(market, r)
    OUT_JSON.write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def print_summary(results: dict) -> None:
    for market, m in results.items():
        print(market, "n=", m["n_obs"], "vol=", round(m["annualized_vol"], 4))
        for name in ["garch", "gjr", "ms_garch", "ms_garch_sentiment", "ms_garch_pos_neg"]:
            mm = m[name]
            print(f"  {name}: aic={mm['aic']:.2f} bic={mm['bic']:.2f} ll={mm['log_likelihood']:.2f}")


if __name__ == "__main__":
    res = run()
    print_summary(res)
    print("Saved:", OUT_JSON)
