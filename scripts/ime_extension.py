"""Run the IME extension models on the real CSI 300 daily sample.

Produces the three-regime MS-GARCH, category-heterogeneity and unified
MSJ-GARCH estimates used by the revised IME manuscript, and writes them to
``docs/ime_extension_results.json`` for table generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PRICE_CSV = REPO / "data" / "raw" / "csi300_daily.csv"
OUT_JSON = REPO / "docs" / "ime_extension_results.json"


def load_returns() -> pd.Series:
    if not PRICE_CSV.exists():
        raise FileNotFoundError(
            f"Missing {PRICE_CSV}. Download CSI 300 daily data via "
            "akshare before running this script."
        )
    prices = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")
    close = prices["close"].astype(float)
    return np.log(close / close.shift(1)).dropna()


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


def run() -> dict:
    from actuarial.mortality import MortalityForecaster
    from model.category_heterogeneity import CategoryHeterogeneityTest
    from model.jump_garch import JumpGARCH
    from model.ms_garch import MarkovSwitchingGARCH
    from model.msj_garch import RegimeSwitchingJumpGARCH
    from model.sentiment_proxy import build_category_sentiment
    from volatility.garch import GARCHModel

    returns = load_returns()
    returns_pct = returns * 100.0
    categories = ["monetary", "industrial", "employment", "geopolitical"]
    sentiment = build_category_sentiment(returns, categories=categories)
    overall = sentiment.mean(axis=1)

    ms2 = MarkovSwitchingGARCH(n_regimes=2)
    r2 = ms2.fit(returns_pct, overall, n_iter=300)

    ms3 = MarkovSwitchingGARCH(n_regimes=3)
    r3 = ms3.fit(returns_pct, overall, n_iter=300)

    ms3_cat = MarkovSwitchingGARCH(n_regimes=3, category_names=categories)
    r3c = ms3_cat.fit(returns_pct, sentiment, n_iter=300)

    msj2 = RegimeSwitchingJumpGARCH(n_regimes=2, category_names=categories)
    rj = msj2.fit(returns_pct, sentiment, n_iter=150)

    het = CategoryHeterogeneityTest(categories=categories)
    het_table = het.estimate(returns_pct, sentiment)
    het_lr = het.likelihood_ratio(returns_pct, sentiment)

    garch = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal")
    garch_res = garch.fit(returns)
    jump = JumpGARCH()
    jump_res = jump.estimate(returns_pct.to_numpy())

    mortality = MortalityForecaster(line="non_pension_type1", gender="M")
    imp = mortality.improvement()
    g = imp["annual_improvement"].to_numpy(dtype=float)
    mu_g = float(g.mean())
    rmse_naive = float(np.sqrt(np.mean((g - mu_g) ** 2)))
    compare = mortality.compare()

    results = {
        "sample": {
            "n_observations": int(len(returns)),
            "start": str(returns.index[0].date()),
            "end": str(returns.index[-1].date()),
            "annualized_vol": float(returns.std() * np.sqrt(242)),
        },
        "models": {
            "garch": {
                "log_likelihood": garch_res["loglikelihood"],
                "aic": garch_res["aic"],
                "bic": garch_res["bic"],
            },
            "jump_garch": {
                "log_likelihood": jump_res["log_likelihood"],
                "aic": jump_res["aic"],
                "bic": jump_res["bic"],
            },
            "ms_garch_2reg": {
                "log_likelihood": r2["log_likelihood"],
                "aic": r2["aic"],
                "bic": r2["bic"],
                "params": r2["params"].tolist(),
                "multipliers": ms2.multipliers().tolist(),
                "transition": ms2.transition_matrix().to_dict("split"),
            },
            "ms_garch_3reg": {
                "log_likelihood": r3["log_likelihood"],
                "aic": r3["aic"],
                "bic": r3["bic"],
                "params": r3["params"].tolist(),
                "multipliers": ms3.multipliers().tolist(),
                "transition": ms3.transition_matrix().to_dict("split"),
            },
            "ms_garch_3reg_categories": {
                "log_likelihood": r3c["log_likelihood"],
                "aic": r3c["aic"],
                "bic": r3c["bic"],
                "params": r3c["params"].tolist(),
                "multipliers": ms3_cat.multipliers().to_dict("split"),
                "transition": ms3_cat.transition_matrix().to_dict("split"),
            },
            "msj_garch_2reg_categories": {
                "log_likelihood": rj["log_likelihood"],
                "aic": rj["aic"],
                "bic": rj["bic"],
                "params": rj["params"].tolist(),
                "jump_intensities": msj2.jump_intensities().to_dict("split"),
            },
        },
        "category_heterogeneity": {
            "table": het_table.to_dict("split"),
            "lr": het_lr,
        },
        "mortality": {
            "line": mortality.line,
            "gender": mortality.gender,
            "n_ages_used": int(len(imp)),
            "mean_annual_improvement": mu_g,
            "rmse_constant_improvement": rmse_naive,
            "compare_10yr": compare.to_dict("split"),
        },
    }
    OUT_JSON.write_text(
        json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def print_summary(results: dict) -> None:
    print("Sample:", results["sample"])
    for name, m in results["models"].items():
        print(f"{name}: ll={m['log_likelihood']:.2f} aic={m['aic']:.2f} bic={m['bic']:.2f}")
    print("Category LR:", results["category_heterogeneity"]["lr"])


if __name__ == "__main__":
    res = run()
    print_summary(res)
    print("Saved:", OUT_JSON)
