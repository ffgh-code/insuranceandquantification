"""Tests for IME revision modules: 3-regime MS-GARCH, MSJ-GARCH, categories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def returns_and_sentiment():
    rng = np.random.default_rng(7)
    n = 220
    r = np.zeros(n)
    h = np.ones(n) * 0.0004
    for t in range(1, n):
        if t > 100:
            h[t] = 0.0012
        if t > 170:
            h[t] = 0.003
        r[t] = rng.normal(0.0, np.sqrt(h[t]))
    sentiment = pd.DataFrame({
        "monetary": np.clip(rng.normal(0.0, 0.3, n), -1, 1),
        "industrial": np.clip(rng.normal(0.05, 0.3, n), -1, 1),
        "employment": np.clip(rng.normal(-0.02, 0.25, n), -1, 1),
        "geopolitical": np.clip(rng.normal(-0.1, 0.4, n), -1, 1),
    })
    return pd.Series(r), sentiment


class TestThreeRegimeMSGARCH:
    def test_fit_three_regimes(self, returns_and_sentiment):
        from model.ms_garch import MarkovSwitchingGARCH
        returns, sentiment = returns_and_sentiment
        model = MarkovSwitchingGARCH(n_regimes=3, category_names=list(sentiment.columns))
        result = model.fit(returns, sentiment, n_iter=20)
        assert np.isfinite(result["log_likelihood"])
        assert result["filtered_probs"].shape == (len(returns), 3)
        assert result["transition"].shape == (3, 3)
        assert model.regime_names == ["calm", "turbulent", "crisis"]
        assert result["aic"] < np.inf

    def test_category_multipliers(self, returns_and_sentiment):
        from model.ms_garch import MarkovSwitchingGARCH
        returns, sentiment = returns_and_sentiment
        model = MarkovSwitchingGARCH(n_regimes=3, category_names=list(sentiment.columns))
        model.fit(returns, sentiment, n_iter=20)
        m = model.multipliers()
        assert isinstance(m, pd.DataFrame)
        assert list(m.columns) == list(sentiment.columns)
        assert list(m.index) == model.regime_names


class TestMSJ:
    def test_fit_msj(self, returns_and_sentiment):
        from model.msj_garch import RegimeSwitchingJumpGARCH
        returns, sentiment = returns_and_sentiment
        model = RegimeSwitchingJumpGARCH(n_regimes=2, category_names=list(sentiment.columns))
        result = model.fit(returns, sentiment, n_iter=15)
        assert np.isfinite(result["log_likelihood"])
        assert len(result["jump_intensities"]) == 2
        assert result["jump_intensities"]["jump_intensity"].between(0, 1).all()


class TestCategoryHeterogeneity:
    def test_estimate(self, returns_and_sentiment):
        from model.category_heterogeneity import CategoryHeterogeneityTest
        returns, sentiment = returns_and_sentiment
        test = CategoryHeterogeneityTest(categories=list(sentiment.columns))
        table = test.estimate(returns, sentiment)
        assert list(table["category"]) == list(sentiment.columns)
        assert table["coefficient"].notna().all()

    def test_likelihood_ratio(self, returns_and_sentiment):
        from model.category_heterogeneity import CategoryHeterogeneityTest
        returns, sentiment = returns_and_sentiment
        test = CategoryHeterogeneityTest(categories=list(sentiment.columns))
        out = test.likelihood_ratio(returns, sentiment)
        assert set(out) >= {"lr_stat", "df", "p_value", "reject_homogeneity"}
        assert out["df"] == len(sentiment.columns) - 1


class TestRealMortalityPanel:
    def test_panel_loaded(self):
        from actuarial.mortality import MortalityForecaster
        panel = MortalityForecaster().load()
        assert len(panel) == 1060
        assert panel["qx"].between(0, 1).all()
        assert panel["source"].notna().all()

    def test_improvement_from_real_vintages(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster(line="non_pension_type1", gender="M")
        imp = mf.improvement()
        assert len(imp) >= 100
        assert imp["age"].max() >= 100
        assert imp["annual_improvement"].mean() > 0
