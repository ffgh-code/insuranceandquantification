"""Tests for refactored actuarial modules."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np; import pandas as pd; import pytest

class TestSolvency:
    def test_import(self):
        from actuarial.solvency import SolvencyCalculator; assert SolvencyCalculator
    def test_var(self):
        from actuarial.solvency import SolvencyCalculator
        v = SolvencyCalculator.calculate_var(0.20, 0.995)
        assert 0.01 < v < 0.10
    def test_cvar(self):
        from actuarial.solvency import SolvencyCalculator
        cv = SolvencyCalculator.calculate_cvar(0.20, 0.995)
        assert cv > 0
    def test_market_risk_scr(self):
        from actuarial.solvency import SolvencyCalculator
        sc = SolvencyCalculator(); sc.set_volatility_input(pd.Series(np.abs(np.random.randn(100))*0.1+0.05))
        df = sc.compute_market_risk_scr()
        assert not df.empty and "scr_solvency_ii" in df.columns
    def test_stress(self):
        from actuarial.solvency import SolvencyCalculator
        sc = SolvencyCalculator(); sc.set_volatility_input(pd.Series(np.abs(np.random.randn(100))*0.1+0.05))
        s = sc.stress_test_extreme_sentiment(1.5)
        assert not s.empty

class TestLoss:
    def test_import(self):
        from actuarial.loss_modeling import LossReserving; assert LossReserving
    def test_load(self):
        from actuarial.loss_modeling import LossReserving
        lr = LossReserving(); d = lr.load()
        assert len(d) == 40
    def test_compare(self):
        from actuarial.loss_modeling import LossReserving
        c = LossReserving().compare()
        assert len(c) == 2

class TestMortality:
    def test_import(self):
        from actuarial.mortality import MortalityForecaster; assert MortalityForecaster
    def test_load(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster(); d = mf.load()
        assert len(d) == 1060
        assert {"vintage", "age", "gender", "line", "qx"}.issubset(d.columns)
        assert set(d["vintage"].unique()) == {2000, 2010}
        assert set(d["line"].unique()) == {
            "non_pension_type1", "non_pension_type2", "pension"
        }
    def test_lee_carter(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster(); d = mf.load()
        fc, rs = mf.lee_carter(d)
        assert len(fc) == 10
    def test_lc_garch(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster(); d = mf.load()
        r = mf.lc_garch(d)
        assert "method" in r
    def test_compare(self):
        from actuarial.mortality import MortalityForecaster
        c = MortalityForecaster().compare()
        assert len(c) == 3
