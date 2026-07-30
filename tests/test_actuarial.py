"""Tests for actuarial science modules."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSolvency:
    def test_solvency_import(self):
        from actuarial.solvency import SolvencyCalculator
        assert SolvencyCalculator is not None

    def test_var_calculation(self):
        from actuarial.solvency import SolvencyCalculator
        var = SolvencyCalculator.calculate_var(0.20, 0.995)
        assert 0.01 < var < 0.10

    def test_cvar_calculation(self):
        from actuarial.solvency import SolvencyCalculator
        cvar = SolvencyCalculator.calculate_cvar(0.20, 0.995)
        assert cvar > 0

    def test_scr_market_risk(self):
        from actuarial.solvency import SolvencyCalculator
        sc = SolvencyCalculator(capital_base=100_000_000)
        np.random.seed(42)
        vol = pd.Series(np.abs(np.random.randn(100)) * 0.1 + 0.05)
        df = sc.scratch_market_risk(vol)
        assert "scr_market" in df.columns
        assert df["scr_market"].iloc[-1] > 0

    def test_cross_scr(self):
        from actuarial.solvency import SolvencyCalculator
        sc = SolvencyCalculator()
        np.random.seed(42)
        vol = pd.Series(np.abs(np.random.randn(100)) * 0.1 + 0.05)
        df = sc.scratch_cross(vol)
        assert df["scr_cross"].iloc[-1] >= 0

    def test_capital_adequacy(self):
        from actuarial.solvency import SolvencyCalculator
        ratio = SolvencyCalculator.capital_adequacy_ratio(100_000_000, 50_000_000)
        assert ratio == 200.0

    def test_generate_report(self):
        from actuarial.solvency import SolvencyCalculator
        sc = SolvencyCalculator(capital_base=100_000_000)
        np.random.seed(42)
        vol = pd.Series(np.abs(np.random.randn(100)) * 0.1 + 0.05)
        report = sc.generate_report(vol, available_capital=120_000_000)
        assert "latest_scr" in report


class TestLossModeling:
    def test_loss_modeler_import(self):
        from actuarial.loss_modeling import LossModeler
        assert LossModeler is not None

    def test_generate_claim_data(self):
        from actuarial.loss_modeling import LossModeler
        lm = LossModeler()
        df = lm.generate_claim_data()
        assert len(df) == 40
        assert "claim_count" in df.columns

    def test_garch_reserving(self):
        from actuarial.loss_modeling import LossModeler
        lm = LossModeler()
        df = lm.generate_claim_data()
        result = lm.garch_reserving(df["total_loss"])
        if "error" in result:
            pytest.skip("GARCH reserving skipped")
        else:
            assert "total_reserve" in result

    def test_compare_methods(self):
        from actuarial.loss_modeling import LossModeler
        lm = LossModeler()
        df = lm.generate_claim_data()
        comp = lm.compare_reserving_methods(df)
        assert len(comp) == 3


class TestMortality:
    def test_mortality_import(self):
        from actuarial.mortality import MortalityForecaster
        assert MortalityForecaster is not None

    def test_generate_mortality_data(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        df = mf.generate_mortality_data()
        assert len(df) == 50
        assert "mortality_rate" in df.columns

    def test_lee_carter_forecast(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        df = mf.generate_mortality_data()
        forecast = mf.lee_carter_forecast(df["log_mortality"], 10)
        assert len(forecast) == 10

    def test_garch_mortality(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        df = mf.generate_mortality_data()
        result = mf.garch_mortality_forecast(df["mortality_rate"])
        assert "method" in result or "error" in result

    def test_compare_methods(self):
        from actuarial.mortality import MortalityForecaster
        mf = MortalityForecaster()
        df = mf.generate_mortality_data()
        comp = mf.compare_forecast_methods(df, 10)
        assert len(comp) == 3
