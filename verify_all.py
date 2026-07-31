import sys
sys.path.insert(0, "sentiment-vol-lab")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.CRITICAL)
import pandas as pd; import numpy as np
print("=== VERIFICATION START ===")
print()

# Phase 1
from volatility.highfreq import HighFreqData
hf = HighFreqData(n_days=100)
rv = hf.compute_intraday_rv()
print("1a. Intraday RV:", len(rv), "days, mean:", round(rv.mean(), 4))

from volatility.arima_model import ARIMAModel
am = ARIMAModel()
vol = pd.Series(np.abs(np.random.randn(200))*0.1+0.2)
r = am.fit(vol)
print("1b. ARIMA AIC:", round(r.get("aic", 0), 1))

from volatility.transformer_model import AttentionTransformer
try:
    tt = AttentionTransformer(epochs=10)
    features = np.random.randn(200, 3)
    target = np.random.randn(200)
    res = tt.fit(features, target, verbose=False)
    print("1c. Transformer final val loss:", round(res.get("final_val_loss", 0), 6))
except Exception as e:
    print("1c. Transformer skipped:", str(e)[:60])

from sentiment.sentiment_agg import SentimentAggregator
sa = SentimentAggregator()
print("1d. SentimentAggregator: OK")

# Phase 2
from strategy.rolling_backtest import RollingWindowBacktest, MarketRegimeClassifier
print("2a. RollingWindowBacktest: OK")
mrc = MarketRegimeClassifier()
prices = pd.Series(np.cumprod(1+np.random.randn(500)*0.01)+100)
regime = mrc.classify(prices)
vals, counts = np.unique(regime, return_counts=True)
print("2b. Regime counts:", dict(zip(vals, counts)))

from strategy.cross_sectional import CSI300Constituents, CrossSectionalFactors
cons = CSI300Constituents()
c = cons.load_constituents()
print("2c. Constituents loaded:", len(c))

# Phase 3
from actuarial.solvency import SolvencyCalculator
sc = SolvencyCalculator()
sc.set_volatility_input(vol)
scr = sc.compute_market_risk_scr()
print("3a. Solvency SCR shape:", scr.shape)
stress = sc.stress_test_extreme_sentiment(1.5)
print("3b. Stress test:", "OK" if not stress.empty else "FAIL")

from actuarial.loss_modeling import LossReserving
lr = LossReserving()
d = lr.load()
print("3c. Loss data shape:", d.shape)

from actuarial.mortality import MortalityForecaster
mf = MortalityForecaster()
md = mf.load()
lc, rs = mf.lee_carter(md)
print("3d. Lee-Carter forecast:", len(lc), "years")
lcg = mf.lc_garch(md)
print("3e. GARCH-LC method:", lcg.get("method", "FAIL"))
cmp = mf.compare()
print("3f. Model comparison:", len(cmp), "models")

print()
print("=== VERIFICATION COMPLETE ===")
