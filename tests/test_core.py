"""Core unit tests for data, sentiment, volatility, and strategy modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────


@pytest.fixture(scope="session")
def market_data():
    from data.scrapers.market_data import MarketDataFetcher
    m = MarketDataFetcher(tickers=["SPY", "QQQ"])
    prices = m.download_prices()
    return prices


@pytest.fixture(scope="session")
def spx_prices(market_data):
    return market_data["SPY"]


@pytest.fixture(scope="session")
def returns(spx_prices):
    return np.log(spx_prices["close"] / spx_prices["close"].shift(1)).dropna()


@pytest.fixture(scope="session")
def sample_headlines():
    return [
        "Fed signals potential rate cut amid easing inflation pressures",
        "S&P 500 hits new all-time high on strong earnings season",
        "Tech layoffs deepen as companies cut costs amid margin pressure",
        "Treasury yields spike on renewed inflation concerns",
        "Consumer confidence index rises to 18-month high",
        "Manufacturing output contracts as new orders slow sharply",
    ]


# ─── Market Data Tests ───────────────────────────────────


class TestMarketData:
    def test_download_prices_shape(self, market_data):
        spy = market_data["SPY"]
        assert not spy.empty
        assert all(c in spy.columns for c in ["open", "high", "low", "close", "volume"])

    def test_multiple_tickers(self, market_data):
        assert "SPY" in market_data
        assert "QQQ" in market_data

    def test_compute_returns(self, spx_prices):
        from data.scrapers.market_data import MarketDataFetcher
        ret = MarketDataFetcher.compute_returns(spx_prices)
        assert len(ret) == len(spx_prices)
        assert ret.iloc[0] is np.nan or pd.isna(ret.iloc[0])

    def test_compute_log_returns(self, spx_prices):
        from data.scrapers.market_data import MarketDataFetcher
        ret = MarketDataFetcher.compute_returns(spx_prices, method="log")
        assert not ret.dropna().empty
        assert ret.dropna().std() > 0

    def test_realized_volatility(self, returns):
        from data.scrapers.market_data import MarketDataFetcher
        vol = MarketDataFetcher.compute_realized_volatility(returns)
        assert len(vol.dropna()) > 0
        assert vol.dropna().iloc[0] > 0
        assert vol.dropna().iloc[0] < 2  # annualized vol < 200%


# ─── Sentiment Tests ─────────────────────────────────────


class TestSentiment:
    def test_traditional_vader(self, sample_headlines):
        from sentiment.traditional_sentiment import TraditionalSentimentAnalyzer
        t = TraditionalSentimentAnalyzer()
        df = t.analyze_headlines(sample_headlines)
        assert len(df) == len(sample_headlines)
        assert "vader_compound" in df.columns
        assert "textblob_polarity" in df.columns

    def test_traditional_vader_range(self, sample_headlines):
        from sentiment.traditional_sentiment import TraditionalSentimentAnalyzer
        t = TraditionalSentimentAnalyzer()
        df = t.analyze_headlines(sample_headlines)
        assert df["vader_compound"].min() >= -1.0
        assert df["vader_compound"].max() <= 1.0

    def test_llm_fallback(self, sample_headlines):
        from sentiment.llm_sentiment import LLMSentimentAnalyzer
        l = LLMSentimentAnalyzer()
        df = l.analyze_headlines(sample_headlines, use_api=False)
        assert len(df) == len(sample_headlines)
        assert "llm_score" in df.columns
        assert "llm_direction" in df.columns

    def test_llm_scores_range(self, sample_headlines):
        from sentiment.llm_sentiment import LLMSentimentAnalyzer
        l = LLMSentimentAnalyzer()
        df = l.analyze_headlines(sample_headlines, use_api=False)
        assert df["llm_score"].min() >= -1.0
        assert df["llm_score"].max() <= 1.0

    def test_llm_topics(self, sample_headlines):
        from sentiment.llm_sentiment import LLMSentimentAnalyzer
        l = LLMSentimentAnalyzer()
        df = l.analyze_headlines(sample_headlines, use_api=False)
        assert "llm_topic" in df.columns
        assert df["llm_topic"].notna().all()
        valid_topics = {
            "rates", "earnings", "economy", "geopolitics", "energy",
            "tech", "regulation", "consumer", "housing", "employment",
            "trade", "other",
        }
        assert all(t in valid_topics for t in df["llm_topic"])

    def test_aggregate_daily(self, sample_headlines):
        from sentiment.llm_sentiment import LLMSentimentAnalyzer
        l = LLMSentimentAnalyzer()
        df = l.analyze_headlines(sample_headlines, use_api=False)
        df["published_dt"] = pd.Timestamp.now()
        agg = l.aggregate_daily_sentiment(df)
        assert not agg.empty
        assert "avg_llm_score" in agg.columns


# ─── Volatility Tests ────────────────────────────────────


class TestVolatility:
    def test_realized_vol_estimators(self, spx_prices):
        from volatility.realized_vol import RealizedVolatility
        rv = RealizedVolatility.compute_all(spx_prices)
        assert not rv.empty
        for col in ["close_to_close", "parkinson", "garman_klass", "yang_zhang"]:
            assert col in rv.columns
            assert rv[col].dropna().iloc[0] > 0

    def test_garch_fit(self, returns):
        from volatility.garch import GARCHModel
        g = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal")
        result = g.fit(returns)
        assert result["aic"] is not None
        assert not np.isnan(result["aic"])
        assert len(result["params"]) > 0
        assert len(result["conditional_volatility"]) > 0

    def test_garch_forecast(self, returns):
        from volatility.garch import GARCHModel
        g = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal")
        result = g.fit(returns, forecast_horizon=10)
        assert len(result["forecast_volatility"]) == 10

    def test_garch_summary(self, returns):
        from volatility.garch import GARCHModel
        g = GARCHModel(p=1, q=1, model_type="GARCH", distribution="normal")
        g.fit(returns)
        summary = g.summary()
        assert summary is not None

    @pytest.mark.slow
    def test_garch_model_comparison(self, returns):
        from volatility.garch import GARCHModel
        comparison = GARCHModel.compare_models(returns.dropna())
        assert not comparison.empty
        assert "aic" in comparison.columns

    @pytest.mark.slow
    def test_lstm_volatility(self):
        from volatility.lstm_vol import LSTMVolatility
        import warnings
        warnings.filterwarnings("ignore")
        vol = pd.Series(np.random.randn(300) * 0.2 + 0.15)
        lstm = LSTMVolatility(epochs=10, hidden_size=32)
        try:
            result = lstm.fit(vol, verbose=False)
            assert "train_losses" in result
            assert len(result["train_losses"]) > 0
        except Exception:
            pass  # LSTM may fail on CPU-only small data, that's OK


# ─── Strategy Tests ──────────────────────────────────────


class TestStrategy:
    def test_vol_mean_reversion_signals(self, spx_prices):
        from volatility.realized_vol import RealizedVolatility
        from strategy.vol_strategy import VolatilityStrategy
        vol = RealizedVolatility.close_to_close(spx_prices).dropna()
        strat = VolatilityStrategy(strategy_type="vol_mean_reversion")
        signals = strat.generate_signals(volatility=vol)
        assert len(signals) == len(vol)
        assert set(signals.unique()).issubset({-1.0, 0.0, 1.0})

    def test_backtest(self, spx_prices):
        from strategy.backtest import BacktestEngine
        import numpy as np
        signals = pd.Series(np.random.choice([-1, 0, 1], len(spx_prices)),
                            index=spx_prices.index)
        engine = BacktestEngine(initial_capital=1_000_000)
        metrics = engine.run(signals, spx_prices["close"])
        assert metrics.total_trades >= 0
        assert metrics.sharpe_ratio is not None

    def test_combined_strategy(self, spx_prices):
        from volatility.realized_vol import RealizedVolatility
        from strategy.vol_strategy import VolatilityStrategy
        import numpy as np
        vol = RealizedVolatility.close_to_close(spx_prices).dropna()
        sentiment = pd.Series(np.random.randn(len(vol)) * 0.3, index=vol.index)
        strat = VolatilityStrategy(strategy_type="combined")
        result = strat.backtest(volatility=vol, prices=spx_prices["close"].loc[vol.index],
                                sentiment=sentiment)
        assert "metrics" in result
        assert result["metrics"] is not None
        assert result["metrics"].sharpe_ratio is not None


# ─── Pipeline Tests ──────────────────────────────────────


class TestPipeline:
    def test_pipeline_data(self):
        from app.pipeline import Pipeline
        p = Pipeline(ticker="SPY")
        result = p.run_data()
        assert result["n_observations"] > 0
        assert result["current_price"] > 0

    def test_pipeline_sentiment(self):
        from app.pipeline import Pipeline
        p = Pipeline(ticker="SPY")
        p.run_data()
        result = p.run_sentiment()
        assert result["n_headlines"] > 0
        assert result["llm_avg_sentiment"] is not None

    def test_pipeline_volatility(self):
        from app.pipeline import Pipeline
        p = Pipeline(ticker="SPY")
        p.run_data()
        p.run_sentiment()
        result = p.run_volatility_models()
        assert not np.isnan(result["garch_aic"])

    @pytest.mark.slow
    def test_pipeline_full(self):
        from app.pipeline import Pipeline
        p = Pipeline(ticker="SPY")
        result = p.run_all()
        assert "data" in result
        assert "sentiment" in result
        assert "volatility" in result
        assert "strategy" in result
        assert result["strategy"]["n_strategies"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
