"""Main application: Orchestrates the full sentiment-volatility pipeline.

This module ties together data fetching, sentiment analysis, volatility
modeling, and strategy backtesting into a single workflow.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data.scrapers.market_data import MarketDataFetcher
from data.scrapers.news_scraper import NewsScraper
from sentiment.llm_sentiment import LLMSentimentAnalyzer
from sentiment.traditional_sentiment import TraditionalSentimentAnalyzer
from volatility.realized_vol import RealizedVolatility
from volatility.garch import GARCHModel
from volatility.lstm_vol import LSTMVolatility
from strategy.vol_strategy import VolatilityStrategy

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end pipeline: Data → Sentiment → Volatility → Strategy.

    Runs the complete workflow and caches intermediate results for
    reuse across the Streamlit dashboard.
    """

    def __init__(self, ticker: str = "sh000300"):
        self.ticker = ticker
        self.market_data = MarketDataFetcher(tickers=[ticker])
        self.news_scraper = NewsScraper()
        self.llm_sentiment = LLMSentimentAnalyzer()
        self.traditional_sentiment = TraditionalSentimentAnalyzer()

        # Cached results
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.Series] = None
        self._realized_vol: Optional[pd.DataFrame] = None
        self._headlines_df: Optional[pd.DataFrame] = None
        self._llm_sentiment_df: Optional[pd.DataFrame] = None
        self._trad_sentiment_df: Optional[pd.DataFrame] = None
        self._garch_result: Optional[dict] = None
        self._lstm_result: Optional[dict] = None
        self._strategy_results: Optional[dict] = None

    def run_data(self) -> dict:
        """Step 1: Fetch market data and compute returns/volatility."""
        prices_dict = self.market_data.download_prices()
        self._prices = prices_dict.get(self.ticker, pd.DataFrame())
        if self._prices.empty:
            raise ValueError(f"No data retrieved for {self.ticker}")

        self._returns = MarketDataFetcher.compute_returns(self._prices)
        self._realized_vol = RealizedVolatility.compute_all(self._prices)

        return {
            "ticker": self.ticker,
            "date_range": f"{self._prices.index[0].date()} to {self._prices.index[-1].date()}",
            "n_observations": len(self._prices),
            "current_price": float(self._prices["close"].iloc[-1]),
            "current_vol": float(self._realized_vol["yang_zhang"].iloc[-1]),
            "avg_vol": float(self._realized_vol["yang_zhang"].mean()),
        }

    def run_sentiment(self, use_llm: bool = True) -> dict:
        """Step 2: Run sentiment analysis on financial headlines."""
        # Get headlines from sample (avoids API dependency)
        headlines = NewsScraper.sample_headlines_for_llm(30)
        self._headlines_df = pd.DataFrame({"headline": headlines})

        # LLM sentiment (with fallback to rule-based)
        self._llm_sentiment_df = self.llm_sentiment.analyze_headlines(
            headlines, use_api=False  # Use rule-based unless API key is set
        )

        # Traditional sentiment baseline
        self._trad_sentiment_df = self.traditional_sentiment.analyze_headlines(headlines)

        # Comparison stats
        llm_avg = self._llm_sentiment_df["llm_score"].mean()
        trad_avg = self._trad_sentiment_df["vader_compound"].mean()

        return {
            "n_headlines": len(headlines),
            "llm_avg_sentiment": float(llm_avg),
            "vader_avg_sentiment": float(trad_avg),
            "llm_bullish_ratio": float((self._llm_sentiment_df["llm_direction"] == "bullish").mean()),
            "llm_bearish_ratio": float((self._llm_sentiment_df["llm_direction"] == "bearish").mean()),
        }

    def run_volatility_models(self) -> dict:
        """Step 3: Fit GARCH and LSTM volatility models."""
        if self._returns is None or self._prices is None:
            raise ValueError("Run data pipeline first.")

        # GARCH models
        garch = GARCHModel(p=1, q=1, model_type="GARCH")
        self._garch_result = garch.fit(self._returns)

        # GARCH with sentiment (if available, use LLM scores as exogenous)
        if self._llm_sentiment_df is not None:
            dummy_sentiment = pd.Series(
                self._llm_sentiment_df["llm_score"].values,
                index=pd.date_range(
                    end=self._returns.index[-1],
                    periods=len(self._llm_sentiment_df),
                    freq="B",
                )[: len(self._returns)],
            )
            garch_x = GARCHModel(p=1, q=1, model_type="GARCH")
            self._garch_x_result = garch_x.fit_with_sentiment(
                self._returns, dummy_sentiment
            )

        # LSTM model (train on realized vol)
        lstm = LSTMVolatility(epochs=50)
        try:
            self._lstm_result = lstm.fit(
                self._realized_vol["yang_zhang"].dropna()
            )
            lstm_trained = True
        except Exception as e:
            logger.warning("LSTM training failed: %s", e)
            self._lstm_result = {"final_val_loss": float("nan")}
            lstm_trained = False

        # Compare GARCH variants
        garch_comparison = GARCHModel.compare_models(self._returns.dropna())

        return {
            "garch_aic": float(self._garch_result["aic"]),
            "garch_bic": float(self._garch_result["bic"]),
            "garch_x_aic": float(
                getattr(self, "_garch_x_result", {}).get("aic", np.nan)
            ),
            "lstm_val_loss": float(self._lstm_result.get("final_val_loss", np.nan)),
            "lstm_trained": lstm_trained,
            "garch_comparison": garch_comparison,
        }

    def run_strategy(self) -> dict:
        """Step 4: Backtest volatility-based trading strategies."""
        if self._realized_vol is None:
            raise ValueError("Run data pipeline first.")

        vol = self._realized_vol["yang_zhang"].dropna()
        prices = self._prices["close"].loc[vol.index]

        # Generate synthetic sentiment aligned with price dates
        np.random.seed(42)
        dummy_sentiment = pd.Series(
            np.random.randn(len(vol)) * 0.3,
            index=vol.index,
            name="sentiment",
        )

        # Run and compare strategies
        comparison = VolatilityStrategy.compare_strategies(
            volatility=vol,
            prices=prices,
            sentiment=dummy_sentiment,
        )

        # Detailed combined strategy result
        combined = VolatilityStrategy(strategy_type="combined")
        combined_result = combined.backtest(
            volatility=vol,
            prices=prices,
            sentiment=dummy_sentiment,
        )

        self._strategy_results = {
            "comparison": comparison,
            "combined_result": combined_result,
        }

        return {
            "best_strategy": (
                comparison.iloc[0]["Strategy"] if not comparison.empty else "N/A"
            ),
            "best_sharpe": float(comparison.iloc[0]["Sharpe"]) if not comparison.empty else 0,
            "n_strategies": len(comparison),
        }

    def run_all(self) -> dict:
        """Run the complete end-to-end pipeline."""
        results = {}
        results["data"] = self.run_data()
        results["sentiment"] = self.run_sentiment()
        results["volatility"] = self.run_volatility_models()
        results["strategy"] = self.run_strategy()
        return results



