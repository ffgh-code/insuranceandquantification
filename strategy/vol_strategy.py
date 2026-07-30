"""Volatility-based trading strategies.

Implements strategies that use volatility predictions and sentiment
signals to generate trading decisions.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityStrategy:
    """Trading strategies based on volatility and sentiment signals.

    Supports multiple strategy types:
    - Volatility mean reversion (buy when vol is high, sell when low)
    - Sentiment-driven (long when sentiment > threshold)
    - Combined (sentiment-confirmed vol signals)
    - Volatility risk premium (short vol when it's expensive)
    """

    def __init__(
        self,
        strategy_type: Literal[
            "vol_mean_reversion",
            "sentiment_driven",
            "combined",
            "vol_risk_premium",
        ] = "combined",
        vol_lookback: int = 21,
        vol_percentile_high: float = 0.8,
        vol_percentile_low: float = 0.2,
        sentiment_threshold: float = 0.2,
    ):
        self.strategy_type = strategy_type
        self.vol_lookback = vol_lookback
        self.vol_percentile_high = vol_percentile_high
        self.vol_percentile_low = vol_percentile_low
        self.sentiment_threshold = sentiment_threshold

    def generate_signals(
        self,
        volatility: pd.Series,
        sentiment: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Generate trading signals based on strategy type.

        Args:
            volatility: Volatility series (e.g., GARCH conditional vol).
            sentiment: Optional sentiment score series.
            prices: Optional price series (used for vol risk premium).

        Returns:
            Series of trading signals: -1 (short), 0 (neutral), +1 (long).
        """
        if self.strategy_type == "vol_mean_reversion":
            return self._vol_mean_reversion_signals(volatility)
        elif self.strategy_type == "sentiment_driven":
            if sentiment is None:
                raise ValueError("Sentiment data required for sentiment_driven strategy")
            return self._sentiment_signals(sentiment)
        elif self.strategy_type == "combined":
            if sentiment is None:
                raise ValueError("Sentiment data required for combined strategy")
            return self._combined_signals(volatility, sentiment)
        elif self.strategy_type == "vol_risk_premium":
            return self._vol_risk_premium_signals(volatility, prices)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def _vol_mean_reversion_signals(self, volatility: pd.Series) -> pd.Series:
        """Volatility mean reversion: buy when vol is high, sell when low.

        Logic: When vol is in the top percentile, it tends to revert down
        (good for long positions). When vol is very low, it tends to spike
        (risk off, go short or flat).
        """
        rolling_high = volatility.rolling(self.vol_lookback).quantile(self.vol_percentile_high)
        rolling_low = volatility.rolling(self.vol_lookback).quantile(self.vol_percentile_low)

        signals = pd.Series(0.0, index=volatility.index)
        signals[volatility >= rolling_high] = 1.0  # High vol -> long (expect mean reversion down)
        signals[volatility <= rolling_low] = -1.0  # Low vol -> short (expect spike)
        return signals

    def _sentiment_signals(self, sentiment: pd.Series) -> pd.Series:
        """Pure sentiment-driven signals.

        Logic: Strong positive sentiment = long, strong negative = short.
        """
        signals = pd.Series(0.0, index=sentiment.index)
        signals[sentiment > self.sentiment_threshold] = 1.0
        signals[sentiment < -self.sentiment_threshold] = -1.0
        return signals

    def _combined_signals(
        self, volatility: pd.Series, sentiment: pd.Series
    ) -> pd.Series:
        """Combined volatility + sentiment signals.

        Logic: Sentiment confirms vol signals. Only trade when both
        vol regime and sentiment agree. This avoids false signals
        from purely mechanical vol rules.
        """
        vol_signals = self._vol_mean_reversion_signals(volatility)
        sent_signals = self._sentiment_signals(sentiment)

        # Only trade when both agree
        combined = pd.Series(0.0, index=volatility.index)
        combined[(vol_signals == 1.0) & (sent_signals == 1.0)] = 1.0
        combined[(vol_signals == -1.0) & (sent_signals == -1.0)] = -1.0

        # If only one signal is active, reduce position to 0.5
        combined[(vol_signals == 1.0) & (sent_signals >= 0)] = 0.5
        combined[(vol_signals == -1.0) & (sent_signals <= 0)] = -0.5

        return combined

    def _vol_risk_premium_signals(
        self, volatility: pd.Series, prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """Volatility risk premium strategy.

        Logic: Short vol (sell options) when implied vol > realized vol.
        This captures the vol risk premium. Uses simple heuristic:
        when short-term vol > long-term vol, vol is expensive.
        """
        short_vol = volatility.rolling(21).mean()
        long_vol = volatility.rolling(63).mean()

        signals = pd.Series(0.0, index=volatility.index)
        # Short vol when short-term > long-term (vol is expensive)
        signals[short_vol > long_vol * 1.2] = -1.0
        # Long vol when short-term < long-term (vol is cheap)
        signals[short_vol < long_vol * 0.8] = 1.0
        return signals

    def backtest(
        self,
        volatility: pd.Series,
        prices: pd.Series,
        sentiment: Optional[pd.Series] = None,
        initial_capital: float = 1_000_000,
    ) -> dict:
        """Run full backtest of the selected strategy.

        Args:
            volatility: Volatility series.
            prices: Asset price series.
            sentiment: Optional sentiment series.
            initial_capital: Starting capital.

        Returns:
            Dict with signals, metrics, and equity curve.
        """
        from .backtest import BacktestEngine

        signals = self.generate_signals(
            volatility=volatility, sentiment=sentiment, prices=prices
        )

        engine = BacktestEngine(initial_capital=initial_capital)
        metrics = engine.run(signals, prices)

        return {
            "signals": signals,
            "metrics": metrics,
            "equity_curve": metrics.equity_curve if metrics else None,
        }

    @staticmethod
    def compare_strategies(
        volatility: pd.Series,
        prices: pd.Series,
        sentiment: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compare all strategy types on the same data.

        Args:
            volatility: Volatility series.
            prices: Asset price series.
            sentiment: Optional sentiment series.

        Returns:
            DataFrame comparing Sharpe, return, max DD across strategies.
        """
        results = []
        strategy_types = ["vol_mean_reversion", "combined"]
        if sentiment is not None:
            strategy_types.append("sentiment_driven")
        strategy_types.append("vol_risk_premium")

        for stype in strategy_types:
            try:
                strat = VolatilityStrategy(strategy_type=stype)
                # Special handling for sentiment_driven
                if stype == "sentiment_driven" and sentiment is None:
                    continue
                result = strat.backtest(
                    volatility=volatility,
                    prices=prices,
                    sentiment=sentiment,
                )
                m = result["metrics"]
                results.append(
                    {
                        "Strategy": stype.replace("_", " ").title(),
                        "Sharpe": round(m.sharpe_ratio, 2) if m else 0,
                        "Return %": round(m.total_return_pct, 2) if m else 0,
                        "Max DD %": round(m.max_drawdown * 100, 2) if m else 0,
                        "Win Rate": f"{m.win_rate:.1%}" if m else "0%",
                        "Trades": m.total_trades if m else 0,
                    }
                )
            except Exception as e:
                logger.warning("Strategy %s failed: %s", stype, e)

        return pd.DataFrame(results).sort_values("Sharpe", ascending=False)
